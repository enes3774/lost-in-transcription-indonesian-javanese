"""
Pure-PyTorch re-implementation of Meta's Omnilingual ASR LLM-ASR models
(``omniASR_LLM_{300M,1B,3B,7B}``) with **no fairseq2 / fairseq2n dependency**.

Only needs: torch (>=2.4), and sentencepiece for detokenisation.

Architecture (read off the fairseq2 sources + checkpoint keys):

    waveform
      -> Wav2Vec2 feature extractor  (7 x conv1d + LayerNorm(fp32) + GELU)
      -> post_extract_layer_norm     (LN over 512)
      -> model_dim_proj              (512 -> D_enc)
      -> conv positional encoder     (weight-normed depthwise-ish conv, k=128, g=16, GELU, residual)
      -> N x pre-norm Transformer    (MHA with bias, GELU FFN)
      -> encoder.layer_norm
      -> encoder_proj                (D_enc -> D_dec)
      -> [audio embeds] + [<lid_marker>] + [lang embed] + [<bos>]
      -> 12 x LLaMA pre-norm decoder (RMSNorm, interleaved RoPE, SwiGLU) with KV cache
      -> final_proj -> greedy / beam decode -> sentencepiece detokenise

Numerically mirrors ``omnilingual_asr.models.wav2vec2_llama`` +
``Wav2Vec2LlamaBeamSearchSeq2SeqGenerator`` (nbest=1) in eval mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------


@dataclass
class OmniConfig:
    # encoder ---------------------------------------------------------------
    enc_dim: int = 2048
    enc_layers: int = 128
    enc_heads: int = 16
    enc_ffn_dim: int = 8192
    feat_dim: int = 512
    layer_descs: list = field(
        default_factory=lambda: [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
    )
    pos_conv_kernel: int = 128
    pos_conv_groups: int = 16
    # decoder ---------------------------------------------------------------
    dec_dim: int = 4096
    dec_layers: int = 12
    dec_heads: int = 8
    dec_kv_heads: int = 8
    dec_ffn_inner: int = 2816
    rope_theta: float = 10_000.0
    rms_eps: float = 1e-5
    max_seq_len: int = 8192
    # vocab -----------------------------------------------------------------
    vocab_size: int = 9818  # == final_proj rows
    n_text_embeds: int = 9819  # == text_frontend rows (vocab + n_special_tokens)
    n_langs: int = 1694
    bos_idx: int = 0
    pad_idx: int = 1
    eos_idx: int = 2
    unk_idx: int = 3

    @property
    def lid_marker(self) -> int:
        # Wav2Vec2LlamaSpecialTokens.lid_marker == target_vocab_info.size
        return self.vocab_size

    @staticmethod
    def from_state_dict(sd: dict) -> "OmniConfig":
        """Infer every shape-dependent field from the checkpoint itself."""
        cfg = OmniConfig()
        cfg.enc_dim = sd["encoder_frontend.model_dim_proj.weight"].shape[0]
        cfg.feat_dim = sd["encoder_frontend.model_dim_proj.weight"].shape[1]
        cfg.enc_layers = 1 + max(
            int(k.split(".")[2]) for k in sd if k.startswith("encoder.layers.")
        )
        cfg.enc_ffn_dim = sd["encoder.layers.0.ffn.inner_proj.weight"].shape[0]
        cfg.dec_dim = sd["encoder_proj.weight"].shape[0]
        cfg.dec_layers = 1 + max(
            int(k.split(".")[2]) for k in sd if k.startswith("llama_decoder.layers.")
        )
        cfg.dec_ffn_inner = sd["llama_decoder.layers.0.ffn.gate_proj.weight"].shape[0]
        cfg.vocab_size = sd["final_proj.weight"].shape[0]
        cfg.n_text_embeds = sd["text_frontend.weight"].shape[0]
        if "lang_embeddings.weight" in sd:
            cfg.n_langs = sd["lang_embeddings.weight"].shape[0]
        # head counts are not derivable from shapes; they are fixed per family
        cfg.enc_heads = 16 if cfg.enc_dim >= 1280 else 12
        cfg.dec_heads = 8
        cfg.dec_kv_heads = 8
        pk = "encoder_frontend.pos_encoder.conv.weight_v"
        if pk not in sd:
            pk = "encoder_frontend.pos_encoder.conv.weight"
        cfg.pos_conv_kernel = sd[pk].shape[-1]
        cfg.pos_conv_groups = cfg.enc_dim // sd[pk].shape[1]
        return cfg


# --------------------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------------------


class Fp32LayerNorm(nn.LayerNorm):
    """fairseq2 ``StandardLayerNorm(cast_fp32=True)``."""

    def forward(self, x: Tensor) -> Tensor:
        dt = x.dtype
        y = F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float(),
            self.bias.float(),
            self.eps,
        )
        return y.to(dt)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.normalized_shape = (dim,)

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)


class FeatureExtractionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel: int, stride: int):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel, stride=stride, bias=True)
        self.layer_norm = Fp32LayerNorm(out_dim, eps=1e-5)

    def forward(self, x: Tensor) -> Tensor:  # (N, C, S)
        x = self.conv(x)
        x = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)
        return F.gelu(x)


class FeatureExtractor(nn.Module):
    def __init__(self, layer_descs):
        super().__init__()
        self.layers = nn.ModuleList()
        in_dim = 1
        for out_dim, k, s in layer_descs:
            self.layers.append(FeatureExtractionLayer(in_dim, out_dim, k, s))
            in_dim = out_dim
        self.layer_descs = list(layer_descs)

    def forward(self, x: Tensor) -> Tensor:  # (N, S) -> (N, T, C)
        x = x.unsqueeze(1)
        for l in self.layers:
            x = l(x)
        return x.transpose(1, 2)

    def out_len(self, n: int) -> int:
        for _, k, s in self.layer_descs:
            n = math.floor((n - k) / s + 1.0)
        return n


class PosConvEncoder(nn.Module):
    """wav2vec 2.0 convolutional position encoder (weight norm folded at load time)."""

    def __init__(self, dim: int, kernel: int, groups: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=groups)
        self.remove_pad = kernel % 2 == 0

    def forward(self, x: Tensor, pad_mask: Tensor | None) -> Tensor:
        if pad_mask is not None:
            x = x.masked_fill(~pad_mask.unsqueeze(-1), 0.0)
        e = self.conv(x.transpose(1, 2))
        if self.remove_pad:
            e = e[:, :, :-1]
        e = F.gelu(e)
        return x + e.transpose(1, 2)


class EncoderAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.output_proj = nn.Linear(dim, dim, bias=True)
        self.heads = heads
        self.head_dim = dim // heads

    def forward(self, x: Tensor, attn_mask: Tensor | None) -> Tensor:
        n, s, _ = x.shape
        q = self.q_proj(x).view(n, s, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(n, s, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(n, s, self.heads, self.head_dim).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        o = o.transpose(1, 2).reshape(n, s, -1)
        return self.output_proj(o)


class EncoderFFN(nn.Module):
    def __init__(self, dim: int, inner: int):
        super().__init__()
        self.inner_proj = nn.Linear(dim, inner, bias=True)
        self.output_proj = nn.Linear(inner, dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.output_proj(F.gelu(self.inner_proj(x)))


class EncoderLayer(nn.Module):
    """pre-norm Transformer encoder layer."""

    def __init__(self, dim: int, heads: int, ffn_dim: int):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(dim, eps=1e-5)
        self.self_attn = EncoderAttention(dim, heads)
        self.ffn_layer_norm = nn.LayerNorm(dim, eps=1e-5)
        self.ffn = EncoderFFN(dim, ffn_dim)

    def forward(self, x: Tensor, attn_mask: Tensor | None) -> Tensor:
        x = x + self.self_attn(self.self_attn_layer_norm(x), attn_mask)
        x = x + self.ffn(self.ffn_layer_norm(x))
        return x


class EncoderFrontend(nn.Module):
    def __init__(self, cfg: OmniConfig):
        super().__init__()
        self.feature_extractor = FeatureExtractor(cfg.layer_descs)
        self.post_extract_layer_norm = nn.LayerNorm(cfg.feat_dim, eps=1e-5)
        self.model_dim_proj = nn.Linear(cfg.feat_dim, cfg.enc_dim, bias=True)
        self.pos_encoder = PosConvEncoder(
            cfg.enc_dim, cfg.pos_conv_kernel, cfg.pos_conv_groups
        )

    def forward(self, wav: Tensor, pad_mask: Tensor | None) -> Tensor:
        x = self.feature_extractor(wav)
        x = self.post_extract_layer_norm(x)
        x = self.model_dim_proj(x)
        x = self.pos_encoder(x, pad_mask)
        return x


class Encoder(nn.Module):
    def __init__(self, cfg: OmniConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(cfg.enc_dim, cfg.enc_heads, cfg.enc_ffn_dim)
            for _ in range(cfg.enc_layers)
        )
        self.layer_norm = nn.LayerNorm(cfg.enc_dim, eps=1e-5)

    def forward(self, x: Tensor, attn_mask: Tensor | None) -> Tensor:
        for l in self.layers:
            x = l(x, attn_mask)
        return self.layer_norm(x)


# ------------------------------- llama decoder ----------------------------------------


def _rope_tables(head_dim: int, max_len: int, theta: float, device, dtype=torch.float32):
    idx = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    freqs = 1.0 / (theta ** (idx / head_dim))
    steps = torch.arange(max_len, device=device, dtype=torch.float32)
    ang = torch.outer(steps, freqs)  # (S, D/2)
    return torch.cos(ang).to(dtype), torch.sin(ang).to(dtype)


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x: (N, H, S, Dh); cos/sin: (S, Dh/2). Interleaved (fairseq2/LLaMA original)."""
    dt = x.dtype
    xf = x.float().unflatten(-1, (-1, 2))
    x0, x1 = xf[..., 0], xf[..., 1]
    c = cos.unsqueeze(0).unsqueeze(0)
    s = sin.unsqueeze(0).unsqueeze(0)
    o0 = x0 * c - x1 * s
    o1 = x0 * s + x1 * c
    return torch.stack((o0, o1), dim=-1).flatten(-2).to(dt)


class KVCache:
    def __init__(self, n_layers, batch, heads, head_dim, max_len, device, dtype):
        self.k = [
            torch.zeros(batch, heads, max_len, head_dim, device=device, dtype=dtype)
            for _ in range(n_layers)
        ]
        self.v = [
            torch.zeros(batch, heads, max_len, head_dim, device=device, dtype=dtype)
            for _ in range(n_layers)
        ]
        self.len = 0


class DecoderAttention(nn.Module):
    def __init__(self, dim: int, heads: int, kv_heads: int):
        super().__init__()
        self.head_dim = dim // heads
        self.heads = heads
        self.kv_heads = kv_heads
        self.q_proj = nn.Linear(dim, heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(heads * self.head_dim, dim, bias=False)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache: KVCache | None,
        layer_idx: int,
        is_causal: bool,
    ) -> Tensor:
        n, s, _ = x.shape
        q = self.q_proj(x).view(n, s, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(n, s, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(n, s, self.kv_heads, self.head_dim).transpose(1, 2)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if cache is not None:
            start = cache.len
            cache.k[layer_idx][:, :, start : start + s] = k
            cache.v[layer_idx][:, :, start : start + s] = v
            k = cache.k[layer_idx][:, :, : start + s]
            v = cache.v[layer_idx][:, :, : start + s]

        if self.kv_heads != self.heads:
            rep = self.heads // self.kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        o = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        o = o.transpose(1, 2).reshape(n, s, -1)
        return self.output_proj(o)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, inner: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, inner, bias=False)
        self.inner_proj = nn.Linear(dim, inner, bias=False)
        self.output_proj = nn.Linear(inner, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.output_proj(F.silu(self.gate_proj(x)) * self.inner_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: OmniConfig):
        super().__init__()
        self.self_attn_layer_norm = RMSNorm(cfg.dec_dim, cfg.rms_eps)
        self.self_attn = DecoderAttention(cfg.dec_dim, cfg.dec_heads, cfg.dec_kv_heads)
        self.ffn_layer_norm = RMSNorm(cfg.dec_dim, cfg.rms_eps)
        self.ffn = SwiGLU(cfg.dec_dim, cfg.dec_ffn_inner)

    def forward(self, x, cos, sin, cache, idx, is_causal):
        x = x + self.self_attn(
            self.self_attn_layer_norm(x), cos, sin, cache, idx, is_causal
        )
        x = x + self.ffn(self.ffn_layer_norm(x))
        return x


class LlamaDecoder(nn.Module):
    def __init__(self, cfg: OmniConfig):
        super().__init__()
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.dec_layers))
        self.layer_norm = RMSNorm(cfg.dec_dim, cfg.rms_eps)

    def forward(self, x, cos, sin, cache, is_causal):
        for i, l in enumerate(self.layers):
            x = l(x, cos, sin, cache, i, is_causal)
        return self.layer_norm(x)


# --------------------------------------------------------------------------------------
# full model
# --------------------------------------------------------------------------------------


class OmniASRLlama(nn.Module):
    def __init__(self, cfg: OmniConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder_frontend = EncoderFrontend(cfg)
        self.encoder = Encoder(cfg)
        self.encoder_proj = nn.Linear(cfg.enc_dim, cfg.dec_dim, bias=True)
        self.text_frontend = nn.Embedding(cfg.n_text_embeds, cfg.dec_dim)
        self.llama_decoder = LlamaDecoder(cfg)
        self.final_proj = nn.Linear(cfg.dec_dim, cfg.vocab_size, bias=False)
        self.lang_embeddings = nn.Embedding(cfg.n_langs, cfg.dec_dim)
        self._rope = None

    # -- rope cache ---------------------------------------------------------
    def rope(self, device, n: int):
        head_dim = self.cfg.dec_dim // self.cfg.dec_heads
        if self._rope is None or self._rope[0].shape[0] < n:
            self._rope = _rope_tables(
                head_dim, max(n, 4096), self.cfg.rope_theta, device
            )
        return self._rope

    # -- audio --------------------------------------------------------------
    @torch.inference_mode()
    def encode_audio(self, wavs: list[Tensor]) -> tuple[Tensor, list[int]]:
        """wavs: list of 1-D float tensors (already normalised). Returns padded
        (N, T, D_dec) embeddings and their true lengths."""
        device = self.encoder_proj.weight.device
        dtype = self.encoder_proj.weight.dtype
        lens = [w.numel() for w in wavs]
        maxlen = max(lens)
        batch = torch.zeros(len(wavs), maxlen, device=device, dtype=dtype)
        for i, w in enumerate(wavs):
            batch[i, : w.numel()] = w.to(device=device, dtype=dtype)

        fe = self.encoder_frontend.feature_extractor
        out_lens = [fe.out_len(l) for l in lens]
        t_max = fe.out_len(maxlen)

        pad_mask = None
        attn_mask = None
        if len(set(out_lens)) > 1:
            ar = torch.arange(t_max, device=device)
            pad_mask = ar.unsqueeze(0) < torch.tensor(
                out_lens, device=device
            ).unsqueeze(1)
            attn_mask = pad_mask[:, None, None, :]

        x = self.encoder_frontend(batch, pad_mask)
        x = self.encoder(x, attn_mask)
        x = self.encoder_proj(x)
        return x, out_lens

    @torch.inference_mode()
    def encode_bottom(self, wavs: list[Tensor], n_shared: int):
        """Front end + the first ``n_shared`` encoder layers. Every fine-tuned variant keeps these
        layers frozen, so their output can be computed once and reused for all variants."""
        device = self.encoder_proj.weight.device
        dtype = self.encoder_proj.weight.dtype
        lens = [w.numel() for w in wavs]
        maxlen = max(lens)
        batch = torch.zeros(len(wavs), maxlen, device=device, dtype=dtype)
        for i, w in enumerate(wavs):
            batch[i, : w.numel()] = w.to(device=device, dtype=dtype)
        fe = self.encoder_frontend.feature_extractor
        out_lens = [fe.out_len(l) for l in lens]
        t_max = fe.out_len(maxlen)
        pad_mask = None
        attn_mask = None
        if len(set(out_lens)) > 1:
            ar = torch.arange(t_max, device=device)
            pad_mask = ar.unsqueeze(0) < torch.tensor(out_lens, device=device).unsqueeze(1)
            attn_mask = pad_mask[:, None, None, :]
        x = self.encoder_frontend(batch, pad_mask)
        for l in self.encoder.layers[:n_shared]:
            x = l(x, attn_mask)
        return x, out_lens, attn_mask

    @torch.inference_mode()
    def encode_top(self, x: Tensor, attn_mask: Tensor | None, n_shared: int) -> Tensor:
        """Remaining encoder layers + final norm + projection (the part fine-tuning changes)."""
        for l in self.encoder.layers[n_shared:]:
            x = l(x, attn_mask)
        x = self.encoder.layer_norm(x)
        return self.encoder_proj(x)

    # -- text ---------------------------------------------------------------
    def embed_tokens(self, ids: Tensor) -> Tensor:
        return self.text_frontend(ids)

    # -- generation ---------------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        wavs: list[Tensor],
        lang_ids: list[int] | None = None,
        max_new_tokens: int = 512,
        use_lang_embed: bool = True,
        audio: tuple[Tensor, list[int]] | None = None,
    ) -> list[list[int]]:
        """Greedy decoding, mirroring the nbest=1 path of
        ``Wav2Vec2LlamaBeamSearchSeq2SeqGenerator``."""
        cfg = self.cfg
        device = self.encoder_proj.weight.device
        dtype = self.encoder_proj.weight.dtype
        if audio is None:
            audio, audio_lens = self.encode_audio(wavs)
        else:  # embeddings from encode_bottom() + encode_top()
            audio, audio_lens = audio
        b = audio.size(0)

        # ---- assemble the prompt: audio | <lid_marker> | lang | <bos>
        extra = []
        if use_lang_embed:
            lid = self.embed_tokens(
                torch.full((b, 1), cfg.lid_marker, dtype=torch.long, device=device)
            )
            li = torch.tensor(
                lang_ids if lang_ids is not None else [0] * b,
                dtype=torch.long,
                device=device,
            ).unsqueeze(1)
            lang = self.lang_embeddings(li).to(dtype)
            extra += [lid, lang]
        bos = self.embed_tokens(
            torch.full((b, 1), cfg.bos_idx, dtype=torch.long, device=device)
        )
        extra.append(bos)
        n_extra = len(extra)
        extra = torch.cat(extra, dim=1).to(dtype)

        ctx_lens = [l + n_extra for l in audio_lens]
        max_ctx = max(ctx_lens)
        total = min(cfg.max_seq_len, max_ctx + max_new_tokens)

        inputs = torch.zeros(b, total, cfg.dec_dim, device=device, dtype=dtype)
        for i in range(b):
            inputs[i, : audio_lens[i]] = audio[i, : audio_lens[i]]
            inputs[i, audio_lens[i] : ctx_lens[i]] = extra[i]

        cache = KVCache(
            cfg.dec_layers,
            b,
            cfg.dec_kv_heads,
            cfg.dec_dim // cfg.dec_heads,
            total,
            device,
            dtype,
        )
        cos, sin = self.rope(device, total)

        # ---- prefill up to the shortest context (minus the BOS step)
        min_ctx = min(ctx_lens) - 1
        _ = self.llama_decoder(
            inputs[:, :min_ctx], cos[:min_ctx], sin[:min_ctx], cache, True
        )
        cache.len = min_ctx

        ctx_t = torch.tensor(ctx_lens, device=device)
        out = torch.full((b, total), cfg.pad_idx, dtype=torch.long, device=device)
        finished = torch.zeros(b, dtype=torch.bool, device=device)

        t = min_ctx
        while True:
            h = self.llama_decoder(
                inputs[:, t : t + 1], cos[t : t + 1], sin[t : t + 1], cache, False
            )
            cache.len = t + 1
            logits = self.final_proj(h[:, -1])
            nxt = logits.argmax(-1)

            nxt = torch.where(
                finished, torch.full_like(nxt, cfg.eos_idx), nxt
            )
            not_yet = t < ctx_t - 1  # still consuming its own (longer) context
            nxt = torch.where(not_yet, torch.full_like(nxt, cfg.pad_idx), nxt)
            out[:, t] = nxt
            finished = finished | (nxt == cfg.eos_idx)

            t += 1
            if bool(finished.all()) or t >= total - 1:
                break

            emb = self.embed_tokens(nxt).to(dtype)
            write = ~not_yet
            if write.any():
                inputs[write, t] = emb[write]

        res = []
        for i in range(b):
            toks = out[i].tolist()
            seq = [
                x
                for x in toks
                if x not in (cfg.pad_idx, cfg.bos_idx, cfg.eos_idx)
            ]
            res.append(seq)
        return res


# --------------------------------------------------------------------------------------
# checkpoint loading
# --------------------------------------------------------------------------------------


def _fold_weight_norm(sd: dict) -> None:
    """conv weight_g/weight_v (dim=2 weight norm) -> plain weight."""
    g = sd.pop("encoder_frontend.pos_encoder.conv.weight_g", None)
    v = sd.pop("encoder_frontend.pos_encoder.conv.weight_v", None)
    if g is None:
        return
    g = g.float()
    v = v.float()
    norm = v.norm(dim=(0, 1), keepdim=True)
    sd["encoder_frontend.pos_encoder.conv.weight"] = (v * (g / norm)).to(v.dtype)


def load_omni_asr(
    path: str,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> OmniASRLlama:
    """Load an ``omniASR-LLM-*.pt`` fairseq2 checkpoint into the pure-torch model."""
    obj = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    sd = dict(sd)
    sd.pop("masker.temporal_mask_embed", None)
    _fold_weight_norm(sd)

    cfg = OmniConfig.from_state_dict(sd)
    with torch.device("meta"):
        model = OmniASRLlama(cfg)
    model.load_state_dict(sd, strict=True, assign=True)
    model.eval()
    return model.to(device=device, dtype=dtype)


def save_converted(model: OmniASRLlama, path: str) -> None:
    torch.save({"cfg": model.cfg.__dict__, "model": model.state_dict()}, path)


def load_converted(
    path: str, device="cuda", dtype: torch.dtype = torch.bfloat16
) -> OmniASRLlama:
    try:
        obj = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except Exception:
        obj = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    cfg = OmniConfig(**obj["cfg"])
    model = OmniASRLlama(cfg)
    model.load_state_dict(obj["model"], strict=True, assign=True)
    model.eval()
    return model.to(device=device, dtype=dtype)


# --------------------------------------------------------------------------------------
# audio helpers
# --------------------------------------------------------------------------------------


def normalize_waveform(w: Tensor) -> Tensor:
    """omnilingual_asr ``apply_audio_normalization``: layer_norm over the whole clip."""
    return F.layer_norm(w.float(), w.shape)


def split_long(w, sr: int = 16000, maxs: float = 34.0):
    """Split at the quietest frame inside the middle of the clip, recursively."""
    import numpy as np

    if len(w) <= maxs * sr:
        return [w]
    lo, hi = int(len(w) * 0.35), int(len(w) * 0.65)
    fr = 512
    e = np.array([np.abs(w[i : i + fr]).mean() for i in range(lo, hi, fr)])
    cut = lo + int(np.argmin(e)) * fr
    return split_long(w[:cut], sr, maxs) + split_long(w[cut:], sr, maxs)
