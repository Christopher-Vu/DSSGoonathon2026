"""SF World Model: pre-LN transformer over a (cell, year) token set, MTM + next-year heads."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CELL_ID, MASK_ID, N_FEAT, N_SPECIAL, PAD_ID, TEXT_DIM


class Fourier2D(nn.Module):
    """Fixed 2-D sinusoidal code of the in-cell offset, then a learned projection."""

    def __init__(self, n_freq: int, d: int):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(n_freq)) * math.pi, persistent=False)
        self.proj = nn.Linear(4 * n_freq, d)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        a = xy[..., :, None] * self.freqs  # [..., 2, F]
        code = torch.cat([a.sin(), a.cos()], dim=-1).flatten(-2)
        return self.proj(code)


class TokenEmbed(nn.Module):
    def __init__(self, n_vocab: int, d: int, n_fourier: int, n_years: int, year_base: int):
        super().__init__()
        self.year_base, self.n_years = year_base, n_years
        self.type_emb = nn.Embedding(n_vocab, d)
        self.feat_mlp = nn.Sequential(nn.Linear(2 * N_FEAT, d), nn.GELU(), nn.Linear(d, d))
        self.text_proj = nn.Linear(TEXT_DIM, d)
        self.pos = Fourier2D(n_fourier, d)
        self.year_emb = nn.Embedding(n_years, d)

    def forward(self, type_ids, feats, text_vec, has_text, xy, year, masked):
        real = type_ids >= N_SPECIAL
        keep = (real & ~masked).unsqueeze(-1).float()  # content visible
        isnan = torch.isnan(feats)
        f_in = torch.cat([torch.nan_to_num(feats, nan=0.0), isnan.float()], dim=-1)
        shown_type = torch.where(masked, torch.full_like(type_ids, MASK_ID), type_ids)
        h = self.type_emb(shown_type)
        h = h + self.feat_mlp(f_in) * keep
        if text_vec is not None:
            h = h + self.text_proj(text_vec) * (keep * has_text.unsqueeze(-1).float())
        h = h + self.pos(xy) * real.unsqueeze(-1).float()  # position kept even when masked
        yi = (year - self.year_base).clamp(0, self.n_years - 1)
        h = h + self.year_emb(yi).unsqueeze(1)  # year kept even when masked
        return h


class WorldModel(nn.Module):
    def __init__(self, n_types: int, d_model=256, n_layers=6, n_heads=8, d_ff=1024, dropout=0.1,
                 n_fourier=8, year_base=2000, n_years=40, **_):
        super().__init__()
        self.n_types, self.d = n_types, d_model
        self.embed = TokenEmbed(N_SPECIAL + n_types, d_model, n_fourier, n_years, year_base)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.type_head = nn.Linear(d_model, n_types)
        self.feat_head = nn.Linear(d_model, n_types * N_FEAT)
        self.next_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, n_types * (1 + N_FEAT)))
        # parcel next-year head: parcel token output -> log-rates of (sale, buyout tenants, block-face Ellis/OMI) at t+1
        self.parcel_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 3))

    def forward(self, batch: dict, masked: torch.Tensor | None = None, text_table: torch.Tensor | None = None):
        type_ids = batch["type_ids"]
        if masked is None:
            masked = torch.zeros_like(type_ids, dtype=torch.bool)
        text_vec, has_text = None, None
        if text_table is not None:
            ti = batch["text_idx"]
            has_text = ti >= 0
            text_vec = text_table[ti.clamp(min=0)].float()
        h = self.embed(type_ids, batch["feats"], text_vec, has_text, batch["xy"], batch["year"], masked)
        h = self.encoder(h, src_key_padding_mask=batch["pad"])
        return self.norm(h)

    # ------------------------------------------------------------ losses
    def mtm_loss(self, h, batch, masked):
        if masked.sum() == 0:
            return h.sum() * 0.0, {}
        hm = h[masked]  # [M,d]
        tgt_type = batch["type_ids"][masked] - N_SPECIAL  # [M]
        ce = F.cross_entropy(self.type_head(hm).float(), tgt_type)
        pf = self.feat_head(hm).float().view(-1, self.n_types, N_FEAT)
        pf = pf[torch.arange(len(tgt_type), device=h.device), tgt_type]  # true type's slice
        tf = batch["feats"][masked]
        v = ~torch.isnan(tf)
        mse = ((pf - torch.nan_to_num(tf)) ** 2)[v].mean() if v.any() else pf.sum() * 0.0
        return ce + mse, {"mtm_ce": ce.detach(), "mtm_mse": mse.detach()}

    def next_pred(self, h_cell):
        out = self.next_head(h_cell).float()
        log_rate = out[:, : self.n_types]
        mu = out[:, self.n_types :].view(-1, self.n_types, N_FEAT)
        return log_rate, mu

    def next_loss(self, h, batch, parcel_weight: float = 1.0):
        hn = batch["has_next"]
        if hn.sum() == 0:
            return h.sum() * 0.0, {}
        log_rate, mu = self.next_pred(h[hn, 0])
        counts = batch["next_counts"][hn]
        pois = F.poisson_nll_loss(log_rate, counts, log_input=True, full=True)
        means = batch["next_means"][hn]
        v = ~torch.isnan(means) & (counts > 0).unsqueeze(-1)  # masked where the type is absent
        mse = ((mu - torch.nan_to_num(means)) ** 2)[v].mean() if v.any() else mu.sum() * 0.0
        loss, parts = pois + mse, {"next_pois": pois.detach(), "next_mse": mse.detach()}
        if "pnext" in batch and parcel_weight > 0:
            lp, per = self.parcel_loss(h, batch["pnext"])
            loss = loss + parcel_weight * lp
            parts.update({"parcel": lp.detach(), **per})
        return loss, parts

    def parcel_pred(self, h_tok):
        return self.parcel_head(h_tok).float()  # log-rates [*, 3]

    def parcel_loss(self, h, pnext):
        """Poisson NLL per target (mean over parcels where it is defined, log(y!) dropped), summed over targets."""
        v = ~torch.isnan(pnext)
        if not v.any():
            return h.sum() * 0.0, {}
        rows = v.any(-1)
        lr = self.parcel_pred(h[rows])
        y, vv = torch.nan_to_num(pnext[rows]), v[rows]
        nll = torch.exp(lr) - y * lr
        tot, per = h.sum() * 0.0, {}
        for k in range(3):
            if vv[:, k].any():
                lk = nll[vv[:, k], k].mean()
                tot = tot + lk
                per[f"parcel_{k}"] = lk.detach()
        return tot, per


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def sample_mask(type_ids: torch.Tensor, pad: torch.Tensor, lo: float, hi: float, gen: torch.Generator) -> torch.Tensor:
    """Mask 15-30% of real tokens per sequence (at least one). [CELL] and [PAD] are never masked."""
    B, L = type_ids.shape
    eligible = (type_ids >= N_SPECIAL) & ~pad
    rate = lo + (hi - lo) * torch.rand(B, 1, generator=gen)
    u = torch.rand(B, L, generator=gen)
    m = (u < rate) & eligible
    # guarantee >=1 masked token in every sequence that has any real token
    none = (m.sum(1) == 0) & (eligible.sum(1) > 0)
    if none.any():
        score = torch.where(eligible, u, torch.full_like(u, 2.0))
        first = score.argmin(1)
        m[none, first[none]] = True
    m[:, 0] = False
    assert not (m & (type_ids == CELL_ID)).any() and not (m & (type_ids == PAD_ID)).any()
    return m
