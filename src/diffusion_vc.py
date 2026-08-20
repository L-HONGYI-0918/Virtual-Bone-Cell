# -*- coding: utf-8 -*-
"""
PathBone-MF latent-space diffusion module
==========================================
1. prepare_lincs_memmap : GCTX -> landmark-978 memmap + meta (one-time preprocessing)
2. GNNEncoder           : molecular graph encoder (29-dim atom features, 3 GAT layers, 256-d output)
3. VAE                  : 978 genes -> 64-dim latent compression
4. CondDDPM             : conditional diffusion on 64-dim latent p(z | molecule embedding, dose, time)
5. sample_drug          : SMILES -> 978-gene logFC (average over repeated sampling)
6. validate_gse28074    : bone-gene direction agreement against GSE28074
"""
import sys, io, csv, json, math, time
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from pathlib import Path
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from rdkit import Chem
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "v20"
OUT.mkdir(parents=True, exist_ok=True)

ATOM_TYPES = ["C","N","O","S","F","Cl","P","Br","I","B","Si","Se","other"]

# ============================================================
# Molecular graph construction
# ============================================================
def atom_feat(a):
    sym = a.GetSymbol()
    ti = ATOM_TYPES.index(sym) if sym in ATOM_TYPES else 12
    toh = [0]*len(ATOM_TYPES); toh[ti] = 1
    d = min(a.GetDegree(), 6); doh = [0]*7; doh[d] = 1
    h = str(a.GetHybridization()); hyb = ["SP","SP2","SP3","SP3D","SP3D2"]; hoh = [0]*5
    for j, x in enumerate(hyb):
        if x in h: hoh[j] = 1; break
    return toh + doh + hoh + [int(a.IsInRing()), int(a.IsInRingSize(6)),
                             int(a.GetIsAromatic()), min(a.GetTotalNumHs(), 4)/4.0]

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    xs = [atom_feat(a) for a in mol.GetAtoms()]
    if not xs: return None
    x = torch.tensor(xs, dtype=torch.float32)
    ei = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ei.append([i, j]); ei.append([j, i])
    if not ei: ei = [[0, 0]]
    return Data(x=x, edge_index=torch.tensor(ei, dtype=torch.long).t().contiguous())

# ============================================================
# Molecular graph encoder
# ============================================================
class GNNEncoder(nn.Module):
    def __init__(self, out_dim=256, dropout=0.1):
        super().__init__()
        self.conv1 = GATConv(29, 128, heads=3, dropout=dropout)
        self.conv2 = GATConv(384, 128, heads=2, dropout=dropout)
        self.conv3 = GATConv(256, 256, heads=1, dropout=dropout)
        self.ln1 = nn.LayerNorm(384); self.ln2 = nn.LayerNorm(256); self.ln3 = nn.LayerNorm(256)
        self.proj = nn.Sequential(nn.Linear(512, out_dim), nn.LayerNorm(out_dim))
    def forward(self, data):
        x, ei, b = data.x, data.edge_index, data.batch
        x = F.relu(self.ln1(self.conv1(x, ei)))
        x = F.relu(self.ln2(self.conv2(x, ei)))
        x = F.relu(self.ln3(self.conv3(x, ei)))
        g = torch.cat([global_mean_pool(x, b), global_max_pool(x, b)], dim=-1)
        return self.proj(g)

# ============================================================
# VAE: 978 -> 64 latent space
# ============================================================
class VAE(nn.Module):
    def __init__(self, n_lm=978, latent=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_lm, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU())
        self.mu = nn.Linear(128, latent)
        self.logvar = nn.Linear(128, latent)
        self.dec = nn.Sequential(nn.Linear(latent, 256), nn.ReLU(),
                                 nn.Linear(256, n_lm))
        self.latent = latent
    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)
    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)
    def decode(self, z):
        return self.dec(z)
    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparam(mu, lv)
        return self.decode(z), mu, lv, z
    def loss(self, x, beta=0.5):
        recon, mu, lv, _ = self.forward(x)
        mse = F.mse_loss(recon, x)
        kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / x.size(0)
        return mse + beta * kl, mse.item(), kl.item()

# ============================================================
# Conditional latent diffusion (DDPM with DDIM sampling)
# ============================================================
def cosine_betas(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float32) / T
    alphas_bar = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
    return torch.clip(betas, 0.0001, 0.02)

class CondDDPM(nn.Module):
    """Conditional latent diffusion (v-prediction + FiLM conditioning + min-SNR weighting + CFG)."""
    def __init__(self, latent=64, cond_dim=258, hidden=512, T=400, cfg_drop=0.1):
        super().__init__()
        self.T = T; self.latent = latent; self.cfg_drop = cfg_drop
        self.register_buffer("betas", cosine_betas(T))
        ab = torch.cumprod(1 - self.betas, 0)
        self.register_buffer("sqrt_ab", torch.sqrt(ab))
        self.register_buffer("sqrt_1m_ab", torch.sqrt(1 - ab))
        self.time_mlp = nn.Sequential(nn.Linear(128, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.x_in = nn.Linear(latent, hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.film1 = nn.Linear(hidden, hidden * 2)
        self.film2 = nn.Linear(hidden, hidden * 2)
        self.film3 = nn.Linear(hidden, hidden * 2)
        self.out = nn.Linear(hidden, latent)
    def time_emb(self, t):
        te = torch.zeros(t.size(0), 128, device=t.device)
        f = torch.exp(torch.arange(0, 128, 2, device=t.device).float() * (-math.log(10000) / 128))
        te[:, 0::2] = torch.sin(t[:, None].float() * f[None])
        te[:, 1::2] = torch.cos(t[:, None].float() * f[None])
        return te
    def _film(self, c, film_layer, h):
        g, b = film_layer(c).chunk(2, dim=-1)
        return h * g + b
    def _net_out(self, x, c, te):
        h = self._film(c, self.film1, self.x_in(x) + te)
        h = F.silu(h)
        h = self._film(c, self.film2, self.fc1(h))
        h = F.silu(h)
        h = self._film(c, self.film3, self.fc2(h))
        h = F.silu(h)
        return self.out(h)
    def forward(self, x0, cond, t, drop_cond=True):
        noise = torch.randn_like(x0)
        xt = self.sqrt_ab[t, None] * x0 + self.sqrt_1m_ab[t, None] * noise
        v = self.sqrt_ab[t, None] * noise - self.sqrt_1m_ab[t, None] * x0
        te = self.time_mlp(self.time_emb(t))
        cc = cond
        if self.training and drop_cond and self.cfg_drop > 0:
            drop = torch.rand(cc.size(0), device=cc.device) < self.cfg_drop
            cc = cc * (~drop).float().unsqueeze(-1)
        c = self.cond_mlp(cc)
        pred = self._net_out(xt, c, te)
        # min-SNR weighting (clamp 5): up-weights low-noise steps so the model uses the condition
        snr = (self.sqrt_ab ** 2) / (self.sqrt_1m_ab ** 2 + 1e-8)
        w = torch.clamp(snr, max=5.0)
        loss = (w[t] * (pred - v).pow(2).mean(dim=-1)).mean()
        return loss, noise
    @torch.no_grad()
    def sample(self, cond, steps=50, n=1, cfg=0.0):
        """v-prediction DDIM sampling; cfg>0 enables classifier-free guidance."""
        self.eval()
        x = torch.randn(n, self.latent, device=cond.device)
        if cond.size(0) == 1:
            cond = cond.repeat(n, 1)
        uncond = torch.zeros_like(cond)
        cu = self.cond_mlp(uncond)
        c = self.cond_mlp(cond)
        ts = torch.linspace(self.T - 1, 0, steps).long().to(cond.device)
        for i, t in enumerate(ts):
            tb = t.expand(n)
            a = self.sqrt_ab[tb]; b = self.sqrt_1m_ab[tb]
            te = self.time_mlp(self.time_emb(tb))
            v = self._net_out(x, c, te)
            if cfg > 0:
                v_u = self._net_out(x, cu, te)
                v = (1 + cfg) * v - cfg * v_u
            x0 = a[:, None] * x - b[:, None] * v
            x0 = x0.clamp(-6, 6)
            if i < steps - 1:
                t_next = ts[i + 1].expand(n)
                a_next = self.sqrt_ab[t_next]; b_next = self.sqrt_1m_ab[t_next]
                eps = b[:, None] * x + a[:, None] * v
                x = a_next[:, None] * x0 + b_next[:, None] * eps
        return x

# ============================================================
# Conditioning: molecule embedding + dose + time
# ============================================================
def build_cond(drug_emb, dose, time_h, device):
    d = torch.zeros((drug_emb.size(0), 1), device=device)
    t = torch.zeros((drug_emb.size(0), 1), device=device)
    for i, (dd, tt) in enumerate(zip(dose, time_h)):
        if dd and dd > 0: d[i, 0] = float(np.clip(math.log10(dd) / 3, -1, 1))
        if tt and tt > 0: t[i, 0] = float(np.clip(math.log10(tt) / 3, 0, 1))
    return torch.cat([drug_emb, d, t], dim=-1)

# ============================================================
# LINCS data: GCTX -> memmap (978 landmark genes)
# ============================================================
def parse_col_id(cid):
    """Supports two LINCS column-ID formats:
       4-part: prefix:pert_id:dose:time   (e.g. ABY001_A375_XH:BRD-A61304759:0.625:24)
       3-part: prefix:pert:dose           (e.g. TSAI002_NPC-8_XH:SAHA:2.5, time defaults to 24h)
       2 parts or fewer: no drug info, returns None
    """
    parts = cid.split(":")
    if len(parts) < 3: return None
    cell = parts[0]
    pert = parts[1]
    if pert.startswith("BRD-") and len(pert.split("-")) > 2:
        pert = "-".join(pert.split("-")[:2])
    try: dose = float(parts[2])
    except: dose = 0.0
    t = 24.0
    if len(parts) >= 4:
        try: t = float(parts[3])
        except: pass
    return cell, pert, dose, t

def prepare_lincs_memmap(gctx_path, geneinfo_path, out_dir, max_rows=None, log=print):
    """Extract landmark-978 genes of valid drug samples (3/4-part column IDs) from GCTX into a float16 memmap. Returns a dict."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    mm_path = out_dir / "lincs_lm.memmap"
    meta_path = out_dir / "lincs_meta.json"
    with open(str(geneinfo_path), encoding="utf-8") as f:
        gr = list(csv.DictReader(f, delimiter="\t"))
    lm_idx = [i for i, r in enumerate(gr) if r.get("feature_space") == "landmark"]
    lm_sym = [gr[i]["gene_symbol"] for i in lm_idx]
    log(f"[DATA] landmark genes: {len(lm_idx)}")
    with h5py.File(str(gctx_path), "r") as f:
        ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f["0/META/COL/id"][...]]
    # keep only samples whose column ID parses to a drug (3-part / 4-part)
    valid = []
    for ci, cid in enumerate(ids):
        pp = parse_col_id(cid)
        if pp is not None:
            valid.append((ci, pp))
    if max_rows is not None:
        valid = valid[:max_rows]
    n_rows = len(valid)
    if mm_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["n"] == n_rows:
            log(f"[DATA] reusing existing preprocessing: {mm_path} ({meta['n']} valid samples)")
            return meta
        log(f"[DATA] sample count mismatch (have {meta['n']}, need {n_rows}); re-preprocessing")
        mm_path.unlink(missing_ok=True); meta_path.unlink(missing_ok=True)
    log(f"[DATA] valid drug samples {n_rows}/{len(ids)}")
    valid_rows = [v[0] for v in valid]  # monotonically increasing original row numbers
    with h5py.File(str(gctx_path), "r") as f:
        mm = np.memmap(str(mm_path), dtype="float16", mode="w+", shape=(n_rows, len(lm_idx)))
        step = 20000
        for s in range(0, n_rows, step):
            e = min(s + step, n_rows)
            r0 = valid_rows[s]; r1 = valid_rows[e - 1] + 1
            # h5py: slice + single-column list is allowed; fancy row indexing needs a numpy second pass
            block = f["0/DATA/0/matrix"][r0:r1, lm_idx].astype(np.float16)
            rel = np.array([r - r0 for r in valid_rows[s:e]])
            mm[s:e] = block[rel]
            mm.flush()
            log(f"[DATA] preprocessing progress: {e}/{n_rows}")
        del mm
    cells = [v[1][0] for v in valid]
    perts = [v[1][1] for v in valid]
    doses = [v[1][2] for v in valid]
    times = [v[1][3] for v in valid]
    meta = {"n": n_rows, "memmap": str(mm_path), "genes": lm_sym,
            "cells": cells, "perts": perts, "doses": doses, "times": times}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    log(f"[DATA] preprocessing done: {n_rows} valid samples x {len(lm_idx)} genes")
    return meta

class LincsBatch:
    """Sample randomly from the memmap."""
    def __init__(self, meta, rng=None):
        self.meta = meta
        self.mm = np.memmap(meta["memmap"], dtype="float16", mode="r",
                            shape=(meta["n"], len(meta["genes"])))
        self.perts = meta["perts"]; self.doses = meta["doses"]; self.times = meta["times"]
        self.rng = rng or np.random.default_rng(0)
        self.n = meta["n"]
    def sample(self, n):
        idx = self.rng.integers(0, self.meta["n"], size=n)
        x = torch.tensor(self.mm[idx].astype(np.float32))
        return x, [self.perts[i] for i in idx], [self.doses[i] for i in idx], [self.times[i] for i in idx]

# ============================================================
# Training
# ============================================================
def train_vae(data, device, epochs=20, batch_size=256, lr=1e-3, out_name="vae.pt",
              n_smoke=None, log=print, n_lm=978):
    model = VAE(n_lm=n_lm).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    total = data.n if n_smoke is None else min(n_smoke, data.n)
    log(f"[VAE] training {epochs} epochs, {total} samples per epoch, device={device}")
    for ep in range(epochs):
        beta = 0.0  # pure AE: the latent is free to carry information; DDPM learns the z distribution
        model.train(); tot = 0.0; nb = 0; kl_sum = 0.0
        for s in range(0, total, batch_size):
            n = min(batch_size, total - s)
            x, *_ = data.sample(n)
            x = x.to(device)
            loss, mse, kl = model.loss(x, beta=beta)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += mse; nb += 1; kl_sum += kl
        if nb:
            with torch.no_grad():
                mu, lv = model.encode(x)
                zstd = mu.std().item()
            log(f"[VAE] epoch {ep+1}/{epochs} recon-MSE={tot/nb:.5f} KL={kl_sum/nb:.3f} zstd={zstd:.3f}")
    torch.save({"model": model.state_dict(), "latent": model.latent}, str(OUT / out_name))
    log(f"[VAE] saved {out_name}")
    return model

def train_ddpm(data, mol_enc, device, epochs=30, batch_size=256, lr=2e-4, T=400,
               out_name="ddpm.pt", n_smoke=None, log=print, mol_enc_frozen=True,
               smiles_map=None, vae=None, n_lm=978, aux_reg=True,
               labels=None, pw_weight=None, core_osteo_idx=None, bone_up=None,
               bone_lambda=0.15, geo_lambda=0.1, bone_every=8, bone_steps_t=200,
               init_ckpt=None, anchors=None, anchor_margin=0.2):
    # anchors: list of (vec_978, sign, weight) - multi-GEO anchor directional supervision (hinge)
    # bone supervision: labeled drug subset (requires real effect data)
    label_perts = []; label_eff = None; label_ys = None; bone_rng = None
    if labels:
        eff_map = dict(zip(data.perts, data.eff))
        lps = [(pt, y) for pt, y in labels.items() if pt in eff_map]
        if lps:
            label_perts = [p for p, _ in lps]
            label_ys = np.array([y for _, y in lps], dtype=np.int64)
            label_eff = np.stack([eff_map[p] for p in label_perts]).astype(np.float32)
            log(f"[DDPM] bone supervision: {len(label_perts)} labeled drugs "
                f"(P={int((label_ys==0).sum())} I={int((label_ys==1).sum())} N={int((label_ys==2).sum())})")
            bone_rng = np.random.default_rng(7)
    cond_dim = 258
    hidden = 512
    model = CondDDPM(latent=64, cond_dim=cond_dim, hidden=hidden, T=T, cfg_drop=0.1).to(device)
    init_stats = None
    if init_ckpt is not None:
        ck_init = torch.load(str(init_ckpt), map_location=device, weights_only=False)
        model.load_state_dict(ck_init["model"])
        init_stats = (ck_init.get("z_mean"), ck_init.get("z_std"))
        log(f"[DDPM] resuming from {Path(init_ckpt).name}")
    mol_enc = mol_enc.to(device)
    if vae is not None:
        vae.eval()
        for p in vae.parameters(): p.requires_grad = False
    if mol_enc_frozen:
        for p in mol_enc.parameters(): p.requires_grad = False
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    aux_head = None
    enc_opt = None
    if not mol_enc_frozen:
        aux_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64)).to(device)
        enc_opt = torch.optim.AdamW(list(mol_enc.parameters()) + list(aux_head.parameters()),
                                    lr=lr * 0.5, weight_decay=1e-5)
    # molecule embeddings cached globally (precomputed by build_emb_cache)
    global _EMB_CACHE, _GRAPH_CACHE
    # auto-fill cache: scan drugs present in the data
    if smiles_map is not None:
        if not mol_enc_frozen:
            todo = [pt for pt in data.perts if pt not in _GRAPH_CACHE and smiles_map.get(pt)]
            log(f"[DDPM] end-to-end mode: building molecule graph cache for {len(todo)}")
            for k, pt in enumerate(todo):
                g = mol_to_graph(smiles_map[pt])
                if g is not None: _GRAPH_CACHE[pt] = g
                if (k + 1) % 4000 == 0: log(f"[DDPM] graph cache {k+1}/{len(todo)}")
            log(f"[DDPM] graph cache complete: {len(_GRAPH_CACHE)}")
        else:
            scan_perts = set()
            for pt in data.perts:
                if pt not in _EMB_CACHE and smiles_map.get(pt):
                    scan_perts.add(pt)
            if scan_perts:
                log(f"[DDPM] found {len(scan_perts)} sample drugs to add to the embedding cache")
                build_emb_cache(mol_enc, smiles_map, list(scan_perts)[:20000], device, log=log, batch=128)
    total = data.n if n_smoke is None else min(n_smoke, data.n)
    log(f"[DDPM] training {epochs} epochs, T={T}, data points={total}, device={device}")
    # prescan: compute latent z mean/std (DDPM trains on standardized z)
    if vae is not None:
        zs = []
        vae.eval()
        with torch.no_grad():
            for s in range(0, min(total, 6000), 256):
                xb, *_ = data.sample(min(256, min(total, 6000) - s))
                mu, _ = vae.encode(xb.to(device))
                zs.append(mu.cpu().numpy())
        z_all = np.concatenate(zs)
        z_mean = z_all.mean(0); z_std = z_all.std(0) + 1e-4
        if init_stats is not None and init_stats[0] is not None:
            z_mean = np.array(init_stats[0], dtype=np.float64)
            z_std = np.array(init_stats[1], dtype=np.float64)
            log("[DDPM] reusing z statistics from init_ckpt")
        else:
            log(f"[DDPM] z stats: mean={z_mean.mean():.3f} std={z_std.mean():.3f}")
    else:
        z_mean = np.zeros(64); z_std = np.ones(64)
    for ep in range(epochs):
        model.train(); tot = 0.0; nb = 0; miss = 0
        for s in range(0, total, batch_size):
            n = min(batch_size, total - s)
            x, perts, doses, times = data.sample(n)
            if not mol_enc_frozen:
                # end-to-end: filter drugs by graph cache
                idx = [i for i in range(n) if _GRAPH_CACHE.get(perts[i]) is not None]
                if len(idx) < max(1, n // 4):
                    miss += 1; continue
                x = x[idx].to(device)
                bg = Batch.from_data_list([_GRAPH_CACHE[perts[i]] for i in idx]).to(device)
                embs = mol_enc(bg)
                cond = build_cond(embs, [doses[i] for i in idx], [times[i] for i in idx], device)
                with torch.no_grad():
                    if vae is not None:
                        mu, _ = vae.encode(x)
                        mu = (mu - torch.tensor(z_mean, dtype=torch.float32, device=device)) / torch.tensor(z_std, dtype=torch.float32, device=device)
                    else:
                        mu = x
                t = torch.randint(0, T, (len(idx),), device=device)
                loss, _ = model(mu, cond, t)
                if aux_reg and aux_head is not None:
                    loss = loss + 0.3 * F.mse_loss(aux_head(embs), mu)
                opt.zero_grad(); loss.backward(); opt.step()
                if enc_opt is not None:
                    enc_opt.step()
                tot += loss.item(); nb += 1
                continue
            # bone-supervision batch: every bone_every batches, use labeled drugs
            is_bone = (label_eff is not None) and (nb % bone_every == bone_every // 2)
            if is_bone:
                bi = bone_rng.integers(0, len(label_perts), size=n)
                bn = [label_perts[i] for i in bi]
                by0 = np.array([label_ys[i] for i in bi])
                idx = [i for i in range(n) if _EMB_CACHE.get(bn[i]) is not None]
                if len(idx) < max(1, n // 4):
                    miss += 1; continue
                x = torch.tensor(label_eff[bi[idx]], dtype=torch.float32).to(device)
                by = torch.tensor([{0: 1.0, 1: -1.0, 2: 0.0}[int(v)] for v in by0[idx]],
                                  dtype=torch.float32, device=device)
                embs = torch.stack([_EMB_CACHE[bn[i]] for i in idx]).to(device)
                cond = build_cond(embs, [1.0]*len(idx), [24.0]*len(idx), device)
                with torch.no_grad():
                    if vae is not None:
                        mu, _ = vae.encode(x)
                        mu = (mu - torch.tensor(z_mean, dtype=torch.float32, device=device)) / torch.tensor(z_std, dtype=torch.float32, device=device)
                    else:
                        mu = x
                t = torch.randint(0, T, (len(idx),), device=device)
                loss, _ = model(mu, cond, t)
                # bone-direction loss: low-noise-step x0 estimate -> decode -> bone pathway scores / GEO anchor cosine
                tb = torch.randint(0, min(bone_steps_t, T), (len(idx),), device=device)
                noise = torch.randn_like(mu)
                aa = model.sqrt_ab[tb]; bb = model.sqrt_1m_ab[tb]
                xt = aa[:, None] * mu + bb[:, None] * noise
                te = model.time_mlp(model.time_emb(tb))
                cc = model.cond_mlp(cond)
                vp = model._net_out(xt, cc, te)
                x0 = (aa[:, None] * xt - bb[:, None] * vp).clamp(-6, 6)
                z_orig = x0 * torch.tensor(z_std, dtype=torch.float32, device=device) \
                         + torch.tensor(z_mean, dtype=torch.float32, device=device)
                gex = vae.decode(z_orig)
                lb = torch.zeros((), device=device)
                if pw_weight is not None and core_osteo_idx is not None:
                    pwt = torch.tensor(pw_weight, dtype=torch.float32, device=device)
                    osteo = (gex @ pwt.t())[:, core_osteo_idx].mean(1)
                    lb = lb + bone_lambda * F.mse_loss(osteo, by)
                if anchors is not None:
                    for vec, sign, w in anchors:
                        bu = torch.tensor(vec, dtype=torch.float32, device=device)
                        cosim = (gex * bu).sum(1) / (gex.norm(2, 1) * bu.norm() + 1e-8)
                        tgt = sign * by  # -1/0/+1
                        la = torch.zeros_like(cosim)
                        pos = tgt > 0; neg = tgt < 0; neu = tgt == 0
                        if pos.any(): la[pos] = F.relu(anchor_margin - cosim[pos])
                        if neg.any(): la[neg] = F.relu(anchor_margin + cosim[neg])
                        if neu.any(): la[neu] = F.relu(cosim[neu].abs() - anchor_margin)
                        lb = lb + geo_lambda * w * la.mean()
                elif bone_up is not None:
                    bu = torch.tensor(bone_up, dtype=torch.float32, device=device)
                    cosim = (gex * bu).sum(1) / (gex.norm(2, 1) * bu.norm() + 1e-8)
                    lb = lb + geo_lambda * F.mse_loss(cosim, by)
                loss = loss + lb
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item(); nb += 1
                continue
            # keep only samples whose drug resolves to a SMILES
            idx = [i for i in range(n) if _EMB_CACHE.get(perts[i]) is not None]
            if len(idx) < max(1, n // 4):
                miss += 1; continue
            idx = idx[:min(len(idx), n)]
            x = x[idx].to(device)
            embs = torch.stack([_EMB_CACHE[perts[i]] for i in idx]).to(device)
            cond = build_cond(embs, [doses[i] for i in idx], [times[i] for i in idx], device)
            with torch.no_grad():
                if vae is not None:
                    mu, _ = vae.encode(x)
                    mu = (mu - torch.tensor(z_mean, dtype=torch.float32, device=device)) / torch.tensor(z_std, dtype=torch.float32, device=device)
                else:
                    mu = x
            t = torch.randint(0, T, (len(idx),), device=device)
            loss, _ = model(mu, cond, t)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if nb: log(f"[DDPM] epoch {ep+1}/{epochs} noise-MSE={tot/nb:.5f} (skipped sample groups={miss})")
    torch.save({"model": model.state_dict(), "T": T, "hidden": hidden,
                "z_mean": z_mean, "z_std": z_std}, str(OUT / out_name))
    model.z_mean = z_mean; model.z_std = z_std
    log(f"[DDPM] saved {out_name}")
    return model

class EffectsBatch:
    """Drug-effect-level data: one average 978-effect vector per drug (much higher SNR than single profiles)."""
    def __init__(self, perts, eff, rng=None):
        self.perts = list(perts); self.eff = eff
        self.rng = rng or np.random.default_rng(0)
        self.n = len(self.perts)
    def sample(self, n):
        idx = self.rng.integers(0, self.n, size=n)
        x = torch.tensor(self.eff[idx], dtype=torch.float32)
        return x, [self.perts[i] for i in idx], [1.0] * n, [24.0] * n

def build_drug_effects(meta, min_samples=3, log=print):
    """Group profiles by drug and average effect vectors. Returns (perts, eff matrix, sample-count dict)."""
    mm = np.memmap(meta["memmap"], dtype="float16", mode="r",
                   shape=(meta["n"], len(meta["genes"])))
    sums = {}; cnts = {}
    for i in range(meta["n"]):
        p = meta["perts"][i]
        if p not in sums:
            sums[p] = np.zeros(len(meta["genes"]), dtype=np.float32); cnts[p] = 0
        sums[p] += mm[i].astype(np.float32); cnts[p] += 1
    keep = [p for p in sums if cnts[p] >= min_samples]
    eff = np.stack([sums[p] / cnts[p] for p in keep]).astype(np.float32)
    log(f"[EFFECT] drugs (>= {min_samples} samples): {len(keep)}/{len(sums)}")
    return keep, eff, {p: cnts[p] for p in keep}

def pert_smiles(pert, compound_map=None):
    """pert_id -> SMILES (injected globally by run_pipeline)."""
    global _SMILES_MAP
    if compound_map is not None:
        return compound_map.get(pert)
    return _SMILES_MAP.get(pert)

_SMILES_MAP = {}

_EMB_CACHE = {}
_GRAPH_CACHE = {}   # pert_id -> molecule embedding (256-d), reused across stages

def build_emb_cache(mol_enc, smiles_map, pert_ids, device="cuda", log=print, batch=256):
    """Precompute molecule embeddings for a batch of perts and store in the global cache. Returns the cache size."""
    global _EMB_CACHE
    todo = [p for p in pert_ids if p not in _EMB_CACHE and smiles_map.get(p)]
    log(f"[EMB] computing embeddings for {len(todo)} drugs")
    mol_enc = mol_enc.to(device).eval()
    from torch_geometric.data import Batch
    with torch.no_grad():
        for s in range(0, len(todo), batch):
            chunk = todo[s:s+batch]
            graphs = []
            ok_ids = []
            for pt in chunk:
                g = mol_to_graph(smiles_map[pt])
                if g is not None:
                    graphs.append(g); ok_ids.append(pt)
            if not graphs: continue
            emb = mol_enc(Batch.from_data_list(graphs).to(device))
            for pt, e in zip(ok_ids, emb):
                _EMB_CACHE[pt] = e.detach().cpu()
            if (s // batch + 1) % 200 == 0:
                log(f"[EMB] {min(s+batch, len(todo))}/{len(todo)}")
    log(f"[EMB] done, cache holds {len(_EMB_CACHE)} drugs")
    return len(_EMB_CACHE)

# ============================================================
# Drug prediction: SMILES -> 978 logFC
# ============================================================
@torch.no_grad()
def sample_drug(ddpm, vae, mol_enc, smiles, dose=1.0, time_h=24.0,
                n_samples=32, steps=50, device="cuda", cfg=0.0):
    ddpm.eval(); vae.eval(); mol_enc.eval()
    g = mol_to_graph(smiles)
    if g is None: return None
    emb = mol_enc(Batch.from_data_list([g]).to(device))
    cond = build_cond(emb, [dose], [time_h], device)
    z = ddpm.sample(cond, steps=steps, n=n_samples, cfg=cfg)
    zm = getattr(ddpm, "z_mean", None)
    if zm is not None:
        z = z * torch.tensor(ddpm.z_std, dtype=torch.float32, device=device) \
            + torch.tensor(ddpm.z_mean, dtype=torch.float32, device=device)
    gex = vae.decode(z)
    return gex.mean(0).cpu().numpy()

# ============================================================
# GSE28074 validation (bone-gene direction agreement)
# ============================================================
BONE_KW = ["OSTEOBLAST","BONE","OSSIF","RUNX2","BMP","WNT","SMAD","BGLAP","SP7",
           "SPP1","COL1A","ALP","TGFB","CTNNB1","IBSP","DLX5","DMP1","SOST","FOS","ATF4"]

def load_gse28074(gz_path, geneinfo_path):
    import gzip
    with gzip.open(str(gz_path), "rt", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    st = []; ds = 0
    for i, ln in enumerate(lines):
        if ln.startswith("!Sample_title"): st = ln.strip().split("\t")[1:]
        if ln.startswith("!series_matrix_table_begin"): ds = i + 1; break
    pids, ev = [], []
    for ln in lines[ds+1:]:
        if ln.startswith("!series_matrix_table_end"): break
        ps = ln.strip().split("\t")
        if len(ps) < 2: continue
        pid = ps[0].strip('"')
        if pid.endswith("_at"):
            try:
                pids.append(int(pid.replace("_at", "")))
                ev.append([float(x.strip('"')) for x in ps[1:]])
            except Exception: pass
    expr = np.array(ev, dtype=np.float32)
    with open(str(geneinfo_path), encoding="utf-8") as f:
        gr = list(csv.DictReader(f, delimiter="\t"))
    e2s = {}
    for r in gr:
        try: e2s[int(r["gene_id"])] = r["gene_symbol"]
        except Exception: pass
    psyms = [e2s.get(e, "") for e in pids]
    groups = {"Ctrl": [i for i, t in enumerate(st) if "control" in t.lower() and "bmp6" not in t.lower()]}
    for tp in ["8hr", "24hr", "96hr", "10d"]:
        groups[tp] = []
    for i, t in enumerate(st):
        tl = t.lower()
        if "bmp6" not in tl or "control" in tl: continue
        for tp, key in [("8hr", "8hr culture"), ("24hr", "24hr culture"),
                        ("96hr", "96hr culture"), ("10d", "10d culture")]:
            if key in tl: groups[tp].append(i)
    s2g = {}
    for i, sym in enumerate(psyms):
        if sym: s2g.setdefault(sym.upper(), []).append(i)
    return expr, groups, s2g

def validate_gse28074(ddpm, vae, mol_enc, drug_smiles_p, gz_path, geneinfo_path,
                      device="cuda", n_samples=16, steps=20, log=print):
    """Predict bone-gene directions for pro-osteogenic drugs and compare with real BMP6-induced directions. Returns a dict."""
    expr, groups, s2g = load_gse28074(gz_path, geneinfo_path)
    genes = load_landmark_genes(geneinfo_path)
    g2i = {g: i for i, g in enumerate(genes)}
    matched = [(g, g2i[g]) for g in genes if g.upper() in s2g and g in g2i]
    bone = [(g, gi) for g, gi in matched if any(kw in g.upper() for kw in BONE_KW)]
    log(f"[GEO] matched genes {len(matched)}, bone genes {len(bone)}")
    if len(bone) < 5:
        log("[GEO] too few bone genes; validation invalid")
        return {"error": "matched bone genes too few"}
    # model-averaged direction
    pred_sum = np.zeros(len(genes), dtype=np.float64); n_done = 0
    for name, smi in drug_smiles_p.items():
        g = sample_drug(ddpm, vae, mol_enc, smi, n_samples=n_samples, steps=steps, device=device)
        if g is None:
            log(f"[GEO] {name} has no valid SMILES; skipping")
            continue
        pred_sum += g; n_done += 1
    if n_done == 0:
        return {"error": "no drug predicted"}
    pred = pred_sum / n_done
    log(f"[GEO] model-averaged over {n_done} pro-osteogenic drugs")
    res = {}
    for tp in ["8hr", "24hr", "96hr", "10d"]:
        gi = groups.get(tp) or []
        if not gi or not groups.get("Ctrl"): continue
        ctrl = expr[:, groups["Ctrl"]].mean(1)
        trt = expr[:, gi].mean(1)
        fc = trt - ctrl
        geo_vec = np.zeros(len(matched))
        for k, (g, _gi) in enumerate(matched):
            geo_vec[k] = np.median(fc[s2g[g.upper()]])
        pred_vec = np.array([pred[gi] for _, gi in matched])
        bm = np.array([any(kw in g.upper() for kw in BONE_KW) for g, _ in matched])
        gv, pv = geo_vec[bm], pred_vec[bm]
        agree = np.mean(np.sign(gv) == np.sign(pv)) if len(gv) else 0.0
        res[tp] = round(float(agree), 4)
        log(f"[GEO] {tp}: direction agreement = {agree:.2%} ({len(gv)} bone genes)")
    res["mean"] = round(float(np.mean(list(res.values()))), 4) if res else 0.0
    return res

def load_landmark_genes(geneinfo_path):
    with open(str(geneinfo_path), encoding="utf-8") as f:
        gr = list(csv.DictReader(f, delimiter="\t"))
    return [r["gene_symbol"] for r in gr if r.get("feature_space") == "landmark"]

def load_smiles_map(compoundinfo_path):
    """Build a SMILES lookup: BRD-ID keys + drug-name keys (3-part samples matched by name)."""
    with open(str(compoundinfo_path), encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    m = {}
    for r in rows:
        s = r.get("canonical_smiles", "").strip()
        if not s or s == "restricted": continue
        m.setdefault(r["pert_id"], s)
        nm = (r.get("cmap_name") or "").strip().lower()
        if nm and nm not in m:
            m[nm] = s
    return m