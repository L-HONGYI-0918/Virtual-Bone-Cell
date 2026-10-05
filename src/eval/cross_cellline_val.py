# -*- coding: utf-8 -*-
""" (see paper): v19 model (see paper) vs  (see paper)U2OS (see paper)
 (see paper): U2OS(train) / A673, SKES1, TC32( (see paper),  (see paper))
 (see paper): outputs/cross_cellline_results.json +  (see paper)/fig10_cross_cellline.pdf
"""
import sys, io, csv, json, time, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
from collections import defaultdict
import numpy as np
import h5py
import torch
import torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from rdkit import Chem
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("G:/ (see paper)"); OUT = ROOT / "outputs"; FIG = ROOT / " (see paper)"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GCTX = ROOT / "level5_beta_trt_cp_n720216x12328.gctx"
CELLS = {"SKES1": "SKES1", "TC32": "TC32", "A673": "A673"}
FIG.mkdir(exist_ok=True)

# ---------- 1. model ----------
ckpt = torch.load(str(OUT / "bone_vc_v19.pt"), map_location=DEV, weights_only=False)
cfg = ckpt["config"]; pw_weight = ckpt["pw_weight"]; core_osteo_idx = ckpt["core_osteo_idx"]
pathways = ckpt["pathways"]
osteo_names = [pathways[i] for i in core_osteo_idx]

ATOM_TYPES = ["C","N","O","S","F","Cl","P","Br","I","B","Si","Se","other"]
def af(a):
    sym = a.GetSymbol(); ti = ATOM_TYPES.index(sym) if sym in ATOM_TYPES else 12
    toh = [0]*len(ATOM_TYPES); toh[ti] = 1
    d = min(a.GetDegree(), 6); doh = [0]*7; doh[d] = 1
    h = str(a.GetHybridization()); hyb = ["SP","SP2","SP3","SP3D","SP3D2"]; hoh = [0]*5
    for j, x in enumerate(hyb):
        if x in h: hoh[j] = 1; break
    return toh + doh + hoh + [int(a.IsInRing()), int(a.IsInRingSize(6)), int(a.GetIsAromatic()), min(a.GetTotalNumHs(),4)/4.0]
def mtg(s):
    mol = Chem.MolFromSmiles(s)
    if mol is None: return None
    mol = Chem.AddHs(mol); x = [af(a) for a in mol.GetAtoms()]
    if not x: return None
    x = torch.tensor(x, dtype=torch.float32); ei = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx(); j = b.GetEndAtomIdx()
        ei.append([i,j]); ei.append([j,i])
    if not ei: ei = [[0,0]]
    return Data(x=x, edge_index=torch.tensor(ei, dtype=torch.long).t().contiguous())

class GNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(29, 128, heads=3, dropout=0.1)
        self.conv2 = GATConv(384, 128, heads=2, dropout=0.1)
        self.conv3 = GATConv(256, 256, heads=1, dropout=0.1)
        self.ln1 = nn.LayerNorm(384); self.ln2 = nn.LayerNorm(256); self.ln3 = nn.LayerNorm(256)
    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.ln1(self.conv1(x, ei))); x = F.relu(self.ln2(self.conv2(x, ei)))
        x = F.relu(self.ln3(self.conv3(x, ei)))
        return torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn = GNNEncoder()
        self.head = nn.Sequential(nn.Linear(512,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, cfg["npw"]))
        self.register_buffer("pw2gene", torch.tensor(pw_weight, dtype=torch.float32))
    def forward(self, bg):
        z = self.gnn(bg); pw = self.head(z); genes = pw @ self.pw2gene
        return pw, genes

model = Model().to(DEV); model.load_state_dict(ckpt["model"]); model.eval()

def torch_geometric_batch(graphs):
    from torch_geometric.data import Batch
    return Batch.from_data_list(graphs)

def predict_osteo(smi):
    g = mtg(smi)
    if g is None: return None
    with torch.no_grad():
        pw, _ = model(torch_geometric_batch([g]).to(DEV))
    return float(pw[0, core_osteo_idx].mean().cpu())

# ---------- 2.  (see paper) SMILES ----------
with open(str(ROOT / "compoundinfo_beta.txt"), encoding="utf-8") as f:
    comp_rows = list(csv.DictReader(f, delimiter="\t"))
cid2smiles = {}; cid2name = {}
for r in comp_rows:
    s = r.get("canonical_smiles", "").strip()
    if s and s != "restricted": cid2smiles[r["pert_id"]] = s
    cid2name[r["pert_id"]] = r.get("cmap_name", "")

def norm_cid(c):
    """BRD-A01145011-001-01-4 -> BRD-A01145011; A07  (see paper) MOAR  (see paper)"""
    if c.startswith("BRD-"):
        m = re.match(r"^(BRD-[A-Z]\d+)", c)
        if m: return m.group(1)
    return c

# ---------- 3. GCTX  (see paper)U2OS (see paper) ----------
print("[1/4]  (see paper) GCTX  (see paper) A673/SKES1/TC32  (see paper)...")
t0 = time.time()
with h5py.File(str(GCTX), "r") as f:
    meta = f["0/META"]; col = meta["COL"]
    col_ids = [x.decode("utf-8","replace") for x in col["id"][...].tolist()]
    ds = f["0/DATA/0/matrix"]
    sel = {}  # cell -> list of col idx
    for i, sid in enumerate(col_ids):
        parts = sid.split(":")
        if len(parts) < 2: continue
        inst = parts[0]
        for k, t in CELLS.items():
            if t in inst:
                sel.setdefault(k, []).append(i)
                break
    all_idx = sorted(set().union(*[set(v) for v in sel.values()]))
    print(f"   (see paper): " + ", ".join(f"{k}={len(v)}" for k, v in sel.items()))
    #  (see paper) landmark  (see paper)
    with open(str(ROOT / "geneinfo_beta.txt"), encoding="utf-8") as f:
        gene_rows = list(csv.DictReader(f, delimiter="\t"))
    id2sym = {str(r["gene_id"]): r["gene_symbol"] for r in gene_rows}
    row_ids = [x.decode("utf-8","replace") for x in meta["ROW"]["id"][...].tolist()]
    lm_sym = [r["gene_symbol"] for r in gene_rows if r["feature_space"] == "landmark"]
    lm_set = set(lm_sym)
    lm_idx = [i for i, gid in enumerate(row_ids) if id2sym.get(str(gid), "").upper() in {x.upper() for x in lm_set}]
    print(f"  landmark  (see paper): {len(lm_idx)}")
    data = ds[all_idx, :][:, lm_idx]
    print(f"   (see paper) {data.shape}  (see paper) {time.time()-t0:.0f}s")
#  (see paper)
def pw_scores(gex, W):
    z = (gex - gex.mean(axis=1, keepdims=True)) / (gex.std(axis=1, keepdims=True) + 1e-6)
    return z @ W.T
W = np.asarray(pw_weight, dtype=np.float32)
pw_all = pw_scores(np.asarray(data, dtype=np.float32), W)  # (n, 3253)
#  (see paper)->(cell, cid)
sample_meta = []
for i, idx in enumerate(all_idx):
    parts = col_ids[idx].split(":")
    cell_k = next(k for k in CELLS if CELLS[k] in col_ids[idx].split(":")[0])
    sample_meta.append((cell_k, norm_cid(parts[1] if len(parts) >= 2 else "")))
print(f"   (see paper) {pw_all.shape}")

# ---------- 4. U2OS  (see paper) (241K npz) ----------
print("[2/4]  (see paper) U2OS  (see paper) (all_pw_data_241k.npz)...")
d = np.load(str(OUT / "all_pw_data_241k.npz"), allow_pickle=True)
pw241 = d["pw"]; cid241 = d["cid"]; cell241 = d["cell"]
u2os_mask = np.array(["U2OS" in c for c in cell241])
pw_u2os = pw241[u2os_mask]
cid_u2os = np.array([norm_cid(c) for c in cid241[u2os_mask]])
print(f"  U2OS  (see paper): {len(pw_u2os)},  (see paper): {len(set(cid_u2os))}")

# ---------- 5.  (see paper) ----------
def aggregate(cids, pws):
    dct = defaultdict(list)
    for c, p in zip(cids, pws):
        if c: dct[c].append(p)
    out = {}
    for c, lst in dct.items():
        arr = np.array(lst)
        out[c] = {"n": len(lst), "osteo": float(arr[:, core_osteo_idx].mean()), "pw": arr.mean(axis=0)}
    return out

agg_u2os = aggregate(cid_u2os, pw_u2os)
agg_other = defaultdict(dict)
for (cell_k, cid), p in zip(sample_meta, pw_all):
    if cid:
        agg_other[cell_k].setdefault(cid, []).append(p)
agg_other_final = {}
for k, dct in agg_other.items():
    agg_other_final[k] = aggregate(list(dct.keys()), [np.mean(v, axis=0) for v in dct.values()])
#  (see paper)U2OS
agg_other_merged = {}
for k in CELLS:
    for c, info in agg_other_final.get(k, {}).items():
        if c not in agg_other_merged:
            agg_other_merged[c] = {"n": 0, "osteo_list": [], "cells": []}
        agg_other_merged[c]["n"] += info["n"]
        agg_other_merged[c]["osteo_list"].append(info["osteo"])
        agg_other_merged[c]["cells"].append(k)
for c in agg_other_merged:
    agg_other_merged[c]["osteo"] = float(np.mean(agg_other_merged[c]["osteo_list"]))

# ---------- 6. model (see paper) ----------
print("[3/4] model (see paper) osteo  (see paper)...")
all_cids = set(agg_other_merged) | set(agg_u2os)
pred_osteo = {}
for cid in all_cids:
    smi = cid2smiles.get(cid)
    if not smi: continue
    o = predict_osteo(smi)
    if o is not None:
        pred_osteo[cid] = o
print(f"   (see paper): {len(pred_osteo)}")

# ---------- 7.  (see paper) ----------
print("[4/4]  (see paper)...")
res = {"note": " (see paper): model (see paper)U2OStrain,  (see paper)(A673/SKES1/TC32,  (see paper)) (see paper)",
       "cell_line_samples": {k: len(v) for k, v in sel.items()},
       "osteogenic_pathways": osteo_names}

def spearman(xs, ys):
    if len(xs) < 3: return None
    r, p = spearmanr(xs, ys)
    return {"r": round(float(r), 4), "p": round(float(p), 4), "n": len(xs)}

# 7a. U2OS vs  (see paper):  (see paper)
common = sorted(set(agg_u2os) & set(agg_other_merged))
common_f = [c for c in common if agg_other_merged[c]["n"] >= 1 and agg_u2os[c]["n"] >= 1]
u2os_v = [agg_u2os[c]["osteo"] for c in common_f]
oth_v = [agg_other_merged[c]["osteo"] for c in common_f]
res["u2os_vs_other"] = {"n_drugs": len(common_f), **({} if not common_f else spearman(u2os_v, oth_v))}
res["u2os_vs_other_pearson"] = ({} if len(common_f) < 3 else
    {"r": round(float(pearsonr(u2os_v, oth_v)[0]), 4), "p": round(float(pearsonr(u2os_v, oth_v)[1]), 4), "n": len(common_f)})

# 7b. model (see paper) vs  (see paper)
both_b = sorted(set(pred_osteo) & set(agg_other_merged))
pred_v = [pred_osteo[c] for c in both_b]
real_v = [agg_other_merged[c]["osteo"] for c in both_b]
res["model_vs_other_real"] = {"n_drugs": len(both_b), **({} if not both_b else spearman(pred_v, real_v))}
res["model_vs_other_pearson"] = ({} if len(both_b) < 3 else
    {"r": round(float(pearsonr(pred_v, real_v)[0]), 4), "p": round(float(pearsonr(pred_v, real_v)[1]), 4), "n": len(both_b)})
#  (see paper) ( (see paper))
if both_b:
    agree = sum(1 for p, r in zip(pred_v, real_v) if (p > 0) == (r > 0))
    res["model_vs_other_direction_agree"] = {"agree": agree, "n": len(both_b), "rate": round(agree/len(both_b), 4)}

# 7c. model (see paper) vs U2OS  (see paper) ( (see paper),  (see paper))
both_u = sorted(set(pred_osteo) & set(agg_u2os))
pred_uv = [pred_osteo[c] for c in both_u]
real_uv = [agg_u2os[c]["osteo"] for c in both_u]
res["model_vs_u2os_real"] = {"n_drugs": len(both_u), **({} if not both_u else spearman(pred_uv, real_uv))}

# 7d.  (see paper): model vs  (see paper)
res["per_cellline_model_vs_real"] = {}
for k in CELLS:
    bk = sorted(set(pred_osteo) & set(agg_other_final.get(k, {})))
    if len(bk) < 3: continue
    pv = [pred_osteo[c] for c in bk]; rv = [agg_other_final[k][c]["osteo"] for c in bk]
    r, p = spearmanr(pv, rv)
    res["per_cellline_model_vs_real"][k] = {"r": round(float(r),4), "p": round(float(p),4), "n": len(bk)}

# 7e.  (see paper): A673  (see paper)6h, SKES1/TC32  (see paper)24h ( (see paper)ID (see paper))
res["note_time"] = "A673=6h, SKES1/TC32=24h  (see paper)"

#  (see paper)
res["drug_table"] = []
for c in sorted(set(common_f) | set(both_b)):
    row = {"pert_id": c, "name": cid2name.get(c, c)}
    if c in agg_u2os: row["u2os_osteo"] = round(agg_u2os[c]["osteo"], 4); row["u2os_n"] = agg_u2os[c]["n"]
    if c in agg_other_merged: row["other_osteo"] = round(agg_other_merged[c]["osteo"], 4); row["other_n"] = agg_other_merged[c]["n"]; row["other_cells"] = agg_other_merged[c]["cells"]
    if c in pred_osteo: row["pred_osteo"] = round(pred_osteo[c], 4)
    res["drug_table"].append(row)

with open(str(OUT / "cross_cellline_results.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

# ---------- 8.  (see paper) ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
if len(u2os_v) >= 3 and len(oth_v) >= 3:
    ax = axes[0]
    ax.scatter(u2os_v, oth_v, alpha=0.6, s=30)
    r = spearman(u2os_v, oth_v)
    ax.set_title(f"U2OS vs  (see paper)\nSpearman r={r['r']} (n={len(u2os_v)})" if r else "U2OS vs  (see paper)")
    ax.set_xlabel("U2OS  (see paper)"); ax.set_ylabel(" (see paper)  (see paper)")
    lim = [min(u2os_v+oth_v), max(u2os_v+oth_v)]
    ax.plot(lim, lim, "k--", lw=1)
if len(pred_v) >= 3 and len(real_v) >= 3:
    ax = axes[1]
    ax.scatter(pred_v, real_v, alpha=0.6, s=30)
    r = spearman(pred_v, real_v)
    ax.set_title(f"model (see paper) vs  (see paper)\nSpearman r={r['r']} (n={len(pred_v)})" if r else "model vs  (see paper)")
    ax.set_xlabel("model (see paper) osteo  (see paper)"); ax.set_ylabel(" (see paper)  (see paper)")
    ax.axhline(0, color="grey", lw=0.8); ax.axvline(0, color="grey", lw=0.8)
if len(pred_uv) >= 3 and len(real_uv) >= 3:
    ax = axes[2]
    ax.scatter(pred_uv, real_uv, alpha=0.6, s=30)
    r = spearman(pred_uv, real_uv)
    ax.set_title(f"model (see paper) vs U2OS  (see paper)( (see paper))\nSpearman r={r['r']} (n={len(pred_uv)})" if r else "model vs U2OS")
    ax.set_xlabel("model (see paper) osteo  (see paper)"); ax.set_ylabel("U2OS  (see paper)")
    ax.axhline(0, color="grey", lw=0.8); ax.axvline(0, color="grey", lw=0.8)
fig.suptitle("PathBone v19  (see paper) (train: U2OS;  (see paper): A673/SKES1/TC32  (see paper))")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(str(FIG / "fig10_cross_cellline.pdf"))
fig.savefig(str(OUT / "figures" / "fig10_cross_cellline.png"), dpi=150)
print(" (see paper): outputs/cross_cellline_results.json +  (see paper)/fig10_cross_cellline.pdf")
print(json.dumps({k: v for k, v in res.items() if k not in ("drug_table",)}, ensure_ascii=False, indent=2))