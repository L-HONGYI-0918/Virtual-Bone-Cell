# -*- coding: utf-8 -*-
"""Scaffold Split v3: 292  (see paper),  (see paper) v19  (see paper) scaffold- (see paper)"""
import json, csv, torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
from torch_geometric.data import Data, Batch
import torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

ROOT = Path("G:/ (see paper)"); OUT = ROOT / "outputs"; DEV = "cuda"

#  (see paper)data
with open(str(ROOT / "compoundinfo_beta.txt"), encoding="utf-8") as f:
    comp_rows = list(csv.DictReader(f, delimiter="\t"))
cid2smiles = {}; cid2name = {}
for r in comp_rows:
    s = r.get("canonical_smiles", "").strip()
    if s and s != "restricted": cid2smiles[r["pert_id"]] = s
    cid2name[r["pert_id"]] = r.get("cmap_name", "")

# 292  (see paper)
labeled = {}
with open(str(OUT / "compound_bone_labels.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        lbl = r.get("bone_label", "")
        if lbl in ("0", "1", "2"):
            labeled[r["pert_id"]] = {0: 0, 1: 1, 2: 0}[int(lbl)]  # CSV: 2= (see paper) →  (see paper) 0

with open(str(OUT / "test_set_drugs.json"), encoding="utf-8") as f:
    test_info = json.load(f)
test_cids = set(test_info["promote_ids"] + test_info["inhibit_ids"] + test_info["neutral_ids"])

def scaffold_of(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except:
        return None

# scaffold  (see paper)
sc2cids = defaultdict(list)
for cid in labeled:
    smi = cid2smiles.get(cid)
    sc = scaffold_of(smi) if smi else None
    if sc:
        sc2cids[sc].append(cid)
print(f" (see paper): {len(labeled)},  (see paper)scaffold: {sum(len(v) for v in sc2cids.values())},  (see paper)scaffold: {len(sc2cids)}")

#  (see paper) scaffold ( (see paper) = scaffold  (see paper))
singleton_sc = {sc: cids[0] for sc, cids in sc2cids.items() if len(cids) == 1}
print(f" (see paper) scaffold  (see paper): {len(singleton_sc)}")

# model
ckpt = torch.load(str(OUT / "bone_vc_v19.pt"), map_location=DEV, weights_only=False)
cfg = ckpt["config"]; core_idx = ckpt["core_osteo_idx"]; npw = cfg["npw"]

AT = ["C","N","O","S","F","Cl","P","Br","I","B","Si","Se","other"]
def af(a):
    sym = a.GetSymbol(); ti = AT.index(sym) if sym in AT else 12
    toh = [0]*len(AT); toh[ti] = 1
    d = min(a.GetDegree(), 6); doh = [0]*7; doh[d] = 1
    h = str(a.GetHybridization()); hyb = ["SP","SP2","SP3","SP3D","SP3D2"]; hoh = [0]*5
    for j, x in enumerate(hyb):
        if x in h: hoh[j] = 1; break
    return toh + doh + hoh + [int(a.IsInRing()), int(a.IsInRingSize(6)), int(a.GetIsAromatic()), min(a.GetTotalNumHs(), 4)/4.0]
def mtg(s):
    mol = Chem.MolFromSmiles(s)
    if mol is None: return None
    mol = Chem.AddHs(mol); x = [af(a) for a in mol.GetAtoms()]
    if not x: return None
    x = torch.tensor(x, dtype=torch.float32); ei = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx(); j = b.GetEndAtomIdx()
        ei.append([i, j]); ei.append([j, i])
    if not ei: ei = [[0, 0]]
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
        x = F.relu(self.ln1(self.conv1(x, ei)))
        x = F.relu(self.ln2(self.conv2(x, ei)))
        x = F.relu(self.ln3(self.conv3(x, ei)))
        return torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)

class Model(nn.Module):
    def __init__(self, npw, n_lm):
        super().__init__()
        self.gnn = GNNEncoder()
        self.head = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, npw))
        self.register_buffer("pw2gene", torch.zeros(npw, n_lm))
    def forward(self, bg):
        z = self.gnn(bg); pw = self.head(z); return pw, pw @ self.pw2gene

model = Model(npw, cfg["n_landmark"]).to(DEV)
model.load_state_dict(ckpt["model"]); model.eval()

THRESH_PI = 0.3960; THRESH_IN = 0.3488
def classify(o):
    if o > THRESH_PI: return 1
    elif o > THRESH_IN: return -1
    else: return 0
LBL = {1: "P", -1: "I", 0: "N"}

#  (see paper) scaffold- (see paper) ( (see paper) scaffold)  (see paper)test,  (see paper) 15  (see paper)test (see paper)
novel_cids = [cid for cid in singleton_sc.values() if cid not in test_cids]
print(f"\nscaffold- (see paper)( (see paper)test (see paper)): {len(novel_cids)}")
print(" (see paper) v19  (see paper) vs  (see paper)...")

#  (see paper)
per_label = {1: {"correct": 0, "total": 0}, -1: {"correct": 0, "total": 0}, 0: {"correct": 0, "total": 0}}
results_rows = []
for cid in novel_cids:
    smi = cid2smiles.get(cid)
    g = mtg(smi) if smi else None
    if g is None: continue
    with torch.no_grad():
        pw, _ = model(g.to(DEV))
        o = pw[0, core_idx].mean().item()
    pred = classify(o)
    true = labeled[cid]
    ok = pred == true
    per_label[true]["total"] += 1
    if ok: per_label[true]["correct"] += 1
    results_rows.append({"name": cid2name.get(cid, cid), "cid": cid, "true": true,
                         "pred": pred, "osteo": round(o, 4), "ok": ok})
    if not ok:
        print(f"  XX {cid2name.get(cid, cid):<25s} true={LBL[true]} pred={LBL[pred]} osteo={o:+.4f}")

total_c = sum(v["correct"] for v in per_label.values())
total_t = sum(v["total"] for v in per_label.values())
print(f"\n=== Scaffold- (see paper) ( (see paper)) ===")
for lbl in [1, -1, 0]:
    s = per_label[lbl]
    print(f"  {LBL[lbl]}: {s['correct']}/{s['total']} = {s['correct']/max(1,s['total'])*100:.1f}%")
print(f"   (see paper): {total_c}/{total_t} = {total_c/max(1,total_t)*100:.1f}%")

with open(str(OUT / "scaffold_novel_results.json"), "w", encoding="utf-8") as f:
    json.dump({"per_label": {str(k): v for k, v in per_label.items()},
               "total_correct": total_c, "total": total_t,
               "rows": results_rows}, f, ensure_ascii=False, indent=2)
print(f"\n[OK]  (see paper): outputs/scaffold_novel_results.json")