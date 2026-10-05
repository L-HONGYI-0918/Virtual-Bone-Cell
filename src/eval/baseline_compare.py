# -*- coding: utf-8 -*-
import numpy as np, json, csv, torch
from pathlib import Path
from rdkit import Chem; from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
from rdkit.Chem import AllChem, Descriptors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = Path("G:/ (see paper)"); OUT = ROOT / "outputs"
with open(str(OUT / "test_set_drugs.json"), encoding="utf-8") as f: test_info = json.load(f)
with open(str(ROOT / "compoundinfo_beta.txt"), encoding="utf-8") as f: comp_rows = list(csv.DictReader(f, delimiter="\t"))
cid2smiles = {}; cid2name = {}
for r in comp_rows:
    s = r.get("canonical_smiles","").strip()
    if s and s != "restricted": cid2smiles[r["pert_id"]] = s
    cid2name[r["pert_id"]] = r.get("cmap_name","")

pw_all = np.load(str(OUT / "pw_all_241k.npy"), mmap_mode="r")
cid_all = np.load(str(OUT / "cid_all_241k.npy"), allow_pickle=True)
ckpt = torch.load(str(OUT / "bone_vc_v19.pt"), map_location="cpu", weights_only=False)
core_osteo_idx = ckpt["core_osteo_idx"]

# Build test set
test_drugs = []
for label_val, cids in [(1, test_info["promote_ids"]), (-1, test_info["inhibit_ids"]), (0, test_info["neutral_ids"])]:
    for cid in cids:
        smi = cid2smiles.get(cid)
        if smi: test_drugs.append({"name": cid2name.get(cid, cid), "smiles": smi, "label": label_val})
test_cids = set(test_info["promote_ids"] + test_info["inhibit_ids"] + test_info["neutral_ids"])

# Build training set (exclude test)
np.random.seed(42); n_train = min(5000, len(cid_all))
all_idx = np.random.choice(len(cid_all), n_train, replace=False)
train_smi = []; train_oste = []
for i in all_idx:
    cid = str(cid_all[i])
    if cid in test_cids: continue
    smi = cid2smiles.get(cid)
    if smi:
        train_smi.append(smi)
        train_oste.append(pw_all[i][core_osteo_idx].mean())
train_oste = np.array(train_oste)
print(f"Training set: {len(train_smi)} samples")

def compute_ecfp(smi, radius=3, nbits=1024):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return np.array(fp, dtype=np.float32)

def compute_desc(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    return np.array([Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol), Descriptors.RingCount(mol),
        Descriptors.TPSA(mol), Descriptors.FractionCSP3(mol),
        Descriptors.NumAromaticRings(mol)], dtype=np.float32)

print("Computing features...")
ecfp_train = np.array([compute_ecfp(s) for s in train_smi])
mfp_train = np.array([compute_ecfp(s, radius=2, nbits=2048) for s in train_smi])
desc_train = np.array([compute_desc(s) for s in train_smi])
ecfp_test = np.array([compute_ecfp(d["smiles"]) for d in test_drugs])
mfp_test = np.array([compute_ecfp(d["smiles"], radius=2, nbits=2048) for d in test_drugs])
desc_test = np.array([compute_desc(d["smiles"]) for d in test_drugs])

# PathBone Osteo scores (from eval)
pb_osteos = [0.4865, 0.4793, 0.2218, 0.4830, 0.4786, 0.0588, 0.4528, 0.3928, -0.4474, 0.3610, 0.1652, 0.0547, 0.3002, 0.0016, -0.0004]
test_labels = [d["label"] for d in test_drugs]

results = {}
thresh_pi = 0.3960; thresh_in = 0.3488

def classify(o):
    if o > thresh_pi: return 1
    elif o > thresh_in: return -1
    else: return 0

def evaluate(name, oste_preds):
    correct = sum(1 for d, o in zip(test_drugs, oste_preds) if classify(o) == d["label"])
    labels_pi = [1 if d["label"]==1 else 0 for d in test_drugs if d["label"]!=0]
    scores_pi = [o for d, o in zip(test_drugs, oste_preds) if d["label"]!=0]
    auroc = roc_auc_score(labels_pi, scores_pi) if len(set(labels_pi))>1 else 0
    return correct/15, auroc

# B1: ECFP+MLP
print("B1: ECFP(r=3)+MLP...")
s1=StandardScaler(); X1=s1.fit_transform(ecfp_train); Xt1=s1.transform(ecfp_test)
m1=MLPRegressor(hidden_layer_sizes=(512,256),max_iter=500,random_state=42,early_stopping=True)
m1.fit(X1,train_oste); o1=m1.predict(Xt1)
acc1,auc1=evaluate("ECFP+MLP",o1); results["ECFP(r=3)+MLP"]={"acc":acc1,"auroc":auc1,"oste":o1.tolist()}
print(f"  Acc={acc1:.1%} AUROC={auc1:.4f}")

# B2: MorganFP+MLP
print("B2: MorganFP(r=2)+MLP...")
s2=StandardScaler(); X2=s2.fit_transform(mfp_train); Xt2=s2.transform(mfp_test)
m2=MLPRegressor(hidden_layer_sizes=(512,256),max_iter=500,random_state=42,early_stopping=True)
m2.fit(X2,train_oste); o2=m2.predict(Xt2)
acc2,auc2=evaluate("MorganFP+MLP",o2); results["MorganFP(r=2)+MLP"]={"acc":acc2,"auroc":auc2,"oste":o2.tolist()}
print(f"  Acc={acc2:.1%} AUROC={auc2:.4f}")

# B3: Desc+MLP (no GNN)
print("B3: Desc+MLP(no GNN)...")
s3=StandardScaler(); X3=s3.fit_transform(desc_train); Xt3=s3.transform(desc_test)
m3=MLPRegressor(hidden_layer_sizes=(256,128),max_iter=500,random_state=42,early_stopping=True)
m3.fit(X3,train_oste); o3=m3.predict(Xt3)
acc3,auc3=evaluate("Desc+MLP",o3); results["Desc+MLP(no_GNN)"]={"acc":acc3,"auroc":auc3,"oste":o3.tolist()}
print(f"  Acc={acc3:.1%} AUROC={auc3:.4f}")

# B4: Larger MLP on ECFP (simulating no pathway bottleneck)
print("B4: ECFP+LargeMLP(no pathway)...")
s4=StandardScaler(); X4=s4.fit_transform(ecfp_train); Xt4=s4.transform(ecfp_test)
m4=MLPRegressor(hidden_layer_sizes=(1024,512,256),max_iter=800,random_state=42,early_stopping=True)
m4.fit(X4,train_oste); o4=m4.predict(Xt4)
acc4,auc4=evaluate("ECFP+LargeMLP",o4); results["ECFP+LargeMLP(noPW)"]={"acc":acc4,"auroc":auc4,"oste":o4.tolist()}
print(f"  Acc={acc4:.1%} AUROC={auc4:.4f}")

# PathBone (with calibrated thresholds)
acc_pb = sum(1 for d, o in zip(test_drugs, pb_osteos) if classify(o) == d["label"])
lp = [1 if d["label"]==1 else 0 for d in test_drugs if d["label"]!=0]
sp = [o for d, o in zip(test_drugs, pb_osteos) if d["label"]!=0]
auc_pb = roc_auc_score(lp, sp) if len(set(lp))>1 else 0

print("")
print("="*70)
print("BASELINE COMPARISON (15-drug Independent Test Set)")
print("="*70)
print(f"{chr(40)}Method{chr(41):<30s} {chr(40)}Acc{chr(41):>8s} {chr(40)}AUROC{chr(41):>8s}")
print("-"*46)
print(f"{chr(40)}PathBone v19 (GNN+Pathway){chr(41):<30s} {acc_pb:>7.1%} {auc_pb:>8.4f}")
for name, r in results.items():
    print(f"{name:<30s} {r[chr(39)+chr(97)+chr(99)+chr(99)+chr(39)]:>7.1%} {r[chr(39)+chr(97)+chr(117)+chr(114)+chr(111)+chr(99)+chr(39)]:>8.4f}")

print("")
label_names = {1: "P", -1: "I", 0: "N"}
print(f"{chr(40)}Drug{chr(41):<30s} {chr(40)}True{chr(41):>4s} {chr(40)}PathBone{chr(41):>8s} {chr(40)}ECFP+MLP{chr(41):>10s} {chr(40)}MFP+MLP{chr(41):>10s} {chr(40)}Desc+MLP{chr(41):>10s} {chr(40)}LargeMLP{chr(41):>10s}")
print("-"*80)
for i, d in enumerate(test_drugs):
    print(f"{d[chr(39)+chr(110)+chr(97)+chr(109)+chr(101)+chr(39)]:<30s} {label_names[d[chr(39)+chr(108)+chr(97)+chr(98)+chr(101)+chr(108)+chr(39)]]:>4s} {pb_osteos[i]:>+8.4f} {o1[i]:>+10.4f} {o2[i]:>+10.4f} {o3[i]:>+10.4f} {o4[i]:>+10.4f}")

print("")
print("Done!")



