# -*- coding: utf-8 -*-
"""PathBone v8:  (see paper) 5  (see paper) LOOCV"""
import sys, json, csv, pathlib, time
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
ROOT=Path('G:/ (see paper)'); sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs'/'v20'
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import torch
from torch_geometric.data import Batch
from src import diffusion_vc
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def ecfp(smi,r=2,bits=2048):
    m=Chem.MolFromSmiles(smi)
    if m is None: return np.zeros(bits,dtype=np.float32)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(m,r,nBits=bits),dtype=np.float32)

def desc(smi):
    m=Chem.MolFromSmiles(smi)
    if m is None: return np.zeros(217,dtype=np.float32)
    vals=[]
    for name,fn in Descriptors.descList:
        try: v=fn(m)
        except Exception: v=0.0
        if v is None or not np.isfinite(v): v=0.0
        vals.append(float(v))
    return np.array(vals,dtype=np.float32)

def build_gnn_cache(ids, smiles_map, device='cuda'):
    cache_file=OUT/'strict_gnn_feats.npz'; cache={}
    if cache_file.exists():
        d=np.load(cache_file); cache={str(n):v.astype(np.float32) for n,v in zip(d['names'],d['feats'])}
    todo=[p for p in ids if p not in cache and smiles_map.get(p)]
    if todo:
        mol_enc=diffusion_vc.GNNEncoder().to(device); mol_enc.load_state_dict(torch.load(str(OUT/'mol_enc.pt'),map_location=device,weights_only=False)['model']); mol_enc.eval()
        graphs=[]; ok=[]
        for p in todo:
            g=diffusion_vc.mol_to_graph(smiles_map[p])
            if g is not None: graphs.append(g); ok.append(p)
        with torch.no_grad():
            for s in range(0,len(graphs),128):
                batch=Batch.from_data_list(graphs[s:s+128]).to(device); emb=mol_enc(batch).cpu().numpy().astype(np.float32)
                for p,e in zip(ok[s:s+128],emb): cache[p]=e
        np.savez_compressed(cache_file, feats=np.array([cache[p] for p in sorted(cache)]), names=np.array(sorted(cache)))
    return cache

def feat_vec(p,gene_feats,gnn_feats,smiles_map,mode):
    smi=smiles_map.get(p,''); parts=[]
    if 'gene' in mode: parts.append(gene_feats.get(p))
    if 'gnn' in mode: parts.append(gnn_feats.get(p))
    if 'ecfp2' in mode: parts.append(ecfp(smi,2,2048))
    if 'ecfp3' in mode: parts.append(ecfp(smi,3,2048))
    if 'desc' in mode: parts.append(desc(smi))
    if any(x is None for x in parts): return None
    return np.concatenate(parts).astype(np.float32)

def clf_factory(kind,seed):
    if kind=='lr': return make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=5000,class_weight='balanced'))
    if kind=='svc': return make_pipeline(StandardScaler(),SVC(C=1.0,gamma='scale',probability=True,class_weight='balanced',random_state=seed))
    if kind=='rf2': return RandomForestClassifier(n_estimators=1200,max_features=None,min_samples_leaf=2,class_weight='balanced_subsample',n_jobs=-1,random_state=seed)
    if kind=='rf': return RandomForestClassifier(n_estimators=800,max_features='sqrt',min_samples_leaf=1,class_weight='balanced_subsample',n_jobs=-1,random_state=seed)
    raise ValueError

def drug_mat(ps_labels,gene_feats,gnn_feats,smiles_map,id2name,mode):
    by=defaultdict(list)
    for p,lab in ps_labels.items():
        if p not in gene_feats: continue
        by[id2name.get(p,p)].append((p,lab))
    drugs=[]; y=[]; X=[]
    for name,items in sorted(by.items()):
        labs=[l for p,l in items]
        if len(set(labs))==2:
            if labs.count(1)==labs.count(-1): continue
            lab=1 if labs.count(1)>labs.count(-1) else 0
        else: lab=1 if labs[0]==1 else 0
        ps=[p for p,l in items]
        vecs=[feat_vec(p,gene_feats,gnn_feats,smiles_map,mode) for p in ps]
        if any(v is None for v in vecs): continue
        X.append(np.stack(vecs).mean(0)); drugs.append(name); y.append(lab)
    return np.array(drugs),np.array(y),np.array(X,dtype=np.float32)

def eval_axis_ensemble(ps_labels,gene_feats,gnn_feats,smiles_map,id2name):
    out={}
    for mode in ['desc+ecfp3','desc+gnn+ecfp3+gene','desc']:
        drugs,y,X=drug_mat(ps_labels,gene_feats,gnn_feats,smiles_map,id2name,mode)
        if len(np.unique(y))<2 or len(drugs)<3:
            out[mode]={'n_drugs':len(drugs),'auc':None}; continue
        for kind in ['lr','svc','rf2','rf']:
            clf=clf_factory(kind,0)
            try:
                pred=cross_val_predict(clf,X,y,cv=LeaveOneGroupOut().split(X,y,groups=drugs),method='predict_proba')[:,1]
                auc=float(roc_auc_score(y,pred))
            except Exception as e:
                auc=None
            out[f'{mode}_{kind}']={'n_drugs':len(drugs),'n_pos':int(y.sum()),'n_neg':int((1-y).sum()),'auc':auc}
            print('AXIS component',mode,kind,out[f'{mode}_{kind}'], flush=True)
    return out

def main():
    d=np.load(str(OUT/'strict_feats_base.npz')); gene_feats={str(n):v.astype(np.float32) for n,v in zip(d['names'],d['feats'])}
    smiles_map=json.loads((OUT/'smiles_map.json').read_text(encoding='utf-8')); gnn_feats=build_gnn_cache(list(gene_feats),smiles_map)
    pni_labels={k:int(v) for k,v in json.loads((ROOT/'outputs'/'labels_expanded.json').read_text(encoding='utf-8')).items()}
    multi=json.loads((ROOT/'outputs'/'multi_axis_labels_pert_v5.json').read_text(encoding='utf-8'))
    with open(ROOT/'compoundinfo_beta.txt',encoding='utf-8') as f: rows=list(csv.DictReader(f,delimiter='\t'))
    id2name={r['pert_id']:(r.get('cmap_name') or r.get('pert_id') or '').lower() for r in rows}
    axes={}
    for ax in ['osteo','osteoclast','adipo','chondro','osteocyte']:
        if ax=='osteo':
            ps={p:(1 if pni_labels[p]==0 else -1) for p in pni_labels if p in gene_feats and pni_labels[p] in (0,1)}
        else:
            ps={p:v[ax] for p,v in multi.items() if ax in v and p in gene_feats and v[ax] in (1,-1)}
        axes[ax]=eval_axis_ensemble(ps,gene_feats,gnn_feats,smiles_map,id2name)
    (OUT/'strict_eval_v8_axes.json').write_text(json.dumps(axes,ensure_ascii=False,indent=1),encoding='utf-8')
    print('saved',OUT/'strict_eval_v8_axes.json')

if __name__=='__main__': main()