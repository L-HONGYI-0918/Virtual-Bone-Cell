# -*- coding: utf-8 -*-
"""PathBone 1.0 135 (see paper)test -  (see paper)"""
import sys, json, csv, pathlib
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
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
def build_gnn_cache(ids,smiles_map):
    cache_file=OUT/'strict_gnn_feats.npz'; cache={}
    if cache_file.exists():
        d=np.load(cache_file); cache={str(n):v.astype(np.float32) for n,v in zip(d['names'],d['feats'])}
    todo=[p for p in ids if p not in cache and smiles_map.get(p)]
    if todo:
        mol_enc=diffusion_vc.GNNEncoder().to('cuda'); mol_enc.load_state_dict(torch.load(str(OUT/'mol_enc.pt'),map_location='cuda',weights_only=False)['model']); mol_enc.eval()
        graphs=[]; ok=[]
        for p in todo:
            g=diffusion_vc.mol_to_graph(smiles_map[p])
            if g is not None: graphs.append(g); ok.append(p)
        with torch.no_grad():
            for s in range(0,len(graphs),128):
                batch=Batch.from_data_list(graphs[s:s+128]).to('cuda'); emb=mol_enc(batch).cpu().numpy().astype(np.float32)
                for p,e in zip(ok[s:s+128],emb): cache[p]=e
        np.savez_compressed(cache_file,feats=np.array([cache[p] for p in sorted(cache)]),names=np.array(sorted(cache)))
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
def make_X(ps,gene_feats,gnn_feats,smiles_map,mode):
    ok=[p for p in ps if feat_vec(p,gene_feats,gnn_feats,smiles_map,mode) is not None]
    return ok,np.array([feat_vec(p,gene_feats,gnn_feats,smiles_map,mode) for p in ok],dtype=np.float32)
def clf_factory(kind,seed=0):
    if kind=='lr': return make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=5000,class_weight='balanced'))
    if kind=='svc': return make_pipeline(StandardScaler(),SVC(C=1.0,gamma='scale',probability=True,class_weight='balanced',random_state=seed))
    if kind=='rf2': return RandomForestClassifier(n_estimators=1200,max_features=None,min_samples_leaf=2,class_weight='balanced_subsample',n_jobs=-1,random_state=seed)
    if kind=='rf': return RandomForestClassifier(n_estimators=800,max_features='sqrt',min_samples_leaf=1,class_weight='balanced_subsample',n_jobs=-1,random_state=seed)

def main():
    d=np.load(str(OUT/'strict_feats_base.npz')); gene_feats={str(n):v.astype(np.float32) for n,v in zip(d['names'],d['feats'])}
    smiles_map=json.loads((OUT/'smiles_map.json').read_text(encoding='utf-8')); gnn_feats=build_gnn_cache(list(gene_feats),smiles_map)
    pni_labels={k:int(v) for k,v in json.loads((ROOT/'outputs'/'labels_expanded.json').read_text(encoding='utf-8')).items()}
    tj=json.loads((ROOT/'outputs'/'test_set_drugs_135.json').read_text(encoding='utf-8'))
    test_ids=set(tj['promote_ids']+tj['inhibit_ids']+tj['neutral_ids']); truth={}
    for p in tj['promote_ids']: truth[p]=0
    for p in tj['inhibit_ids']: truth[p]=1
    for p in tj['neutral_ids']: truth[p]=2
    with open(ROOT/'compoundinfo_beta.txt',encoding='utf-8') as f: rows=list(csv.DictReader(f,delimiter='\t'))
    id2name={r['pert_id']:(r.get('cmap_name') or r.get('pert_id') or '').lower() for r in rows}
    train_p=[p for p in pni_labels if p not in test_ids and p in gene_feats]
    test_p=[p for p in test_ids if p in gene_feats]
    comps=[('rf2','desc+ecfp3'),('lr','desc+gnn+ecfp3+gene'),('svc','desc+ecfp3'),('lr','desc+ecfp3+gene'),('rf','desc')]
    cv_ps=[]; te_ps=[]; te_ids=[]
    for kind,mode in comps:
        oktr,Xtr=make_X(train_p,gene_feats,gnn_feats,smiles_map,mode); Ytr=np.array([pni_labels[p] for p in oktr])
        okte,Xte=make_X(test_p,gene_feats,gnn_feats,smiles_map,mode); Yte2=np.array([truth[p] for p in okte])
        clf=clf_factory(kind)
        cvp=cross_val_predict(clf,Xtr,Ytr,cv=StratifiedKFold(5,shuffle=True,random_state=42),method='predict_proba')
        clf.fit(Xtr,Ytr); pte=clf.predict_proba(Xte)
        cv_ps.append(cvp); te_ps.append(pte); te_ids.append(okte)
    # train meta on CV probs
    Ztr=np.concatenate(cv_ps,axis=1); Zte=np.concatenate(te_ps,axis=1)
    meta=LogisticRegression(C=0.03,max_iter=5000,class_weight='balanced').fit(Ztr,Ytr)
    ptr=meta.predict_proba(Ztr); pte=meta.predict_proba(Zte)
    otr=ptr[:,0]-ptr[:,1]; ote=pte[:,0]-pte[:,1]
    best=(-1,None,None)
    for t1 in np.arange(-0.75,1.0,0.05):
        for t2 in np.arange(-1.0,0.75,0.05):
            if t1<t2: continue
            pred=np.where(otr>t1,0,np.where(otr<t2,1,2)); acc=np.mean(pred==Ytr)
            if acc>best[0]: best=(acc,t1,t2)
    _,t1,t2=best
    # aggregate per drug name
    drug_ps=defaultdict(list)
    for i,p in enumerate(te_ids[0]):
        drug_ps[id2name.get(p,p)].append(i)
    names=[]; yg=[]; pg=[]
    for name,idxs in sorted(drug_ps.items()):
        # majority label from truth
        labs=[truth[te_ids[0][i]] for i in idxs]
        lab=max(set(labs),key=labs.count)
        names.append(name); yg.append(lab); pg.append(pte[idxs].mean(0))
    pg=np.array(pg); yg=np.array(yg)
    ote=pg[:,0]-pg[:,1]; pred=np.where(ote>t1,0,np.where(ote<t2,1,2))
    acc=accuracy_score(yg,pred); mf1=f1_score(yg,pred,average='macro'); cm=confusion_matrix(yg,pred,labels=[0,1,2]).tolist()
    p,r,f,s=precision_recall_fscore_support(yg,pred,labels=[0,1,2],average=None)
    auc3=np.mean([roc_auc_score((yg==c).astype(int),pg[:,c]) for c in range(3)])
    mask=yg!=2; auc_pi=roc_auc_score((yg[mask]==0).astype(int),ote[mask])
    out={'test_unique_drugs':len(names),'acc':float(acc),'macro_f1':float(mf1),'auc3':float(auc3),'auc_pi':float(auc_pi),'thresholds':[float(t1),float(t2)],'cm':cm,'precision':p.tolist(),'recall':r.tolist(),'f1':f.tolist(),'support':s.tolist(),'classes':['P','I','N'],'n_train_profiles':len(Ytr),'n_test_profiles':len(Yte2)}
    (OUT/'strict_eval_release_metrics_135_drug.json').write_text(json.dumps(out,ensure_ascii=False,indent=1),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=1))
if __name__=='__main__': main()