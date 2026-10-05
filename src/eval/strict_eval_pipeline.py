# -*- coding: utf-8 -*-
"""PathBone 1.0  (see paper) v1
-  (see paper): ddpm.pt ( (see paper))
- PNI: 45  (see paper)test (see paper), train (see paper)test (see paper)
-  (see paper): Leave-One-Drug-Out / 5-fold CV,  (see paper)
-  (see paper): outputs/v20/strict_eval_v1.json
"""
import sys, json, csv, pathlib
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
ROOT=Path('G:/ (see paper)'); sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs'/'v20'
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from src import diffusion_vc, bone_axis

AXES={'osteoclast':'osteoclast_up','adipo':'adipo_up','chondro':'chondro_up','osteocyte':'osteocyte_up'}
DEVICE='cuda'
N_SAMPLES=32
STEPS=40

def load_base():
    meta=json.loads((OUT/'lincs_meta.json').read_text(encoding='utf-8'))
    smiles_map=json.loads((OUT/'smiles_map.json').read_text(encoding='utf-8'))
    diffusion_vc._SMILES_MAP=smiles_map
    vae=diffusion_vc.VAE(n_lm=len(meta['genes'])).to(DEVICE)
    vae.load_state_dict(torch.load(str(OUT/'vae.pt'),map_location=DEVICE,weights_only=False)['model'])
    mol_enc=diffusion_vc.GNNEncoder().to(DEVICE)
    mol_enc.load_state_dict(torch.load(str(OUT/'mol_enc.pt'),map_location=DEVICE,weights_only=False)['model'])
    ck=torch.load(str(OUT/'ddpm.pt'),map_location=DEVICE,weights_only=False)
    ddpm=diffusion_vc.CondDDPM(latent=64,cond_dim=258,hidden=ck.get('hidden',512),T=ck.get('T',400)).to(DEVICE)
    ddpm.load_state_dict(ck['model']); ddpm.z_mean=ck.get('z_mean'); ddpm.z_std=ck.get('z_std')
    return meta,smiles_map,vae,mol_enc,ddpm

def ecfp(smi,r=2,bits=2048):
    m=Chem.MolFromSmiles(smi)
    if m is None:return np.zeros(bits,dtype=np.float32)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(m,r,nBits=bits),dtype=np.float32)

def sample_feats(ddpm,vae,mol_enc,smiles_map,perts):
    feats={}
    for i,p in enumerate(perts):
        smi=smiles_map.get(p)
        if not smi: continue
        g=diffusion_vc.sample_drug(ddpm,vae,mol_enc,smi,n_samples=N_SAMPLES,steps=STEPS,device=DEVICE)
        if g is not None: feats[p]=g.astype(np.float32)
        if (i+1)%50==0: print(f'   (see paper) {i+1}/{len(perts)}', flush=True)
    return feats

def main():
    meta,smiles_map,vae,mol_enc,ddpm=load_base()
    pni_labels={k:int(v) for k,v in json.loads((ROOT/'outputs'/'labels_expanded.json').read_text(encoding='utf-8')).items()}
    tj=json.loads((ROOT/'outputs'/'test_set_drugs.json').read_text(encoding='utf-8'))
    test_ids=set(tj['promote_ids']+tj['inhibit_ids']+tj['neutral_ids'])
    truth={}
    for p in tj['promote_ids']: truth[p]=0
    for p in tj['inhibit_ids']: truth[p]=1
    for p in tj['neutral_ids']: truth[p]=2
    multi=json.loads((ROOT/'outputs'/'multi_axis_labels_pert_v5.json').read_text(encoding='utf-8'))
    all_perts=sorted(set(list(pni_labels.keys())+[p for p in test_ids]+[p for p,axs in multi.items() if any(k in axs for k in AXES)]))
    print(' (see paper)',len(all_perts), flush=True)
    feats=sample_feats(ddpm,vae,mol_enc,smiles_map,all_perts)
    names=sorted(feats)
    np.savez_compressed(OUT/'strict_feats_base.npz', feats=np.array([feats[p] for p in names]), names=np.array(names))
    print(' (see paper)',len(names), flush=True)
    # id->drug
    with open(ROOT/'compoundinfo_beta.txt',encoding='utf-8') as f: rows=list(csv.DictReader(f,delimiter='\t'))
    id2name={r['pert_id']:(r.get('cmap_name') or '').lower() for r in rows}
    report={'n_features':len(feats),'pni':{},'axes':{}}
    def make_X(ps,use_ecfp=True):
        X=[]
        for p in ps:
            g=feats[p]; f=ecfp(smiles_map.get(p,''))
            X.append(np.concatenate([g,f]))
        return np.array(X)
    # PNI head 5fold on non-test features, evaluate test
    train_p=[p for p in pni_labels if p not in test_ids and p in feats]
    test_p=[p for p in test_ids if p in feats]
    Ytr=np.array([pni_labels[p] for p in train_p]); Yte=np.array([truth[p] for p in test_p])
    Xtr=make_X(train_p); Xte=make_X(test_p)
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=2000,class_weight='balanced'))
    clf.fit(Xtr,Ytr)
    pr=clf.predict_proba(Xte); ote=pr[:,0]-pr[:,1]
    # threshold by train 5fold CV median same logic? simple grid on train predictions via CV
    from sklearn.model_selection import cross_val_predict
    prtr=cross_val_predict(clf,Xtr,Ytr,cv=StratifiedKFold(5,shuffle=True,random_state=42),method='predict_proba')
    otr=prtr[:,0]-prtr[:,1]
    best=(-1,None,None)
    for t1 in np.arange(-0.5,1.0,0.05):
        for t2 in np.arange(-1.0,0.5,0.05):
            if t1<t2: continue
            pred=np.where(otr>t1,0,np.where(otr<t2,1,2))
            acc=np.mean(pred==Ytr)
            if acc>best[0]: best=(acc,t1,t2)
    _,t1,t2=best
    pred=np.where(ote>t1,0,np.where(ote<t2,1,2))
    pni_acc=float(accuracy_score(Yte,pred))
    pni_auc3=float(np.mean([roc_auc_score((Yte==c).astype(int),pr[:,c]) for c in range(3)]))
    mask=Yte!=2; pni_auc_pi=float(roc_auc_score((Yte[mask]==0).astype(int),ote[mask]))
    report['pni']={'acc':pni_acc,'auc3':pni_auc3,'auc_pi':pni_auc_pi,'n_train':len(train_p),'n_test':len(test_p)}
    print('PNI',report['pni'], flush=True)
    # axes LOOCV and 5fold
    for ax,anchor in AXES.items():
        pos=[p for p,v in multi.items() if v.get(ax)==1 and p in feats]
        neg=[p for p,v in multi.items() if v.get(ax)==-1 and p in feats]
        ps=pos+neg; y=np.array([1]*len(pos)+[0]*len(neg)); groups=np.array([id2name.get(p,p) for p in ps])
        X=make_X(ps)
        prof=drug=None
        try:
            predp=cross_val_predict(make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=2000,class_weight='balanced')),X,y,cv=StratifiedKFold(5,shuffle=True,random_state=42),method='predict_proba')[:,1]
            prof=float(roc_auc_score(y,predp))
        except Exception: pass
        from collections import defaultdict
        gs=defaultdict(list)
        for x,label,g in zip(X,y,groups): gs[g].append((x,label))
        uniq=list(gs); Xg=np.array([np.stack([a for a,l in gs[g]]).mean(0) for g in uniq]); yg=np.array([gs[g][0][1] for g in uniq])
        try:
            predg=cross_val_predict(make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=2000,class_weight='balanced')),Xg,yg,cv=LeaveOneGroupOut().split(Xg,yg,groups=uniq),method='predict_proba')[:,1]
            drug=float(roc_auc_score(yg,predg))
        except Exception: pass
        report['axes'][ax]={'pos_profiles':len(pos),'neg_profiles':len(neg),'unique_pos':len(set(groups[y==1])),'unique_neg':len(set(groups[y==0])),'profile_cv_auc':prof,'drug_loocv_auc':drug}
        print('AXIS',ax,report['axes'][ax], flush=True)
    (OUT/'strict_eval_v1.json').write_text(json.dumps(report,ensure_ascii=False,indent=1),encoding='utf-8')
    print('saved',OUT/'strict_eval_v1.json')

if __name__=='__main__': main()