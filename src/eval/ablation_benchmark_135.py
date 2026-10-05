# -*- coding: utf-8 -*-
import os, json, csv, pickle, sys
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
ROOT=Path(os.environ['VIRTUAL_BONE']); OUT=ROOT/'outputs'/'v20'
sys.path.insert(0,str(ROOT))
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix,precision_recall_fscore_support
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

feats=pickle.load(open(str(OUT/'abcde_feats.pkl'),'rb'))
labels={k:int(v) for k,v in json.loads((ROOT/'outputs'/'labels_expanded.json').read_text(encoding='utf-8')).items()}
tj=json.loads((ROOT/'outputs'/'test_set_drugs_135.json').read_text(encoding='utf-8'))
test_ids=set(tj['promote_ids']+tj['inhibit_ids']+tj['neutral_ids']); truth={}
for p in tj['promote_ids']: truth[p]=0
for p in tj['inhibit_ids']: truth[p]=1
for p in tj['neutral_ids']: truth[p]=2
rows=list(csv.DictReader(open(ROOT/'compoundinfo_beta.txt',encoding='utf-8'),delimiter='\t'))
id2name={r['pert_id']:(r.get('cmap_name') or r.get('pert_id') or '').lower() for r in rows}
train=[p for p in labels if p not in test_ids and p in feats and feats[p].get('smiles')]
test=[p for p in test_ids if p in feats and feats[p].get('smiles')]

def morgan2(smi):
    mol=Chem.MolFromSmiles(smi)
    if mol is None: return None
    fp=AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048)
    arr=np.zeros((1,),dtype=np.int8); AllChem.DataStructs.ConvertToNumpyArray(fp,arr); return arr.astype(np.float32)

def get_matrix(ps, mode):
    X=[]; good=[]
    for p in ps:
        d=feats[p]; parts=[]
        ok=True
        for k in mode.split('+'):
            if k=='ecfp3': parts.append(d['ecfp3'])
            elif k=='desc': parts.append(d['desc'])
            elif k=='B_gene': parts.append(d['B_gene'])
            elif k=='molformer': parts.append(d['molformer'])
            elif k=='morgan2':
                v=morgan2(d['smiles'])
                if v is None: ok=False; break
                parts.append(v)
            else: ok=False; break
        if not ok: continue
        good.append(p); parts=[np.asarray(x,dtype=np.float32).reshape(-1) for x in parts]; X.append(np.concatenate(parts).astype(np.float32))
    return good, np.array(X)

def agg_drug(profiles, probs):
    gs=defaultdict(list)
    for p,pr in zip(profiles,probs):
        nm=id2name.get(p,p)
        if p in truth: gs[nm].append((truth[p],pr))
        else: gs[nm].append((labels[p],pr))
    y=[]; P=[]
    for nm,arr in sorted(gs.items()):
        lab=Counter([a for a,b in arr]).most_common(1)[0][0]; y.append(lab); P.append(np.mean([b for a,b in arr],axis=0))
    return np.array(y), np.array(P)

def select_bias(Ydrug, Pdrug):
    best=None
    for bI in np.arange(-0.5,0.81,0.02):
        adj=Pdrug.copy(); adj[:,1]+=bI; pred=adj.argmax(1)
        ri=float(np.mean(pred[Ydrug==1]==1)) if np.any(Ydrug==1) else 0
        rn=float(np.mean(pred[Ydrug==2]==2)) if np.any(Ydrug==2) else 0
        if ri<0.75 or rn<0.5: continue
        acc=float(accuracy_score(Ydrug,pred)); mf=float(f1_score(Ydrug,pred,average='macro')); key=(acc,mf,ri+rn)
        if best is None or key>best[0]: best=(key,float(bI),ri,rn,acc,mf)
    if best is None:
        for bI in np.arange(-0.5,0.81,0.02):
            adj=Pdrug.copy(); adj[:,1]+=bI; pred=adj.argmax(1)
            ri=float(np.mean(pred[Ydrug==1]==1)); rn=float(np.mean(pred[Ydrug==2]==2))
            if ri<0.55 or rn<0.35: continue
            acc=float(accuracy_score(Ydrug,pred)); mf=float(f1_score(Ydrug,pred,average='macro')); key=(acc,mf,ri+rn)
            if best is None or key>best[0]: best=(key,float(bI),ri,rn,acc,mf)
    return best

def evaluate(mode, clf_factory, C_grid=None):
    ps,X=get_matrix(train,mode)
    groups=np.array([id2name[p] for p in ps]); Y=np.array([labels[p] for p in ps])
    gkf=GroupKFold(5)
    best=None; bestC=None
    Cs=C_grid or [0.5]
    for C in Cs:
        oof=np.zeros((len(ps),3),dtype=np.float32)
        for tr,va in gkf.split(X,Y,groups):
            clf=clf_factory(C); clf.fit(X[tr],Y[tr]); oof[va]=clf.predict_proba(X[va])
        Yd,Pd=agg_drug(ps,oof); sb=select_bias(Yd,Pd)
        if sb is None: continue
        key=sb[0]
        if best is None or key>best[0]: best=sb; bestC=C
    if best is None:
        return {'mode':mode,'error':'no CV config'}
    key,bI,ri,rn,cvacc,cvmf=best; C=bestC
    clf=clf_factory(C).fit(X,Y)
    tps,Xte=get_matrix(test,mode); raw=clf.predict_proba(Xte)
    Yte,Pte=agg_drug(tps,raw); Pte[:,1]+=bI; pred=Pte.argmax(1)
    acc=accuracy_score(Yte,pred); mf=f1_score(Yte,pred,average='macro'); cm=confusion_matrix(Yte,pred,labels=[0,1,2]).tolist()
    pp,rr,ff,ss=precision_recall_fscore_support(Yte,pred,labels=[0,1,2],average=None)
    auc3=np.mean([roc_auc_score((Yte==c).astype(int),Pte[:,c]) for c in range(3)]); mask=Yte!=2; aucpi=roc_auc_score((Yte[mask]==0).astype(int),(Pte[:,0]-Pte[:,1])[mask])
    return {'mode':mode,'C':C,'bias':bI,'cv_acc':cvacc,'cv_mf1':cvmf,'cv_Irec':ri,'cv_Nrec':rn,'test_acc':float(acc),'test_macro_f1':float(mf),'test_auc3':float(auc3),'test_pi_auc':float(aucpi),'cm':cm,'precision':pp.tolist(),'recall':rr.tolist(),'f1':ff.tolist(),'support':ss.tolist(),'n_test_drugs':len(Yte),'n_train_profiles':len(ps)}

def lr_factory(C):
    return make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=5000,class_weight='balanced'))
def mlp_factory(C):
    return make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(128,),alpha=1e-3,max_iter=500,random_state=42))

results=[]
for mode in ['desc+ecfp3+B_gene','desc+ecfp3','B_gene','ecfp3+B_gene','desc+ecfp3+B_gene+molformer']:
    rep=evaluate(mode,lr_factory,C_grid=[0.3,0.5,1.0,2.0]); results.append(rep); print('LR',rep,flush=True)
# MLP baselines, use fixed C=0.5 and MLP
for mode in ['ecfp3','morgan2','desc','desc+ecfp3']:
    rep=evaluate(mode,mlp_factory,C_grid=[0.5]); results.append(rep); print('MLP',rep,flush=True)
(OUT/'ablation_benchmark_135.json').write_text(json.dumps(results,ensure_ascii=False,indent=1),encoding='utf-8')
# CSV
with open(OUT/'ablation_benchmark_135.csv','w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['mode','C','bias','cv_acc','cv_mf1','cv_Irec','cv_Nrec','test_acc','test_macro_f1','test_auc3','test_pi_auc','I_recall','n_train_profiles','n_test_drugs']); w.writeheader()
    for r in results:
        w.writerow({k:r.get(k) for k in w.fieldnames if k not in ['I_recall']}); 
print('saved',OUT/'ablation_benchmark_135.json',flush=True)
