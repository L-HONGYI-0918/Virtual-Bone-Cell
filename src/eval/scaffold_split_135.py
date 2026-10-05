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
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix,precision_recall_fscore_support
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

feats=pickle.load(open(str(OUT/'abcde_feats.pkl'),'rb'))
labels={k:int(v) for k,v in json.loads((ROOT/'outputs'/'labels_expanded.json').read_text(encoding='utf-8')).items()}
tj=json.loads((ROOT/'outputs'/'test_set_drugs_135.json').read_text(encoding='utf-8'))
test_ids=set(tj['promote_ids']+tj['inhibit_ids']+tj['neutral_ids']); truth={}
for p in tj['promote_ids']: truth[p]=0
for p in tj['inhibit_ids']: truth[p]=1
for p in tj['neutral_ids']: truth[p]=2
rows=list(csv.DictReader(open(ROOT/'compoundinfo_beta.txt',encoding='utf-8'),delimiter='\t'))
id2name={r['pert_id']:(r.get('cmap_name') or r.get('pert_id') or '').lower() for r in rows}
train_all=[p for p in labels if p not in test_ids and p in feats and feats[p].get('smiles')]
test=[p for p in test_ids if p in feats and feats[p].get('smiles')]

def scaffold_of(smi):
    mol=Chem.MolFromSmiles(smi)
    if mol is None: return None
    try:
        sc=MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return sc or None
    except Exception:
        return None

train_scaf=[scaffold_of(feats[p]['smiles']) for p in train_all]
test_scaf=[scaffold_of(feats[p]['smiles']) for p in test]
test_scaf_set={s for s in test_scaf if s}
seen_test_scaf=[s for s in test_scaf if s in {s for s in train_scaf if s}]
unseen_test_scaf=[s for s in test_scaf if s and s not in {s for s in train_scaf if s}]
print('train',len(train_all),'test',len(test),'test unique scaffolds',len(test_scaf_set),'seen',len(seen_test_scaf),'unseen',len(unseen_test_scaf),flush=True)

mode='desc+ecfp3+B_gene'
def vec(p):
    d=feats[p]
    return np.concatenate([np.asarray(d[k],dtype=np.float32).reshape(-1) for k in mode.split('+')]).astype(np.float32)

def fit_predict(tr_profiles, te_profiles):
    X=np.array([vec(p) for p in tr_profiles]); Y=np.array([labels[p] for p in tr_profiles])
    groups=np.array([id2name[p] for p in tr_profiles]); gkf=GroupKFold(5)
    best=None
    for C in [0.3,0.5,1.0,2.0,5.0]:
        oof=np.zeros((len(tr_profiles),3),dtype=np.float32)
        for tr,va in gkf.split(X,Y,groups):
            clf=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=5000,class_weight='balanced')).fit(X[tr],Y[tr]); oof[va]=clf.predict_proba(X[va])
        for bI in np.arange(-0.4,0.81,0.02):
            adj=oof.copy(); adj[:,1]+=bI; pred=adj.argmax(1)
            ri=float(np.mean(pred[Y==1]==1)); rn=float(np.mean(pred[Y==2]==2))
            if ri<0.75 or rn<0.5: continue
            acc=float(accuracy_score(Y,pred)); mf=float(f1_score(Y,pred,average='macro')); key=(acc,mf,ri+rn)
            if best is None or key>best[0]: best=(key,C,float(bI),ri,rn,acc,mf)
    if best is None:
        # relaxed fallback
        for C in [0.5]:
            oof=np.zeros((len(tr_profiles),3),dtype=np.float32)
            for tr,va in gkf.split(X,Y,groups):
                clf=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=5000,class_weight='balanced')).fit(X[tr],Y[tr]); oof[va]=clf.predict_proba(X[va])
            for bI in np.arange(-0.4,0.81,0.02):
                adj=oof.copy(); adj[:,1]+=bI; pred=adj.argmax(1)
                ri=float(np.mean(pred[Y==1]==1)); rn=float(np.mean(pred[Y==2]==2))
                if ri<0.6 or rn<0.4: continue
                acc=float(accuracy_score(Y,pred)); mf=float(f1_score(Y,pred,average='macro')); key=(acc,mf,ri+rn)
                if best is None or key>best[0]: best=(key,C,float(bI),ri,rn,acc,mf)
    key,C,bI,ri,rn,cvacc,cvmf=best
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=5000,class_weight='balanced')).fit(X,Y)
    Xte=np.array([vec(p) for p in te_profiles]); raw=clf.predict_proba(Xte)
    gs=defaultdict(list)
    for p,prob in zip(te_profiles,raw): gs[id2name.get(p,p)].append((truth[p],prob))
    y=[]; P=[]
    for n,arr in sorted(gs.items()):
        lab=Counter([a for a,b in arr]).most_common(1)[0][0]; y.append(lab); P.append(np.mean([b for a,b in arr],axis=0))
    P=np.array(P); y=np.array(y); P[:,1]+=bI; pred=P.argmax(1)
    acc=accuracy_score(y,pred); mf=f1_score(y,pred,average='macro'); cm=confusion_matrix(y,pred,labels=[0,1,2]).tolist()
    pp,rr,ff,ss=precision_recall_fscore_support(y,pred,labels=[0,1,2],average=None)
    auc3=np.mean([roc_auc_score((y==c).astype(int),P[:,c]) for c in range(3)]); mask=y!=2; aucpi=roc_auc_score((y[mask]==0).astype(int),(P[:,0]-P[:,1])[mask])
    return {'C':C,'bias':bI,'cv_acc':cvacc,'cv_mf1':cvmf,'cv_Irec':ri,'cv_Nrec':rn,'test_acc':float(acc),'test_macro_f1':float(mf),'test_auc3':float(auc3),'test_pi_auc':float(aucpi),'cm':cm,'precision':pp.tolist(),'recall':rr.tolist(),'f1':ff.tolist(),'support':ss.tolist(),'n_test_drugs':len(y),'n_train_profiles':len(tr_profiles)}

# A: full train -> full test
full_rep=fit_predict(train_all,test)
# B: scaffold-disjoint train -> full test
train_scaf_arr=np.array(train_scaf, dtype=object); test_scaf_set={s for s in test_scaf if s}
disjoint_train=[p for p,s in zip(train_all,train_scaf) if s is None or s not in test_scaf_set]
disjoint_full_rep=fit_predict(disjoint_train,test)
# C: full train -> unseen-scaffold test drugs (profiles)
unseen_test=[p for p,s in zip(test,test_scaf) if s is not None and s not in {s for s in train_scaf if s}]
unseen_rep=fit_predict(train_all,unseen_test) if unseen_test else None
report={'n_train_profiles':len(train_all),'n_disjoint_train_profiles':len(disjoint_train),'n_test_profiles':len(test),'n_unseen_scaffold_test_profiles':len(unseen_test),'full_train_full_test':full_rep,'scaffold_disjoint_train_full_test':disjoint_full_rep,'full_train_unseen_scaffold_test':unseen_rep}
(OUT/'scaffold_split_135.json').write_text(json.dumps(report,ensure_ascii=False,indent=1),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=1),flush=True)
# CSV summary
rows=[]
for name,rep in [('full_train_full_test',full_rep),('scaffold_disjoint_train_full_test',disjoint_full_rep),('full_train_unseen_scaffold_test',unseen_rep or {})]:
    rows.append({'setting':name,'train_profiles':rep.get('n_train_profiles'),'test_drugs':rep.get('n_test_drugs'),'test_acc':rep.get('test_acc'),'test_auc3':rep.get('test_auc3'),'test_pi_auc':rep.get('test_pi_auc'),'I_recall':(rep.get('recall') or [None]*3)[1]})
with open(OUT/'scaffold_split_135.csv','w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['setting','train_profiles','test_drugs','test_acc','test_auc3','test_pi_auc','I_recall']); w.writeheader(); w.writerows(rows)
print('saved scaffold_split_135.json/csv',flush=True)
