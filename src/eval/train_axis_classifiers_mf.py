# -*- coding: utf-8 -*-
"""Train four independent bone-lineage classifiers for PathBone-MF.

Feature mode: desc + ECFP3 + B_gene + MolFormer.
Labels:
  - osteoblast: P=1, I=0 from the PNI literature labels.
  - osteoclast/adipocyte/chondrocyte: multi_axis_labels_pert_v5.json.

The 135-drug hold-out is fixed and is not used for training or model selection.
"""
import os, sys, json, csv, pickle
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(os.environ['VIRTUAL_BONE'])
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'outputs' / 'v20'
REL = ROOT / ' (see paper)' / 'model'

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score


AXES = ['osteoblast', 'osteoclast', 'adipocyte', 'chondrocyte']
FEATURE_MODE = 'desc+ecfp3+B_gene+molformer'


def feature_vec(p, feats, mode):
    d = feats.get(p)
    if d is None or not d.get('smiles'):
        return None
    parts = []
    for key in mode.split('+'):
        val = d.get(key)
        if val is None:
            return None
        parts.append(np.asarray(val, dtype=np.float32).reshape(-1))
    return np.concatenate(parts).astype(np.float32)


def axis_label_map(ax, pni_labels, multi, feats):
    if ax == 'osteoblast':
        return {p: (1 if pni_labels[p] == 0 else 0)
                for p in pni_labels
                if pni_labels[p] in (0, 1) and feature_vec(p, feats, FEATURE_MODE) is not None}
    key = {'osteoclast': 'osteoclast', 'adipocyte': 'adipo', 'chondrocyte': 'chondro'}[ax]
    return {p: (1 if v[key] == 1 else 0)
            for p, v in multi.items()
            if key in v and v[key] in (1, -1)
            and feature_vec(p, feats, FEATURE_MODE) is not None}


def drug_aggregate(label_map, ps, prob, id2name):
    groups = defaultdict(list)
    for p, pr in zip(ps, prob):
        groups[id2name.get(p, p)].append((p, pr))
    names, y, scores = [], [], []
    for name, arr in sorted(groups.items()):
        labels = [label_map[p] for p, _ in arr]
        cnt = Counter(labels)
        mx = max(cnt.values())
        tied = [c for c, n in cnt.items() if n == mx]
        if len(tied) == 1:
            lab = tied[0]
        else:
            lab = 1
        names.append(name)
        y.append(lab)
        scores.append(float(np.mean([pr for _, pr in arr])))
    return np.array(names), np.array(y), np.array(scores)


def main():
    feats = pickle.load(open(str(OUT / 'abcde_feats.pkl'), 'rb'))
    pni_labels = {k: int(v) for k, v in json.loads((ROOT / 'outputs' / 'labels_expanded.json').read_text(encoding='utf-8')).items()}
    multi = json.loads((ROOT / 'outputs' / 'multi_axis_labels_pert_v5.json').read_text(encoding='utf-8'))
    tj = json.loads((ROOT / 'outputs' / 'test_set_drugs_135.json').read_text(encoding='utf-8'))
    test_ids = set(tj['promote_ids'] + tj['inhibit_ids'] + tj['neutral_ids'])
    with open(ROOT / 'compoundinfo_beta.txt', encoding='utf-8') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    id2name = {r['pert_id']: (r.get('cmap_name') or r.get('pert_id') or '').lower() for r in rows}

    models = {}
    metrics = {}
    for ax in AXES:
        label_map = axis_label_map(ax, pni_labels, multi, feats)
        train = sorted([p for p in label_map if p not in test_ids])
        testp = sorted([p for p in label_map if p in test_ids])
        if not train or not testp:
            metrics[ax] = {'status': 'insufficient_data'}
            continue
        Xtr = np.array([feature_vec(p, feats, FEATURE_MODE) for p in train], dtype=np.float32)
        ytr = np.array([label_map[p] for p in train], dtype=np.float32)
        Xte = np.array([feature_vec(p, feats, FEATURE_MODE) for p in testp], dtype=np.float32)
        yte = np.array([label_map[p] for p in testp], dtype=np.float32)

        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000, class_weight='balanced'))
        clf.fit(Xtr, ytr)
        prob_te = clf.predict_proba(Xte)[:, 1]
        models[ax] = clf

        # Leave-one-drug-out CV on training only.
        groups = np.array([id2name.get(p, p) for p in train])
        cv_auc = None
        if len(np.unique(ytr)) >= 2 and len(train) >= 5:
            try:
                prob_cv = cross_val_predict(clf, Xtr, ytr, cv=LeaveOneGroupOut().split(Xtr, ytr, groups=groups), method='predict_proba')[:, 1]
                cv_auc = float(roc_auc_score(ytr, prob_cv))
            except Exception:
                cv_auc = None

        names, ydrug, scored = drug_aggregate(label_map, testp, prob_te, id2name)
        drug_auc = None
        drug_acc = None
        if len(np.unique(ydrug)) >= 2:
            drug_auc = float(roc_auc_score(ydrug, scored))
            pred_drug = (scored >= 0.5).astype(int)
            drug_acc = float(accuracy_score(ydrug, pred_drug))

        profile_auc = None
        profile_acc = None
        if len(np.unique(yte)) >= 2:
            profile_auc = float(roc_auc_score(yte, prob_te))
            profile_acc = float(accuracy_score(yte, (prob_te >= 0.5).astype(int)))

        metrics[ax] = {
            'train_profiles': int(len(train)),
            'train_pos': int(ytr.sum()),
            'train_neg': int((1 - ytr).sum()),
            'test_profiles': int(len(testp)),
            'test_pos': int(yte.sum()),
            'test_neg': int((1 - yte).sum()),
            'profile_auc': profile_auc,
            'profile_acc': profile_acc,
            'drug_level': {
                'unique_drugs': int(len(names)),
                'auc': drug_auc,
                'acc': drug_acc
            },
            'loocv_auc_training': cv_auc,
            'model': 'LogisticRegression(C=0.3, balanced)',
            'feature_mode': FEATURE_MODE
        }
        print(ax, json.dumps(metrics[ax], ensure_ascii=False), flush=True)

    REL.mkdir(parents=True, exist_ok=True)
    joblib.dump({'feature_mode': FEATURE_MODE, 'axes': AXES, 'models': models}, str(REL / 'pathbone_v2_mf_axes.joblib'))
    (OUT / 'axis_classifiers_mf_metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print('SAVED', REL / 'pathbone_v2_mf_axes.joblib')


if __name__ == '__main__':
    main()
