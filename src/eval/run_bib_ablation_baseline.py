# -*- coding: utf-8 -*-
"""BIB-style ablation and baseline benchmark for PathBone-MF.

Fixed 135-drug/131-unique-drug hold-out; all models use the same train/test
profiles. CV is repeated 5-fold drug-group split (3 repeats). Metrics:
Acc, Macro-F1, 3-class macro AUC, P/I AUC, per-class recall. Paired
bootstrap and exact McNemar-style tests compare each model to the full model.
"""
import os, sys, json, csv, pickle
from pathlib import Path
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(os.environ['VIRTUAL_BONE'])
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'outputs' / 'v20'
FIG = ROOT / ' (see paper)'

import numpy as np
import joblib
from scipy import stats
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_fscore_support

# ----------------------------- data loading -----------------------------

def load_data():
    feats = pickle.load(open(str(OUT / 'abcde_feats.pkl'), 'rb'))
    labels = {k: int(v) for k, v in json.loads((ROOT / 'outputs' / 'labels_expanded.json').read_text(encoding='utf-8')).items()}
    tj = json.loads((ROOT / 'outputs' / 'test_set_drugs_135.json').read_text(encoding='utf-8'))
    test_ids = set(tj['promote_ids'] + tj['inhibit_ids'] + tj['neutral_ids'])
    truth = {}
    for p in tj['promote_ids']:
        truth[p] = 0
    for p in tj['inhibit_ids']:
        truth[p] = 1
    for p in tj['neutral_ids']:
        truth[p] = 2
    with open(ROOT / 'compoundinfo_beta.txt', encoding='utf-8') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    id2name = {r['pert_id']: (r.get('cmap_name') or r.get('pert_id') or '').lower() for r in rows}

    gnn_d = np.load(str(OUT / 'strict_gnn_feats.npz'))
    gnn_feats = {str(n): v.astype(np.float32) for n, v in zip(gnn_d['names'], gnn_d['feats'])}

    return feats, labels, test_ids, truth, id2name, gnn_feats

FEATS, LABELS, TEST_IDS, TRUTH, ID2NAME, GNN_FEATS = load_data()

# fingerprint caches

def _mol(smiles):
    return Chem.MolFromSmiles(smiles)

def ecfp4(smiles):
    mol = _mol(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)

def morgan2(smiles):
    mol = _mol(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useFeatures=True)
    arr = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)

_ECFP4 = {}
_MORGAN2 = {}

def feature_vector(p, mode):
    d = FEATS.get(p)
    if d is None or not d.get('smiles'):
        return None
    parts = []
    for key in mode.split('+'):
        if key == 'desc':
            val = d.get('desc')
        elif key == 'ecfp3':
            val = d.get('ecfp3')
        elif key == 'ecfp4':
            if p not in _ECFP4:
                _ECFP4[p] = ecfp4(d['smiles'])
            val = _ECFP4[p]
        elif key == 'morgan2':
            if p not in _MORGAN2:
                _MORGAN2[p] = morgan2(d['smiles'])
            val = _MORGAN2[p]
        elif key == 'molformer':
            val = d.get('molformer')
        elif key == 'B_gene':
            val = d.get('B_gene')
        elif key == 'old_gene':
            val = d.get('old_gene')
        elif key == 'gnn':
            val = GNN_FEATS.get(p)
        else:
            val = d.get(key)
        if val is None:
            return None
        parts.append(np.asarray(val, dtype=np.float32).reshape(-1))
    return np.concatenate(parts).astype(np.float32)

def build_matrix(ps, mode):
    ok = []
    X = []
    for p in ps:
        v = feature_vector(p, mode)
        if v is not None:
            ok.append(p)
            X.append(v)
    return ok, np.array(X, dtype=np.float32)

def drug_aggregate(profiles, labels_or_truth, proba):
    gs = defaultdict(list)
    for p, pr in zip(profiles, proba):
        name = ID2NAME.get(p, p)
        gs[name].append((labels_or_truth[p], pr))
    names = []
    y = []
    P = []
    for name, arr in sorted(gs.items()):
        labels = [a for a, _ in arr]
        cnt = Counter(labels)
        mx = max(cnt.values())
        tied = [c for c, n in cnt.items() if n == mx]
        if len(tied) == 1:
            lab = tied[0]
        else:
            mean_prob = {c: np.mean([b for a, b in arr if a == c], axis=0) for c in tied}
            lab = max(tied, key=lambda c: mean_prob[c][c])
        names.append(name)
        y.append(lab)
        P.append(np.mean([b for _, b in arr], axis=0))
    return np.array(names), np.array(y), np.array(P, dtype=np.float32)

def metrics(y, pred, proba):
    acc = float(accuracy_score(y, pred))
    mf1 = float(f1_score(y, pred, average='macro'))
    auc3 = float(np.mean([roc_auc_score((y == c).astype(int), proba[:, c]) for c in range(3)]))
    mask = y != 2
    auc_pi = float(roc_auc_score((y[mask] == 0).astype(int), (proba[:, 0] - proba[:, 1])[mask])) if mask.sum() else float('nan')
    pp, rr, ff, ss = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        'acc': acc,
        'macro_f1': mf1,
        'auc3': auc3,
        'auc_pi': auc_pi,
        'precision': pp.tolist(),
        'recall': rr.tolist(),
        'f1': ff.tolist(),
        'support': ss.tolist()
    }

def majority_label(profiles, labels):
    return Counter([labels[p] for p in profiles]).most_common(1)[0][0]

def run_repeated_cv(train_ps, mode, clf_factory, n_repeats=3, n_splits=5, random_seeds=(42, 43, 44)):
    ok, X = build_matrix(train_ps, mode)
    y = np.array([LABELS[p] for p in ok])
    groups = np.array([ID2NAME.get(p, p) for p in ok])
    unique_drugs = sorted(set(groups.tolist()))
    drug_lab = {d: majority_label([p for p in ok if ID2NAME.get(p, p) == d], LABELS) for d in unique_drugs}
    y_drug = np.array([drug_lab[d] for d in unique_drugs])
    fold_acc = []
    fold_mf1 = []
    fold_auc3 = []
    for seed in random_seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr_drug_idx, va_drug_idx in skf.split(unique_drugs, y_drug):
            tr_drugs = set(unique_drugs[i] for i in tr_drug_idx)
            va_drugs = set(unique_drugs[i] for i in va_drug_idx)
            tr_idx = [i for i, g in enumerate(groups) if g in tr_drugs]
            va_idx = [i for i, g in enumerate(groups) if g in va_drugs]
            clf = clf_factory()
            clf.fit(X[tr_idx], y[tr_idx])
            proba = clf.predict_proba(X[va_idx])
            va_ps = [ok[i] for i in va_idx]
            _, yv, Pv = drug_aggregate(va_ps, LABELS, proba)
            pred = Pv.argmax(1)
            m = metrics(yv, pred, Pv)
            fold_acc.append(m['acc'])
            fold_mf1.append(m['macro_f1'])
            fold_auc3.append(m['auc3'])
    return {
        'cv_acc_mean': float(np.mean(fold_acc)),
        'cv_acc_std': float(np.std(fold_acc)),
        'cv_macro_f1_mean': float(np.mean(fold_mf1)),
        'cv_macro_f1_std': float(np.std(fold_mf1)),
        'cv_auc3_mean': float(np.mean(fold_auc3)),
        'cv_auc3_std': float(np.std(fold_auc3)),
        'cv_folds': len(fold_acc),
        'fold_acc': [float(x) for x in fold_acc],
        'fold_mf1': [float(x) for x in fold_mf1],
        'fold_auc3': [float(x) for x in fold_auc3],
    }

def evaluate_holdout(train_ps, test_ps, mode, clf_factory):
    ok_tr, Xtr = build_matrix(train_ps, mode)
    ytr = np.array([LABELS[p] for p in ok_tr])
    ok_te, Xte = build_matrix(test_ps, mode)
    clf = clf_factory()
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)
    yte_truth = {p: TRUTH[p] for p in ok_te}
    names, yte, Pte = drug_aggregate(ok_te, yte_truth, proba)
    pred = Pte.argmax(1)
    return metrics(yte, pred, Pte), names, yte, pred, Pte

def paired_accuracy_test(pred_full, y_full, pred_other, y_other, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_full)
    correct_full = (pred_full == y_full).astype(int)
    correct_other = (pred_other == y_other).astype(int)
    diff = correct_full - correct_other
    boot_diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_diffs.append(float(np.mean(diff[idx])))
    boot_diffs = np.array(boot_diffs)
    p_two = float(2 * min(np.mean(boot_diffs >= 0), np.mean(boot_diffs <= 0)))
    a = int(np.sum((correct_full == 1) & (correct_other == 0)))
    b = int(np.sum((correct_full == 0) & (correct_other == 1)))
    if a + b == 0:
        mcnemar_p = 1.0
    else:
        mcnemar_p = float(stats.binomtest(min(a, b), a + b, 0.5, alternative='two-sided').pvalue)
    return {
        'paired_acc_boot_mean_diff': float(np.mean(boot_diffs)),
        'paired_acc_boot_p': p_two,
        'mcnemar_discordant_full_other': [a, b],
        'mcnemar_p': mcnemar_p
    }

# ----------------------------- model factories -----------------------------

def lr_factory():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000, class_weight='balanced'))

def svm_factory():
    return make_pipeline(StandardScaler(), SVC(C=1.0, kernel='linear', probability=True, class_weight='balanced', max_iter=5000, random_state=42))

def mlp_factory(hidden=(128,)):
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=hidden, alpha=1e-3, max_iter=500, early_stopping=True, random_state=42))

def gbdt_factory():
    return GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)

def gnn_mlp_factory():
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=500, early_stopping=True, random_state=42))

# ----------------------------- run specs -----------------------------

def main():
    needed = ['smiles', 'desc', 'ecfp3', 'B_gene', 'molformer', 'old_gene']
    common = [p for p in LABELS if all(k in FEATS[p] and FEATS[p].get(k) is not None for k in needed) and p in GNN_FEATS]
    train_ps = sorted([p for p in common if p not in TEST_IDS])
    test_ps = sorted([p for p in TEST_IDS if p in common])
    print('common', len(common), 'train profiles', len(train_ps), 'test profiles', len(test_ps), flush=True)

    full_mode = 'desc+ecfp3+B_gene+molformer'

    specs = [
        ('PathBone-MF_full', full_mode, 'lr', 'main'),
        ('ablation_no_molformer', 'desc+ecfp3+B_gene', 'lr', 'ablation'),
        ('ablation_no_B_gene', 'desc+ecfp3+molformer', 'lr', 'ablation'),
        ('ablation_no_ecfp3', 'desc+B_gene+molformer', 'lr', 'ablation'),
        ('ablation_no_desc', 'ecfp3+B_gene+molformer', 'lr', 'ablation'),
        ('ablation_no_geo_anchor', 'desc+ecfp3+old_gene+molformer', 'lr', 'ablation'),
        ('baseline_desc_LogReg', 'desc', 'lr', 'baseline'),
        ('baseline_ECFP4_LogReg', 'ecfp4', 'lr', 'baseline'),
        ('baseline_MorganFP_LogReg', 'morgan2', 'lr', 'baseline'),
        ('baseline_MolFormer_LogReg', 'molformer', 'lr', 'baseline'),
        ('baseline_ECFP4_GBDT', 'ecfp4', 'gbdt', 'baseline'),
        ('baseline_ECFP4_SVM', 'ecfp4', 'svm', 'baseline'),
        ('baseline_ECFP4_MLP', 'ecfp4', 'mlp', 'baseline'),
        ('baseline_GNN_MLP', 'gnn', 'gnn_mlp', 'baseline'),
    ]

    factory_map = {
        'lr': lr_factory,
        'svm': svm_factory,
        'mlp': lambda: mlp_factory((128,)),
        'gbdt': gbdt_factory,
        'gnn_mlp': gnn_mlp_factory,
    }

    results = []
    full_pred = None
    full_y = None

    for name, mode, factory_name, group in specs:
        print('RUN', name, mode, flush=True)
        cv = run_repeated_cv(train_ps, mode, factory_map[factory_name], n_repeats=3, n_splits=5)
        holdout, names, yte, pred, Pte = evaluate_holdout(train_ps, test_ps, mode, factory_map[factory_name])
        item = {
            'name': name,
            'group': group,
            'mode': mode,
            'factory': factory_name,
            'cv': cv,
            'test_metrics': holdout,
        }
        if group == 'main':
            full_names = names
            full_pred = pred
            full_y = yte
        results.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)

    for item in results:
        if item['name'] == 'PathBone-MF_full':
            item['paired_vs_full'] = {'role': 'reference'}
        else:
            _, names_other, yte_other, pred_other, _ = evaluate_holdout(train_ps, test_ps, item['mode'], factory_map[item['factory']])
            idx_full = {n:i for i,n in enumerate(full_names)}
            idx_other = {n:i for i,n in enumerate(names_other)}
            common_names = [n for n in full_names if n in idx_other]
            if len(common_names) >= 0.9 * len(full_names):
                jf = [idx_full[n] for n in common_names]
                jo = [idx_other[n] for n in common_names]
                item['paired_vs_full'] = paired_accuracy_test(full_pred[jf], full_y[jf], pred_other[jo], yte_other[jo])
            else:
                item['paired_vs_full'] = {'error': 'y mismatch', 'common_drugs': len(common_names)}

    # Paired Wilcoxon signed-rank test: full model vs each comparison model,
    # on the paired repeated-CV folds (same split seeds -> paired fold metrics).
    full_item = next(it for it in results if it['name'] == 'PathBone-MF_full')
    fold_metric_map = {'acc': 'fold_acc', 'macro_f1': 'fold_mf1', 'auc3': 'fold_auc3'}
    wilcox_rows = []
    for item in results:
        if item['name'] == 'PathBone-MF_full':
            continue
        rec = {'name': item['name'], 'group': item['group']}
        for metric, key in fold_metric_map.items():
            a = np.array(full_item['cv'][key], dtype=float)
            b = np.array(item['cv'][key], dtype=float)
            n = min(len(a), len(b))
            a, b = a[:n], b[:n]
            d = a - b
            zero_mask = np.abs(d) < 1e-12
            d = d[~zero_mask]
            if len(d) == 0:
                stat, p = float('nan'), 1.0
            else:
                stat, p = stats.wilcoxon(d, alternative='two-sided')
            rec[f'{metric}_wilcoxon_stat'] = float(stat) if not np.isnan(stat) else None
            rec[f'{metric}_wilcoxon_p'] = float(p)
            rec[f'{metric}_neglog10p'] = float(-np.log10(p)) if p > 0 else float('inf')
            rec[f'{metric}_fold_n_pairs'] = int(len(d))
        item['wilcoxon_vs_full'] = rec
        wilcox_rows.append(rec)
        print('WILCOX', rec, flush=True)

    out_json = OUT / 'bib_ablation_baseline_results.json'
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

    rows = []
    for item in results:
        m = item['test_metrics']
        cv = item['cv']
        pv = item.get('paired_vs_full', {})
        rows.append({
            'name': item['name'],
            'group': item['group'],
            'mode': item['mode'],
            'factory': item['factory'],
            'cv_acc_mean': round(cv['cv_acc_mean'], 4),
            'cv_acc_std': round(cv['cv_acc_std'], 4),
            'cv_macro_f1_mean': round(cv['cv_macro_f1_mean'], 4),
            'cv_macro_f1_std': round(cv['cv_macro_f1_std'], 4),
            'cv_auc3_mean': round(cv['cv_auc3_mean'], 4),
            'cv_auc3_std': round(cv['cv_auc3_std'], 4),
            'test_acc': round(m['acc'], 4),
            'test_macro_f1': round(m['macro_f1'], 4),
            'test_auc3': round(m['auc3'], 4),
            'test_pi_auc': round(m['auc_pi'], 4),
            'test_recall_P': round(m['recall'][0], 4),
            'test_recall_I': round(m['recall'][1], 4),
            'test_recall_N': round(m['recall'][2], 4),
            'paired_acc_boot_mean_diff': pv.get('paired_acc_boot_mean_diff', ''),
            'paired_acc_boot_p': pv.get('paired_acc_boot_p', ''),
            'mcnemar_p': pv.get('mcnemar_p', ''),
        })
    out_csv = OUT / 'bib_ablation_baseline_results.csv'
    with open(out_csv, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    fig_rows = []
    for item in results:
        m = item['test_metrics']
        for metric, value in [('Acc', m['acc']), ('Macro_F1', m['macro_f1']), ('AUC3', m['auc3']), ('P_I_AUC', m['auc_pi'])]:
            fig_rows.append({'name': item['name'], 'group': item['group'], 'metric': metric, 'value': round(value, 4), 'cv_mean': '', 'cv_std': ''})
        cv = item['cv']
        for metric, key in [('Acc', 'cv_acc_mean'), ('Macro_F1', 'cv_macro_f1_mean'), ('AUC3', 'cv_auc3_mean')]:
            fig_rows.append({'name': item['name'], 'group': item['group'], 'metric': metric + '_CV', 'value': round(cv[key], 4), 'cv_mean': round(cv[key], 4), 'cv_std': round(cv[key.replace('_mean', '_std')], 4) if key.endswith('_mean') else ''})
    fig_csv = FIG / ' (see paper)_BIB_ (see paper)baseline_plotdata.csv'
    with open(fig_csv, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['name', 'group', 'metric', 'value', 'cv_mean', 'cv_std'])
        w.writeheader()
        w.writerows(fig_rows)

    # Wilcoxon heatmap data: rows = comparison models, cols = metrics, value = -log10(p)
    heat_rows = []
    for item in results:
        if item['name'] == 'PathBone-MF_full':
            continue
        wv = item['wilcoxon_vs_full']
        for metric in ['acc', 'macro_f1', 'auc3']:
            heat_rows.append({
                'name': item['name'],
                'group': item['group'],
                'metric': metric,
                'wilcoxon_p': wv[f'{metric}_wilcoxon_p'],
                'neglog10p': wv[f'{metric}_neglog10p'],
                'fold_n_pairs': wv[f'{metric}_fold_n_pairs'],
            })
    heat_csv = FIG / ' (see paper)_BIB_Wilcoxon (see paper)data.csv'
    with open(heat_csv, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['name', 'group', 'metric', 'wilcoxon_p', 'neglog10p', 'fold_n_pairs'])
        w.writeheader()
        w.writerows(heat_rows)

    print('saved', out_json)
    print('saved', out_csv)
    print('saved', fig_csv)
    print('saved', heat_csv)

if __name__ == '__main__':
    main()
