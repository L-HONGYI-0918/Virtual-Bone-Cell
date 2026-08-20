# -*- coding: utf-8 -*-
"""ABC step A: GO/KEGG pathway-bottleneck supervised model.

Input: precomputed 256d GNN drug embedding.
Bottleneck: hidden -> 3253 pathway scores -> fixed pw_weight -> 978 genes.
Multi-task: gene MSE + PNI cross-entropy + signed multi-axis BCE.
Strict: 135 test drugs and scaffold-overlap training profiles are excluded.
"""
import sys, json, time, argparse
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'outputs' / 'v20'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

AXES = ['osteo', 'osteoclast', 'adipo', 'chondro', 'osteocyte']

class PathBoneBottleneck(nn.Module):
    def __init__(self, in_dim=256, hidden=512, npw=3253, ngene=978, n_class=3, axes=AXES, pw_weight=None):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.pw_head = nn.Linear(hidden, npw)
        self.scale = nn.Parameter(torch.tensor(0.05))
        self.pni_head = nn.Linear(hidden, n_class)
        self.axis_heads = nn.ModuleDict({ax: nn.Linear(hidden, 1) for ax in axes})
        if pw_weight is not None:
            self.register_buffer('pw_weight', torch.tensor(pw_weight, dtype=torch.float32))
        else:
            self.register_buffer('pw_weight', torch.zeros(npw, ngene))
    def forward(self, emb):
        h = self.enc(emb)
        p = self.pw_head(h)
        g = self.scale * (p @ self.pw_weight)
        logits = self.pni_head(h)
        ax = {k: self.axis_heads[k](h).squeeze(-1) for k in self.axis_heads}
        return h, p, g, logits, ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--out', default='pathbottleneck_A.pt')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    device = 'cuda'
    meta = json.loads((OUT/'lincs_meta.json').read_text(encoding='utf-8'))
    smiles_map = json.loads((OUT/'smiles_map.json').read_text(encoding='utf-8'))
    tj = json.loads((ROOT/'outputs'/'test_set_drugs_135.json').read_text(encoding='utf-8'))
    test_ids = set(tj['promote_ids'] + tj['inhibit_ids'] + tj['neutral_ids'])
    pni_all = {k:int(v) for k,v in json.loads((ROOT/'outputs'/'labels_expanded.json').read_text(encoding='utf-8')).items()}
    multi_all = json.loads((ROOT/'outputs'/'multi_axis_labels_pert_v5.json').read_text(encoding='utf-8'))
    # GNN features for unique drugs
    gd = np.load(str(OUT/'unified_gnn_feats.npz'))
    gnn = {str(n): v.astype(np.float32) for n,v in zip(gd['names'], gd['feats'])}
    # Aggregated effects from non-leak indices
    keep_idx = np.load(str(OUT/'unified_indices.npy'))
    mm = np.memmap(meta['memmap'], dtype='float16', mode='r', shape=(meta['n'], len(meta['genes'])))
    sums = {}; cnts = {}
    for i in keep_idx:
        p = meta['perts'][int(i)]
        if p in test_ids:
            continue
        v = mm[int(i)].astype(np.float32)
        if p not in sums:
            sums[p] = v.copy(); cnts[p] = 1
        else:
            sums[p] += v; cnts[p] += 1
    keep = sorted(p for p in sums if cnts[p] >= 3 and p in gnn and smiles_map.get(p))
    eff = np.stack([sums[p]/cnts[p] for p in keep]).astype(np.float32)
    emb_all = np.stack([gnn[p] for p in keep]).astype(np.float32)
    print('A drugs', len(keep), 'effect shape', eff.shape, flush=True)

    ck19 = torch.load(str(ROOT/'outputs'/'bone_vc_v19.pt'), map_location='cpu', weights_only=False)
    pw_weight = np.asarray(ck19['pw_weight'], dtype=np.float32)
    Aanch = np.load(str(OUT/'geo_anchors_orth.npz'))
    axis_vectors = {ax: Aanch[ax+'_up'].astype(np.float32) for ax in AXES}

    model = PathBoneBottleneck(in_dim=256, hidden=512, npw=pw_weight.shape[0], ngene=pw_weight.shape[1], pw_weight=pw_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    pni_train = {p:v for p,v in pni_all.items() if p not in test_ids and p in gnn}
    multi_train = {p:v for p,v in multi_all.items() if p not in test_ids and p in gnn}
    pos_by_axis = defaultdict(set); neg_by_axis = defaultdict(set)
    for p, axs in multi_train.items():
        for ax, val in axs.items():
            if ax not in AXES: continue
            if val == 1: pos_by_axis[ax].add(p)
            elif val == -1: neg_by_axis[ax].add(p)
    # Include osteo axis from PNI: promote -> osteo positive, inhibit -> osteo negative.
    for p,v in pni_train.items():
        if v == 0: pos_by_axis['osteo'].add(p)
        elif v == 1: neg_by_axis['osteo'].add(p)

    idx = np.arange(len(keep))
    rng = np.random.default_rng(42)
    epochs = 2 if args.smoke else args.epochs
    for ep in range(epochs):
        rng.shuffle(idx); total = 0.0; nb = 0; gene_total = 0.0; cls_total = 0.0; axis_total = 0.0
        bs = 128
        for s in range(0, len(idx), bs):
            b = idx[s:s+bs]
            embs = torch.tensor(emb_all[b], dtype=torch.float32, device=device)
            y = torch.tensor(eff[b], dtype=torch.float32, device=device)
            h, p, g, logits, ax_logits = model(embs)
            loss_gene = F.mse_loss(g, y)
            loss = loss_gene
            # PNI
            ps = [keep[i] for i in b]
            lab = [pni_train.get(x) for x in ps]
            valid = [(j,x) for j,x in enumerate(lab) if x is not None]
            if valid:
                jj = torch.tensor([j for j,x in valid], dtype=torch.long, device=device)
                yy = torch.tensor([lab[j] for j,x in valid], dtype=torch.long, device=device)
                loss_cls = F.cross_entropy(logits[jj], yy)
                loss = loss + 0.5 * loss_cls
            else:
                loss_cls = torch.zeros((), device=device)
            # axes BCE
            loss_ax = torch.zeros((), device=device)
            used_ax = 0
            for ax in AXES:
                ax_pos = [j for j,x in enumerate(ps) if x in pos_by_axis[ax]]
                ax_neg = [j for j,x in enumerate(ps) if x in neg_by_axis[ax]]
                if not ax_pos and not ax_neg:
                    continue
                jj = torch.tensor(ax_pos + ax_neg, dtype=torch.long, device=device)
                yy = torch.tensor([1]*len(ax_pos) + [0]*len(ax_neg), dtype=torch.float32, device=device)
                loss_ax = loss_ax + F.binary_cross_entropy_with_logits(ax_logits[ax][jj], yy)
                used_ax += 1
            if used_ax:
                loss = loss + 0.2 * (loss_ax / used_ax)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); gene_total += loss_gene.item(); cls_total += loss_cls.item(); axis_total += loss_ax.item(); nb += 1
        print(f'epoch {ep+1}/{epochs} loss={total/max(1,nb):.5f} gene={gene_total/max(1,nb):.5f} cls={cls_total/max(1,nb):.5f} axis={axis_total/max(1,nb):.5f}', flush=True)
        if (ep+1)%5 == 0 or ep == epochs-1:
            torch.save({'model': model.state_dict(), 'in_dim':256, 'hidden':512, 'npw':pw_weight.shape[0], 'ngene':pw_weight.shape[1], 'scale':float(model.scale.item()), 'genes':meta['genes']}, str(OUT/args.out))
            print('saved', args.out, flush=True)
    print('A_DONE', args.out, flush=True)

if __name__ == '__main__':
    main()