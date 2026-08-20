# -*- coding: utf-8 -*-
"""PathBone-MF release predictor.

Input: drug name or SMILES.
Outputs:
  - P/I/N class from PathBone-MF: desc + ECFP3 + B_gene + MolFormer classifier
  - 978 landmark genes and 12,328 expanded genes
  - 3,253 GO/KEGG pathway scores from the GEO-anchored E bottleneck model
  - signed bone-axis scores
"""
import sys, json, csv, argparse, time, math
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
ROOT = Path(__file__).resolve().parent
PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))
OUT = ROOT / 'models'
import numpy as np
import torch
import joblib
from scipy import stats
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger
from torch_geometric.data import Batch
RDLogger.DisableLog('rdApp.*')
from src import diffusion_vc
from src.train_abcde_A import PathBoneBottleneck

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
# Four lineage axes (osteoblast/osteoclast/adipocyte/chondrocyte) + net bone direction (osteoblast - osteoclast)
AXIS_KEYS = ['osteoblast', 'osteoclast', 'adipocyte', 'chondrocyte']
AXIS_CN = {'osteoblast': 'Osteoblast', 'osteoclast': 'Osteoclast', 'adipocyte': 'Adipocyte', 'chondrocyte': 'Chondrocyte'}


class PathBoneV2:
    def __init__(self):
        self.smiles_map = json.loads((OUT/'pathbone_v2_smiles_map.json').read_text(encoding='utf-8'))
        self.meta = json.loads((OUT/'pathbone_v2_lincs_meta.json').read_text(encoding='utf-8'))
        self.genes978 = list(self.meta['genes'])
        # Generative components: unified DDPM v2
        self.vae = diffusion_vc.VAE(n_lm=978).to(DEV)
        self.vae.load_state_dict(torch.load(str(OUT/'pathbone_v2_vae.pt'), map_location=DEV, weights_only=False)['model']); self.vae.eval()
        self.mol_enc = diffusion_vc.GNNEncoder().to(DEV)
        self.mol_enc.load_state_dict(torch.load(str(OUT/'pathbone_v2_mol_enc.pt'), map_location=DEV, weights_only=False)['model']); self.mol_enc.eval()
        ck = torch.load(str(OUT/'pathbone_v2_ddpm.pt'), map_location=DEV, weights_only=False)
        self.ddpm = diffusion_vc.CondDDPM(latent=64, cond_dim=ck['cond_dim'], hidden=ck.get('hidden',512), T=ck.get('T',400)).to(DEV)
        self.ddpm.load_state_dict(ck['model']); self.ddpm.eval()
        self.cell_vocab = ck.get('cell_vocab') or []
        self.time_vocab = ck.get('time_vocab') or []
        # Pathway bottleneck E (GEO-anchored, GSE28074 held out)
        ck19 = torch.load(str(OUT/'pathbone_v2_pw_names.pt'), map_location='cpu', weights_only=False)
        self.pw_weight = np.asarray(ck19['pw_weight'], dtype=np.float32)
        self.pathway_names = [str(x) for x in ck19['pathways']]
        ckA = torch.load(str(OUT/'pathbone_v2_pathway.pt'), map_location=DEV, weights_only=False)
        self.A = PathBoneBottleneck(in_dim=256, hidden=512, npw=self.pw_weight.shape[0], ngene=self.pw_weight.shape[1], pw_weight=self.pw_weight).to(DEV)
        self.A.load_state_dict(ckA['model']); self.A.eval()
        # 978 -> 12328 expansion
        D = np.load(str(OUT/'pathbone_v2_gene_expansion.npz'))
        self.expand_W = D['W']; self.expand_intercept = D['intercept']; self.genes12328 = [str(x) for x in D['all_symbols']]
        # Release PNI classifier: PathBone-MF (falls back to the older classifier only if MF is absent)
        mf_clf = OUT/'pathbone_v2_mf_pni.joblib'
        if mf_clf.exists():
            self.pni_clf = joblib.load(str(mf_clf))
            self.pni_feature_mode = 'desc+ecfp3+B_gene+molformer'
            self.pni_bias = 0.56
            dec = OUT/'pathbone_v2_mf_pni_decision.json'
            if dec.exists():
                self.pni_bias = float(json.loads(dec.read_text(encoding='utf-8')).get('bias', self.pni_bias))
        else:
            self.pni_clf = joblib.load(str(OUT/'pathbone_v2_pni.joblib'))
            self.pni_feature_mode = 'desc+ecfp3+B_gene'
            self.pni_bias = 0.0
            dec = OUT/'pathbone_v2_pni_decision.json'
            if dec.exists():
                self.pni_bias = float(json.loads(dec.read_text(encoding='utf-8')).get('bias', 0.0))
        # MolFormer is loaded lazily and kept on CPU to stay within 8GB GPU memory.
        self._mf_tokenizer = None
        self._mf_model = None
        self._mf_cache = {}
        self.mf_dir = OUT/'MoLFormer-XL-both-10pct'
        if not self.mf_dir.exists():
            alt = ROOT/'data'/'models'/'MoLFormer-XL-both-10pct'
            if alt.exists():
                self.mf_dir = alt
        # Four independent bone-lineage classifiers (fixed 135-drug holdout trained).
        self.axis_models = {}
        self.axis_feature_mode = FEATURE_MODE = 'desc+ecfp3+B_gene+molformer'
        ax_file = OUT/'pathbone_v2_mf_axes.joblib'
        if ax_file.exists():
            ax_pkg = joblib.load(str(ax_file))
            self.axis_feature_mode = ax_pkg.get('feature_mode', FEATURE_MODE)
            self.axis_models = ax_pkg.get('models', {})
    def resolve(self, drug):
        key = drug.strip()
        if key.lower() in self.smiles_map:
            return self.smiles_map[key.lower()], key
        if key in self.smiles_map:
            return self.smiles_map[key], key
        mol = Chem.MolFromSmiles(key)
        if mol is not None:
            return key, key
        return None, key
    @torch.no_grad()
    def gnn_emb(self, smiles):
        g = diffusion_vc.mol_to_graph(smiles)
        if g is None: return None
        return self.mol_enc(Batch.from_data_list([g]).to(DEV)).cpu().numpy()[0].astype(np.float32)
    @torch.no_grad()
    def b_gene_samples(self, smiles, n_samples=50, steps=40):
        g = diffusion_vc.mol_to_graph(smiles)
        if g is None: return None
        emb = self.mol_enc(Batch.from_data_list([g]).to(DEV))
        nc=len(self.cell_vocab); nt=len(self.time_vocab)
        cell='U2OS' if 'U2OS' in self.cell_vocab else (self.cell_vocab[0] if self.cell_vocab else '')
        tc='24H' if '24H' in self.time_vocab else (self.time_vocab[0] if self.time_vocab else '')
        cell_oh=np.eye(nc,dtype=np.float32)[self.cell_vocab.index(cell)] if nc else np.zeros(0,dtype=np.float32)
        time_oh=np.eye(nt,dtype=np.float32)[self.time_vocab.index(tc)] if nt else np.zeros(0,dtype=np.float32)
        cond=torch.cat([emb, torch.zeros((1,2),device=DEV), torch.tensor(cell_oh,device=DEV)[None,:], torch.tensor(time_oh,device=DEV)[None,:]],dim=1).float()
        z=self.ddpm.sample(cond, steps=steps, n=n_samples, cfg=0.0)
        return self.vae.decode(z).cpu().numpy().astype(np.float32)
    def b_gene(self, smiles, n_samples=32, steps=40):
        samples = self.b_gene_samples(smiles, n_samples=n_samples, steps=steps)
        if samples is None: return None
        return samples.mean(0)
    def desc(self, smiles):
        m=Chem.MolFromSmiles(smiles)
        if m is None: return np.zeros(217,dtype=np.float32)
        vals=[]
        for name,fn in Descriptors.descList:
            try: v=fn(m)
            except Exception: v=0.0
            if v is None or not np.isfinite(v): v=0.0
            vals.append(float(v))
        return np.array(vals,dtype=np.float32)
    def ecfp(self, smiles, r=3, bits=2048):
        m=Chem.MolFromSmiles(smiles)
        if m is None: return np.zeros(bits,dtype=np.float32)
        return np.array(AllChem.GetMorganFingerprintAsBitVect(m,r,nBits=bits),dtype=np.float32)
    def molformer_emb(self, smiles):
        if smiles in self._mf_cache:
            return self._mf_cache[smiles]
        if self._mf_model is None:
            if not self.mf_dir.exists():
                raise FileNotFoundError(
                    'MoLFormer weights not found at ' + str(self.mf_dir) + '. '
                    'Download them first, e.g. `huggingface-cli download ibm/MoLFormer-XL-both-10pct '
                    '--local-dir models/MoLFormer-XL-both-10pct` (see README).')
            from transformers import AutoTokenizer, AutoModel
            self._mf_tokenizer = AutoTokenizer.from_pretrained(str(self.mf_dir), trust_remote_code=True)
            self._mf_model = AutoModel.from_pretrained(str(self.mf_dir), trust_remote_code=True)
            self._mf_model.eval()
        enc = self._mf_tokenizer(smiles, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            out = self._mf_model(**enc)
        emb = out.last_hidden_state[0].mean(dim=0).numpy().astype(np.float32)
        self._mf_cache[smiles] = emb
        return emb
    def predict(self, drug, n_samples=50, compute_stats=True):
        smi, shown = self.resolve(drug)
        if smi is None: raise ValueError(f'cannot resolve: {drug}')
        b_samples = self.b_gene_samples(smi, n_samples=n_samples, steps=40)
        if b_samples is None: raise ValueError(f'SMILES failed: {smi}')
        b = b_samples.mean(0)
        emb = self.gnn_emb(smi)
        if emb is None: raise ValueError(f'graph failed: {smi}')
        with torch.no_grad():
            h, path, agene, logits, axes = self.A(torch.tensor(emb, device=DEV)[None,:])
        path = path.cpu().numpy()[0].astype(np.float32)
        agene = agene.cpu().numpy()[0].astype(np.float32)
        a_head_axes = {k: float(torch.sigmoid(v).cpu().numpy()[0]) for k, v in axes.items()}
        feat_parts = [self.desc(smi), self.ecfp(smi,3,2048), b]
        if self.pni_feature_mode == 'desc+ecfp3+B_gene+molformer':
            mf = self.molformer_emb(smi)
            if mf is None or mf.shape[0] != 768:
                raise ValueError(f'MolFormer embedding failed: {smi}')
            feat_parts.append(mf)
        axes = {}
        if self.axis_models:
            X_axis = np.concatenate(feat_parts).astype(np.float32)[None, :]
            for key in AXIS_KEYS:
                if key in self.axis_models:
                    axes[AXIS_CN[key]] = float(self.axis_models[key].predict_proba(X_axis)[0, 1])
        else:
            axes = {AXIS_CN[k]: a_head_axes[k] for k in a_head_axes if k in AXIS_CN}
        X = np.concatenate(feat_parts).astype(np.float32)[None,:]
        prob = self.pni_clf.predict_proba(X)[0]
        adj_prob = prob + np.array([0.0, self.pni_bias, 0.0], dtype=np.float32)
        cls = int(adj_prob.argmax())
        cls_label = {0:'Promotes bone formation',1:'Inhibits bone formation',2:'Unrelated to bone formation'}[cls]
        # Four lineage axes are always reported; osteoblast/osteoclast are the calibrated axes, adipocyte/chondrocyte exploratory
        # Net bone direction = osteoblast - osteoclast (which side of the formation/resorption balance a drug tilts)
        net_direction = None
        if 'Osteoblast' in axes and 'Osteoclast' in axes:
            net_direction = float(axes['Osteoblast'] - axes['Osteoclast'])
        # point estimate from the GEO-anchored pathway bottleneck
        gene12328 = (agene @ self.expand_W + self.expand_intercept).astype(np.float32)
        top_idx = np.argsort(-np.abs(gene12328))
        top_path = np.argsort(-np.abs(path))
        result = {'drug':shown,'smiles':smi,'class':cls,'class_label':cls_label,
                  'P_prob':float(prob[0]),'I_prob':float(prob[1]),'N_prob':float(prob[2]),'I_decision_bias':self.pni_bias,
                  'Osteo_score':float(prob[0]-prob[1]),
                  'net_direction':net_direction,
                  'axes':axes,'genes978':agene,'gene_names978':self.genes978,
                  'genes12328':gene12328,'gene_names12328':self.genes12328,'top_idx':top_idx,
                  'pathway_scores':path,'pathway_names':self.pathway_names,'top_path_idx':top_path,
                  'A_gene978':agene}
        if compute_stats and n_samples >= 3:
            stat = self._gene_stats(b_samples, n_samples)
            result.update(stat)
        return result

    def _gene_stats(self, b_samples, n_samples):
        gene_samples = (b_samples @ self.expand_W + self.expand_intercept).astype(np.float32)
        mean = gene_samples.mean(0)
        sd = gene_samples.std(0, ddof=1)
        n = int(n_samples)
        tstat = mean / (sd / np.sqrt(n) + 1e-12)
        pval = 2.0 * (1.0 - stats.t.cdf(np.abs(tstat), df=n - 1))
        order = np.argsort(pval)
        qval = np.empty_like(pval, dtype=np.float64)
        qval[order] = pval[order] * len(pval) / (np.arange(1, len(pval) + 1, dtype=np.float64))
        qmin = np.inf
        for idx in order[::-1]:
            qval[idx] = min(qval[idx], qmin)
            qmin = min(qmin, qval[idx])
        sig = qval < 0.05
        up = (mean > 0) & sig
        down = (mean < 0) & sig
        return {
            'gene_stats_mean': mean,
            'gene_stats_sd': sd,
            'gene_stats_t': tstat,
            'gene_stats_p': pval,
            'gene_stats_fdr': qval,
            'gene_stats_sig': sig,
            'gene_stats_up': up,
            'gene_stats_down': down,
            'gene_stats_n_samples': n
        }
    def to_summary(self, r, top_n=20):
        row={'drug':r['drug'],'smiles':r['smiles'],'class':r['class'],'class_label':r['class_label'],
             'P_prob':round(r['P_prob'],4),'I_prob':round(r['I_prob'],4),'N_prob':round(r['N_prob'],4),'I_decision_bias':r.get('I_decision_bias',0.0),
             'Osteo_score':round(r['Osteo_score'],4)}
        if r.get('net_direction') is not None:
            row['net_direction'] = round(r['net_direction'], 4)
        for ax,v in r['axes'].items(): row[ax+'_score']=round(v,4)
        if 'gene_stats_fdr' in r:
            row['FDR_up'] = int(r['gene_stats_up'].sum())
            row['FDR_down'] = int(r['gene_stats_down'].sum())
            row['FDR_total'] = int(r['gene_stats_sig'].sum())
        ups=[r['gene_names12328'][i] for i in r['top_idx'] if r['genes12328'][i]>0][:top_n]
        dns=[r['gene_names12328'][i] for i in r['top_idx'] if r['genes12328'][i]<0][:top_n]
        for i,g in enumerate(ups,1): row[f'top_up_gene_{i}']=g
        for i,g in enumerate(dns,1): row[f'top_down_gene_{i}']=g
        ups=[r['pathway_names'][i] for i in r['top_path_idx'] if r['pathway_scores'][i]>0][:10]
        dns=[r['pathway_names'][i] for i in r['top_path_idx'] if r['pathway_scores'][i]<0][:10]
        for i,p in enumerate(ups,1): row[f'top_up_pathway_{i}']=p
        for i,p in enumerate(dns,1): row[f'top_down_pathway_{i}']=p
        return row


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--drug'); ap.add_argument('--smiles'); ap.add_argument('--batch'); ap.add_argument('--excel',action='store_true')
    ap.add_argument('--n_samples', type=int, default=50, help='DDPM sampling repeats for gene-level statistics')
    ap.add_argument('--no_stats', action='store_true', help='disable gene-level P-value/FDR to speed up batch screening')
    args=ap.parse_args()
    pred=PathBoneV2()
    def write_output(r):
        outdir=ROOT/'outputs'/'predictions_v2'; outdir.mkdir(parents=True,exist_ok=True)
        tag=r['drug'].replace('/','_').replace('\\','_')
        summary=outdir/f'pathbone_v2_{tag}_{time.strftime("%Y%m%d_%H%M%S")}_summary.csv'
        with open(summary,'w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(pred.to_summary(r).keys())); w.writeheader(); w.writerow(pred.to_summary(r))
        gene=outdir/f'pathbone_v2_{tag}_{time.strftime("%Y%m%d_%H%M%S")}_genes.csv'
        with open(gene,'w',newline='',encoding='utf-8-sig') as f:
            w=csv.writer(f); w.writerow(['gene_symbol','pred_logFC'])
            order=np.argsort(-np.abs(r['genes12328']))
            for i in order: w.writerow([r['gene_names12328'][i], round(float(r['genes12328'][i]),6)])
        path=outdir/f'pathbone_v2_{tag}_{time.strftime("%Y%m%d_%H%M%S")}_pathways.csv'
        with open(path,'w',newline='',encoding='utf-8-sig') as f:
            w=csv.writer(f); w.writerow(['pathway','score'])
            order=np.argsort(-np.abs(r['pathway_scores']))
            for i in order[:500]: w.writerow([r['pathway_names'][i], round(float(r['pathway_scores'][i]),6)])
        stat_path = None
        if 'gene_stats_fdr' in r:
            stat_path=outdir/f'pathbone_v2_{tag}_{time.strftime("%Y%m%d_%H%M%S")}_gene_stats.csv'
            with open(stat_path,'w',newline='',encoding='utf-8-sig') as f:
                w=csv.writer(f)
                w.writerow(['gene_symbol','mean_logFC','sd','t_stat','p_value','FDR','direction','FDR_significant'])
                for i in np.argsort(r['gene_stats_p']):
                    direction='UP' if r['gene_stats_mean'][i] > 0 else 'DOWN'
                    w.writerow([r['gene_names12328'][i], round(float(r['gene_stats_mean'][i]),6), round(float(r['gene_stats_sd'][i]),6),
                                round(float(r['gene_stats_t'][i]),6), float(r['gene_stats_p'][i]), float(r['gene_stats_fdr'][i]),
                                direction, 'Yes' if r['gene_stats_sig'][i] else 'No'])
        print('SUMMARY',summary); print('GENES',gene); print('PATHWAYS',path); print('GENE_STATS',stat_path)
    if args.drug or args.smiles:
        r=pred.predict(args.drug or args.smiles, n_samples=args.n_samples, compute_stats=not args.no_stats)
        print('class',r['class_label'],'P/I/N',[round(r['P_prob'],3),round(r['I_prob'],3),round(r['N_prob'],3)])
        print('axes',r['axes'])
        ups=[r['gene_names12328'][i] for i in r['top_idx'] if r['genes12328'][i]>0][:10]
        dns=[r['gene_names12328'][i] for i in r['top_idx'] if r['genes12328'][i]<0][:10]
        print('up genes',ups); print('down genes',dns)
        if 'gene_stats_fdr' in r:
            print('FDR<0.05', 'up', int(r['gene_stats_up'].sum()), 'down', int(r['gene_stats_down'].sum()), 'total', int(r['gene_stats_sig'].sum()))
        write_output(r)
    elif args.batch:
        rows=list(csv.DictReader(open(args.batch,encoding='utf-8-sig')))
        out=[]
        for row in rows:
            x=row.get('smiles') or row.get('drug') or row.get('name','')
            try:
                r=pred.predict(x, n_samples=args.n_samples, compute_stats=not args.no_stats); out.append(pred.to_summary(r))
            except Exception as e:
                out.append({'drug':x,'smiles':'','class':'ERROR','class_label':str(e)})
        outdir=ROOT/'outputs'/'predictions_v2'; outdir.mkdir(parents=True,exist_ok=True)
        path=outdir/f'pathbone_v2_batch_{time.strftime("%Y%m%d_%H%M%S")}.csv'
        with open(path,'w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
        print('batch',path)
    else:
        ap.print_help()

if __name__=='__main__':
    main()
