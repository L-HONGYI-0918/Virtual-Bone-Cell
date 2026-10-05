# PathBone-MF: A lineage-aware interpretable virtual bone cell model for drug bone-effect prediction

PathBone-MF predicts a drug's effect on bone lineage cell fate **from the molecular structure (SMILES) alone**, without any wet-lab input. It outputs:

- **P/I/N classification** — whether the drug promotes bone formation (P), inhibits bone formation (I), or is unrelated to bone formation (N);
- **Four lineage direction scores** — osteoblast, osteoclast, adipocyte, chondrocyte;
- **Net bone direction** (osteoblast − osteoclast), a quantitative readout of bone formation–resorption coupling;
- **Mechanistic output** — 3,253 pathway activation scores and 12,328 gene-level expression changes, providing a "compound → pathway → gene" interpretation chain.

## System requirements

- Python 3.10+
- CUDA-capable GPU recommended (8 GB VRAM); pure CPU also works but is slower
- 16 GB RAM recommended

## Installation

```bash
pip install -r requirements.txt
```

## Model weights

Place the model weights and the MoLFormer tokenizer/model under the `models/` directory. The script locates them automatically. Required files:

- `pathbone_v2_mf_pni.joblib` — PNI classifier
- `pathbone_v2_mf_pni_decision.json` — anti-osteogenic decision bias
- `pathbone_v2_mf_axes.joblib` — four independent lineage classifiers
- `pathbone_v2_*.pt` / `*.npz` — VAE, DDPM, GNN, pathway bottleneck and gene-expansion weights
- `MoLFormer-XL-both-10pct/` — the pre-trained MoLFormer model

> **MoLFormer is not bundled in this repository** (it is a public IBM pre-trained model, ~179 MB single weight file, exceeding GitHub's file limit). Download it from HuggingFace into `models/MoLFormer-XL-both-10pct/`:
>
> ```bash
> pip install -U huggingface_hub
> huggingface-cli download ibm/MoLFormer-XL-both-10pct --local-dir models/MoLFormer-XL-both-10pct
> ```
>
> Alternatively, the script also looks for it at `data/models/MoLFormer-XL-both-10pct`.

## Quick start

```bash
# by drug name
python predict_v2.py --drug simvastatin

# by SMILES
python predict_v2.py --smiles "O=C1c2c(O)cc(O)cc2OC(c2ccc(O)c(O)c2)=C1O"

# batch screening (input CSV with columns: name,smiles)
python predict_v2.py --batch example_input.csv

# enable gene-level statistics (50 DDPM sampling repeats)
python predict_v2.py --drug quercetin --n_samples 50

# fast screening without gene-level statistics
python predict_v2.py --batch example_input.csv --no_stats
```

## Outputs

Results are written to `outputs/predictions_v2/`:

| File | Content |
|------|---------|
| `*_summary.csv` | P/I/N class, probabilities, Osteo score, four lineage scores, net bone direction, FDR-significant gene counts, top genes/pathways |
| `*_genes.csv` | 12,328 gene-level point-estimate logFC, ranked by effect size |
| `*_gene_stats.csv` | mean, SD, t-statistic, p-value, BH-FDR and direction for 12,328 genes |
| `*_pathways.csv` | 3,253 pathway activation scores |

Gene-level statistics: the model performs `N` DDPM generation repeats per input, expands each 978-landmark-gene sample to 12,328 genes, then applies a one-sample t-test per gene with Benjamini–Hochberg FDR correction (significance threshold FDR < 0.05).

## Interpretation notes

- `P` = pro-osteogenic tendency; `I` = anti-osteogenic tendency; `N` = neutral / no clear bone effect
- `Osteo_score = P_prob − I_prob`
- The four lineage axes are independent classifiers. Net bone direction = osteoblast − osteoclast score.
- Pathway/gene outputs support mechanistic hypothesis generation; they are not a substitute for experimental validation.
- Predictions are intended for candidate screening and hypothesis generation, **not** for clinical decision-making.

## Data availability & reproducibility

- Training data: LINCS L1000 (NCBI GEO `GSE92742`, `GSE70138`)
- External validation: `GSE28074` (evaluation only, excluded from training)
- This repository ships the inference code and model weights. Training data and training scripts are not included.

## Known limitations

- Training cells are LINCS cancer cell lines (mainly U2OS), not primary osteoblasts/BMSCs.
- Three-class accuracy is 75.6% (131 unique test drugs); suitable for screening, not clinical use.
- Static single-timepoint prediction; does not simulate differentiation dynamics or cell–cell coupling.

## License

See `LICENSE`.

## Citation

If you use PathBone-MF, please cite: [citation to be added upon publication].

## Reproducing the reported results

Evaluation code is provided in `src/eval/`:

| Script | Produces |
|---|---|
| `make_test_set_135.py` | the strict independent test set (135 perturbation samples, 131 unique drugs) |
| `scaffold_split_135.py` | scaffold-aware split with same-scaffold exclusion |
| `final_metrics_135_drug.py` | Table 1 (three-way accuracy, per-class recall, AUC) |
| `run_bib_ablation_baseline.py` | Table 2 (feature ablation) and Table 3 (baselines), with 15-fold cross-validation and paired tests |
| `train_axis_classifiers_mf.py` | Table S2 (four lineage classifiers) |
| `cross_cellline_val.py` | Fig. 7 (cross-cell-line consistency) |

Supporting data are in `data/`:

| File | Content |
|---|---|
| `labels_expanded.json` | the label library: 407 compounds with ternary bone-effect labels (label 0: 129 / label 1: 149 / label 2: 129) |
| `compound_bone_labels.csv` | per-perturbation label assignment (LINCS pert_id, cmap_name, label); rows with label -1 are unlabelled perturbations |
| `label_qa_summary.json` | PubMed evidence for each labelled compound |
| `multi_axis_labels_v1.csv` | lineage-axis labels (osteoblast, osteoclast, adipocyte, chondrocyte) |
| `test_set_drugs_135.json` | the strict independent test set |

The decision rule is documented in `models/pathbone_v2_mf_pni_decision.json`: an offset of +0.56 is added to the
inhibits-class probability before the argmax, selected on training-drug grouped cross-validation only.

The evaluation scripts expect the LINCS L1000 Level 5 data (GSE92742, GSE70138) and the weights in `models/`.
Absolute paths at the top of each script may need to be adjusted to your local data layout.