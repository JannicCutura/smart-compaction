# Predictive Compaction for Lakehouse Table Formats

Paper for ICCNS 2026. Predicts compaction utility from Apache Iceberg metadata using XGBoost.

## Quick Start

```bash
make install_py    # create venv, install deps
make generate      # generate 648 Iceberg tables (~200 GB)
./run_pipeline.sh  # extract → snapshot → compact → label → train → evaluate
make tex           # compile paper
```

## Pipeline

| Stage | Script | Output |
|-------|--------|--------|
| Parameters | `code/params.py` | `code/grid.csv` (648 configs) |
| Generate | `code/generate.py` | Iceberg tables in `/mnt/data/warehouse` |
| Extract | `code/extract.py` | `data/features.csv` (17 features + 7 params) |
| Compact | `code/compact.py` | `data/compaction.csv` |
| Label | `code/label.py` | `data/dataset.csv` |
| Train | `code/train.py` | `data/model_clf.json`, `data/model_reg.json` |
| Evaluate | `code/evaluate.py` | `plots/*.pdf` (9 figures) |

## Infrastructure

- Spark 3.5.4 local mode, Iceberg 1.7.1, Java 21 Corretto
- EC2 t3a.2xlarge (8 vCPU, 32 GB), 1 TB gp3 EBS
- XGBoost 3.2, scikit-learn 1.7, statsmodels 0.14

## Structure

```
code/           Pipeline scripts (params, generate, extract, compact, label, train, evaluate)
data/           CSV outputs and trained models
paper/          LaTeX paper (IEEEtran 10pt)
plots/          Evaluation figures
```
