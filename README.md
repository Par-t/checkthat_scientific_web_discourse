# checkthat_scientific_web_discourse

Modular research repo for information retrieval experiments on scientific web discourse.

This repository provides a simple, extensible pipeline for preprocessing, retrieval (e.g., BM25), optional reranking, and evaluation.

## Quick start

1. Create an environment using `envs/environment.yml` (conda) or `requirements.txt` (pip).
2. See `configs/default_pipeline.yml` and `configs/experiments/exp_example.yml` for configuration examples.

## Structure

See the directory tree in the project root for modules and purpose.

## How to run (Windows CMD)

1) Create and activate env (Conda):
```
conda create -y -n ctswd -c conda-forge python=3.10 numpy pandas scikit-learn ipykernel
conda activate ctswd
```
2) Install project deps:
```
pip install -r requirements.txt
```
3) Run demos:
```
python -m src.pipeline.pipeline
python -m src.experiment.runner
```

Notes:
- Run from the repo root so `src` is importable. If needed: `set PYTHONPATH=%CD%`.
- Example config: `configs/experiments/exp_example.yml` (uses dynamic `impl`).


