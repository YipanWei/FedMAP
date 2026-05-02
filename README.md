# FedMAP

Official code release for **Rethinking Federated Prompt Learning for Medical Images: From Textual Tuning to Visual Manifold Anchoring**, accepted as a regular paper at ICML 2026.

FedMAP studies federated prompt learning for medical image classification with CLIP-based vision-language models. The main method uses fixed semantic anchors and visual prototype topology alignment for federated visual prompt tuning.

## Repository Layout

```text
FedMAP/
  Scripts/              # training code, configs, trainers, and the unified launcher
    run_experiment.sh   # single public entrypoint
    federated_main.py   # federated training entrypoint
    trainers/           # FedMAP and baseline trainers
    configs/            # dataset configs
    embeddings/         # small text-anchor tensors required by FedMAP
  README.md
  LICENSE
  requirements.txt
```

This public release intentionally does not include internal notes, rebuttal files, raw datasets, historical logs, or generated experiment outputs. Training results are written to local `Scripts/output/`, which is ignored by git.

## Setup

```bash
conda create -n fedmap python=3.10 -y
conda activate fedmap
pip install -r requirements.txt
```

Prepare the datasets under:

```text
Scripts/data/
```

The dataset configs are in `Scripts/configs/datasets/`. Each config defines the dataset name, number of clients, domains, input size, and optimization settings.

## Run

Use the unified launcher from the repository root:

```bash
bash Scripts/run_experiment.sh --trainer VPT_Ma --dataset fedisic --seed 1 --gpu 0
```

You can switch methods and datasets:

```bash
bash Scripts/run_experiment.sh --trainer FedCLIP --dataset fedcamelyon --seed 1 --gpu 0
bash Scripts/run_experiment.sh --trainer VPT_Ma_T --dataset whu --seed 42 --gpu 1
```

Additional arguments are passed through to `federated_main.py`:

```bash
bash Scripts/run_experiment.sh --trainer VPT_Ma --dataset fedisic --gpu 0 --num_workers 4 --lambda_struct 10
```

## Citation

```bibtex
@inproceedings{wei2026rethinking,
  title={Rethinking Federated Prompt Learning for Medical Images: From Textual Tuning to Visual Manifold Anchoring},
  author={Wei, Yipan},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```
