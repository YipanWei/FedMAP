<div align="center">

# FedMAP

### Rethinking Federated Prompt Learning for Medical Images: From Textual Tuning to Visual Manifold Anchoring

**ICML 2026 Regular Paper**

<p>
  <b>Federated Learning</b> ·
  <b>Medical Imaging</b> ·
  <b>Prompt Tuning</b> ·
  <b>Vision-Language Models</b>
</p>

<p>
  <a href="#-overview">Overview</a> ·
  <a href="#-quick-start">Quick Start</a> ·
  <a href="#-repository-layout">Layout</a> ·
  <a href="#-citation">Citation</a>
</p>

</div>

## 🔎 Overview

FedMAP is a federated prompt learning framework for medical image classification with CLIP-based vision-language models. It revisits a common assumption in federated prompt tuning: that textual prompt adaptation is the central lever for cross-client generalization. Instead, FedMAP anchors the learning process with fixed semantic prototypes and aligns the visual manifold through class-level topology.

The released code contains the training pipeline, method implementations, dataset configurations, and lightweight semantic anchor resources needed to reproduce FedMAP-style experiments. Raw datasets, internal notes, review materials, historical logs, and generated outputs are intentionally excluded from this public repository.

## ✨ Method Highlights

| Component | Role |
| --- | --- |
| 🧭 **Medical semantic anchors** | Fixed text-side class anchors provide a stable cross-client reference frame. |
| 🧬 **Visual manifold anchoring** | Visual prototypes are maintained during federated training and aligned with the semantic topology. |
| 🛰️ **Federated prompt tuning** | Only lightweight prompt parameters are adapted and communicated. |
| 🛠️ **Unified public entrypoint** | One launcher selects the trainer, dataset, seed, GPU, and extra experiment options. |

## 🗂️ Repository Layout

```text
FedMAP/
  Scripts/
    run_experiment.sh      # public launcher
    federated_main.py      # federated training controller
    trainers/              # FedMAP and baseline trainers
    configs/datasets/      # federated dataset protocols
    datasets/              # dataset wrappers
    utils/                 # config, data, training, and FL utilities
    Dassl/                 # local Dassl-based training infrastructure
    clip/                  # CLIP implementation
    embeddings/            # lightweight text-anchor tensors
  README.md
  LICENSE
  requirements.txt
```

Generated files are written locally to `Scripts/output/` and `Scripts/logs/`; both are ignored by git.

## ⚙️ Installation

```bash
conda create -n fedmap python=3.10 -y
conda activate fedmap
pip install -r requirements.txt
```

The code expects PyTorch with CUDA support for full training. Install the PyTorch build that matches your CUDA version if the default `pip install` route is not appropriate for your machine.

## 🩺 Data Preparation

Place datasets under:

```text
Scripts/data/
```

Dataset protocols are configured in:

```text
Scripts/configs/datasets/
```

Each YAML file defines the dataset name, client/domain split, input resolution, optimizer, and communication rounds. The public release does not redistribute raw medical datasets.

## 🚀 Quick Start

Run FedMAP on a dataset:

```bash
bash Scripts/run_experiment.sh --trainer VPT_Ma --dataset fedisic --seed 1 --gpu 0
```

Switch method or dataset:

```bash
bash Scripts/run_experiment.sh --trainer FedCLIP --dataset fedcamelyon --seed 1 --gpu 0
bash Scripts/run_experiment.sh --trainer VPT_Ma_T --dataset whu --seed 42 --gpu 1
```

Pass additional options directly to `federated_main.py`:

```bash
bash Scripts/run_experiment.sh --trainer VPT_Ma --dataset fedisic --gpu 0 --num_workers 4 --lambda_struct 10
```

Common trainers include `VPT_Ma`, `VPT_Ma_T`, `FedCLIP`, `FedMVP`, `PromptFL`, `LPT`, and `CLIP`.

## 📦 Outputs

Experiment outputs are generated locally:

```text
Scripts/output/<dataset>/beta:<beta>/<trainer>/
Scripts/logs/
```

These directories are not versioned. This keeps the GitHub repository lightweight and lets users reproduce results from the released scripts and configurations.

## 📚 Citation

```bibtex
@inproceedings{wei2026rethinking,
  title={Rethinking Federated Prompt Learning for Medical Images: From Textual Tuning to Visual Manifold Anchoring},
  author={Wei, Yipan},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```

## 📄 License

This repository is released under the MIT License.
