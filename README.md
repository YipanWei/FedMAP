# FedMAP: Rethinking Federated Prompt Learning for Medical Images

<p align="center">
  <a href="https://openreview.net/forum?id=3LymHCdeRd">
    <img src="https://img.shields.io/badge/Paper-OpenReview-red" alt="FedMAP paper on OpenReview">
  </a>
  <a href="https://github.com/YipanWei/FedMAP">
    <img src="https://img.shields.io/badge/Code-GitHub-000000?logo=github" alt="Code">
  </a>
  <a href="#-data-preparation">
    <img src="https://img.shields.io/badge/Data-User%20Prepared-ffcc00" alt="Data preparation">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT license">
  </a>
</p>

<p align="center">
  <b>Rethinking Federated Prompt Learning for Medical Images: From Textual Tuning to Visual Manifold Anchoring</b>
</p>

<p align="center">
  Yipan Wei, Wenke Huang, Yapeng Li, He Li, Qixin Zhang, Mang Ye, Bo Du
</p>

<p align="center">
  <b>ICML 2026 Regular Paper</b>
</p>

<p align="center">
  <a href="#-introduction">Introduction</a> •
  <a href="#-motivation">Motivation</a> •
  <a href="#-methods">Methods</a> •
  <a href="#-contributions">Contributions</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-main-results">Main Results</a> •
  <a href="#-citation">Citation</a>
</p>

<p align="center">
  <img src="assets/readme/motivation-overview.png" alt="FedMAP motivation overview" width="78%">
</p>

<p align="center">
  <sub><em><strong>Motivation overview.</strong> Medical image features often suffer from manifold collapse and cross-client topological misalignment, making purely textual prompt tuning unreliable under federated heterogeneity.</em></sub>
</p>

## 📌 Introduction

Federated prompt learning adapts vision-language models to privacy-sensitive medical image classification by exchanging only lightweight prompt parameters. Most existing methods focus on **textual tuning**, assuming that the frozen CLIP image encoder already provides a reliable visual geometry.

**FedMAP** challenges this assumption. In medical images, subtle morphology and cross-center acquisition shifts can collapse discriminative visual directions and misalign class topology across clients. FedMAP shifts the paradigm from textual prompt adaptation to **visual manifold anchoring**, using semantic anchors and topology-aware regularization to reshape the client-side visual space without exposing raw medical images.

The current release provides the public training code, dataset wrappers, semantic-anchor tensors, baseline trainers, and launcher scripts needed to reproduce the public FedISIC and FedCamelyon17 protocols. Private medical data used in the paper is not redistributed.

## 🎯 Motivation

FedMAP targets two geometric failure modes of medical federated prompt learning: intra-client manifold collapse and inter-client topological misalignment.

<p align="center">
  <img src="assets/readme/motivation-rank.png" alt="Effective-rank evidence for medical manifold collapse" width="78%">
</p>

<p align="center">
  <sub><em><strong>Medical manifold collapse.</strong> Frozen CLIP visual features have lower effective rank on medical datasets than on natural-image domains, indicating compressed class manifolds.</em></sub>
</p>

<p align="center">
  <img src="assets/readme/motivation-consistency.png" alt="Neighborhood-consistency evidence for medical topological misalignment" width="78%">
</p>

<p align="center">
  <sub><em><strong>Topological misalignment.</strong> Medical clients exhibit lower neighborhood consistency, showing that local visual geometry is less stable across sites.</em></sub>
</p>

These observations suggest that shifting text prompts alone is insufficient: the visual manifold itself needs a stable semantic reference during federated optimization.

## 🧠 Methods

FedMAP inserts lightweight visual prompts into the frozen CLIP image encoder. A server-side semantic codebook is built from class labels and LLM-derived attributes, then used as a client-invariant synchronization signal. Each client optimizes visual prompts with classification supervision plus geometry-aware losses, and the server aggregates only prompt parameters.

<p align="center">
  <img src="assets/readme/method-overview.png" alt="FedMAP method overview" width="78%">
</p>

<p align="center">
  <sub><em><strong>FedMAP pipeline.</strong> Manifold Semantic Anchoring aligns image features to semantic anchors, while Topology Structural Alignment preserves class-relation structure across clients.</em></sub>
</p>

FedMAP has three main ingredients:

- **Manifold Semantic Anchoring:** aligns visual features with semantic anchors to recover discriminative directions suppressed by medical manifold collapse.
- **Topology Structural Alignment:** distills a text-derived class-relation matrix into the visual prototype geometry, reducing cross-client structural drift.
- **Federated visual prompting:** uploads only lightweight visual prompts for FedAvg-style aggregation while keeping image data local.

## ✨ Contributions

- **A geometry-first view of medical FPL:** We identify medical manifold collapse and inter-client topological misalignment as two failure modes of text-centric federated prompt learning.
- **Visual Manifold Anchoring:** FedMAP uses semantic anchors and relation alignment to reshape visual features rather than only shifting text-side decision boundaries.
- **Lightweight federated adaptation:** The method updates compact visual prompts, giving strong performance with low communication overhead.
- **Consistent empirical gains:** FedMAP improves accuracy and Macro-F1 across FedISIC, FedCamelyon17, and the private ultrasound benchmark reported in the paper.

## 🚀 Quick Start

### ⚙️ Setup Environment

```bash
git clone https://github.com/YipanWei/FedMAP.git
cd FedMAP

conda create -n fedmap python=3.10 -y
conda activate fedmap
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build that matches your machine if the default package resolver does not select the correct one.

### 🩺 Data Preparation

Put raw datasets under:

```text
Scripts/data/
```

The public release supports:

| Dataset key | Dataset wrapper | Clients |
| --- | --- | ---: |
| `fedisic` | `FedISIC` | 6 |
| `fedcamelyon` | `FedCamelyon17MD` | 5 |

Dataset behavior is controlled by YAML files in `Scripts/configs/datasets/`. Minimal semantic resources are kept under `Scripts/embeddings/<dataset>/text/`; `*_template_attributes.pt` is used by `FedMAP`, and `*_class_attributes.pt` is used by `FedMVP`.

The repository does not redistribute raw medical datasets. Users should obtain datasets from their official sources and arrange them according to the dataset wrappers.

### 🏃 Run Experiments

Run FedMAP on FedISIC:

```bash
bash Scripts/run_experiment.sh \
  --trainer FedMAP \
  --dataset fedisic \
  --seed 1 \
  --gpu 0
```

Switch method or dataset:

```bash
bash Scripts/run_experiment.sh --trainer FedCLIP --dataset fedcamelyon --seed 1 --gpu 0
bash Scripts/run_experiment.sh --trainer FedMVP  --dataset fedisic     --seed 42 --gpu 1
```

Run a short smoke test:

```bash
bash Scripts/run_experiment.sh \
  --trainer FedProxLPT \
  --dataset fedisic \
  --seed 1 \
  --gpu 0 \
  --num_workers 0 \
  OPTIM.ROUND 2 OPTIM.MAX_EPOCH 1 OUTPUT_DIR output/smoke_fedproxlpt
```

Extra arguments after the launcher options are passed directly to `federated_main.py` and the YACS config system.

## 📊 Main Results

FedMAP consistently improves over federated prompt-learning and CLIP adaptation baselines on dermoscopy, histopathology, and a private ultrasound benchmark. The private benchmark is reported as part of the paper results, but the public code release does not redistribute private data.

<p align="center">
  <img src="assets/readme/main-results.png" alt="Main comparative results on FedISIC, FedCamelyon17, and the private benchmark" width="92%">
</p>

<p align="center">
  <sub><em><strong>Main comparison.</strong> FedMAP achieves the best average Accuracy and Macro-F1 on all three benchmarks reported in the paper.</em></sub>
</p>

| Benchmark | FedMAP Accuracy | FedMAP Macro-F1 | Improvement |
| --- | ---: | ---: | --- |
| FedISIC | 73.29 ± 1.29 | 59.47 ± 1.98 | +7.16 Acc / +19.56 F1 |
| FedCamelyon17 | 93.15 ± 0.87 | 92.99 ± 0.81 | +2.62 Acc / +2.01 F1 |
| Private ultrasound | 89.60 ± 1.33 | 88.04 ± 1.83 | +18.85 Acc / +20.84 F1 |

## 🔬 Interesting Results

<p align="center">
  <img src="assets/readme/analysis-ablation.png" alt="FedMAP module ablation" width="78%">
</p>

<p align="center">
  <sub><em><strong>Module ablation.</strong> MSA and TSA are complementary, and their combination gives the strongest performance.</em></sub>
</p>

<p align="center">
  <img src="assets/readme/analysis-convergence.png" alt="FedMAP convergence curves" width="78%">
</p>

<p align="center">
  <sub><em><strong>Convergence.</strong> FedMAP converges faster and reaches a higher final accuracy across communication rounds.</em></sub>
</p>

<p align="center">
  <img src="assets/readme/analysis-lambda.png" alt="FedMAP lambda sensitivity" width="52%">
</p>

<p align="center">
  <sub><em><strong>Structural-loss sensitivity.</strong> Performance is stable across a broad range and peaks around the setting used in the paper.</em></sub>
</p>

## 🗂️ Repository Structure

```text
FedMAP/
├── Scripts/
│   ├── federated_main.py        # federated training entrypoint
│   ├── run_experiment.sh        # one-command public launcher
│   ├── trainers/                # FedMAP and public baselines
│   ├── datasets/                # public dataset wrappers
│   ├── configs/datasets/        # federated dataset protocols
│   ├── embeddings/              # minimal semantic-anchor resources
│   ├── Dassl/                   # local training infrastructure
│   ├── clip/                    # CLIP implementation used by trainers
│   └── utils/                   # config, FL, logging, and aggregation helpers
├── assets/readme/               # lightweight README figures
├── requirements.txt
├── LICENSE
└── README.md
```

Local-only folders such as `Scripts/output/`, `Scripts/logs/`, raw `Scripts/data/`, caches, and internal notes are intentionally excluded from git.

## 🧪 Reproducibility

Default public configs use `50` federated rounds and `1` local epoch:

```text
OPTIM.ROUND: 50
OPTIM.MAX_EPOCH: 1
```

Supported public trainers:

| Trainer | Role |
| --- | --- |
| `FedMAP` | Visual prompt tuning with manifold semantic anchoring and topology structural alignment. |
| `VPT` | Visual prompt tuning baseline. |
| `PROMPTFL` | Federated prompt learning baseline. |
| `FedCLIP` | Federated CLIP-style prompt baseline. |
| `FedAPT` | Federated adaptive prompt tuning baseline. |
| `FedCoCoOP` | Federated CoCoOP-style baseline. |
| `FedKgCoOP` | Knowledge-guided CoCoOP-style baseline. |
| `FedMVP` | Multi-view prompt baseline. |
| `FedProxLPT` | FedProx regularized language prompt tuning baseline. |
| `FOCoOP` | Federated out-of-distribution CoOp-style baseline. |
| `CLIP` | Zero-shot / frozen CLIP reference. |

Outputs are generated locally:

```text
Scripts/output/<dataset>/beta:<beta>/<trainer>/
Scripts/logs/
```

These folders are ignored by git so the public repository remains lightweight. The public release has been smoke-tested with all listed trainers on `fedisic` using `OPTIM.ROUND=1`, and a 2-round regression check was run for `FedProxLPT`.

## 📚 Citation

```bibtex
@inproceedings{wei2026rethinking,
  title={Rethinking Federated Prompt Learning for Medical Images: From Textual Tuning to Visual Manifold Anchoring},
  author={Wei, Yipan and Huang, Wenke and Li, Yapeng and Li, He and Zhang, Qixin and Ye, Mang and Du, Bo},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```

## 📄 License

This repository is released under the MIT License.
