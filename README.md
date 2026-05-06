# 🧠 MemoryWatch
### Classical & Quantum-Inspired Anomaly Detection for Host-Based Intrusion Detection

> Host-based IDS detecting memory-based attacks (memory dumping, credential scraping, unauthorized process access) using ML anomaly detection. Classical pipeline benchmarked against quantum-inspired techniques via Qiskit on UNSW-NB15/CICIDS datasets. EPITA AL Project 2026.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Qiskit](https://img.shields.io/badge/Qiskit-quantum--inspired-blueviolet?style=flat-square&logo=ibm)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![EPITA](https://img.shields.io/badge/EPITA-AL%20Project%202026-red?style=flat-square)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Project Architecture](#-project-architecture)
- [Deliverables](#-deliverables)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Team](#-team)
- [Acknowledgements](#-acknowledgements)

---

## 🔍 Overview

**MemoryWatch** is a research project developed as part of the EPITA Action Learning (AL) program, Spring 2026, under the supervision of Professor Salman Nadeem.

The project investigates whether machine learning — both classical and quantum-inspired — can detect **memory-based cyberattacks** on a host system by learning what "normal" process and memory behavior looks like, and flagging deviations.

The core research question:
> *Can quantum-inspired ML techniques (via Qiskit) meaningfully improve anomaly detection accuracy over classical baselines for host-based intrusion detection?*

---

## 🚨 Problem Statement

When programs execute, sensitive data — passwords, cryptographic keys, session tokens — resides in RAM in **plain text**. This exposes systems to:

| Attack Type | Description |
|---|---|
| **Memory Dumping** | Full RAM snapshot extraction to read secrets offline |
| **Unauthorized Process Access** | Attaching to a running process to read its live memory |
| **Credential Scraping** | Targeting specific memory regions where credentials are stored |

Traditional antivirus tools rely on **signature-based detection** and fail completely against zero-day memory attacks. MemoryWatch takes an **anomaly detection approach** — learning normal behavior and flagging deviations — requiring no prior knowledge of the attack signature.

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      MemoryWatch IDS                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   [Dataset: UNSW-NB15 / CICIDS]                         │
│           │                                             │
│           ▼                                             │
│   [Preprocessing Pipeline]                              │
│   - Missing value handling                              │
│   - Feature scaling & encoding                          │
│   - Class imbalance treatment                           │
│           │                                             │
│           ├──────────────────────┐                      │
│           ▼                      ▼                      │
│   [Classical ML Model]   [Quantum-Inspired Model]       │
│   - Isolation Forest     - Quantum Kernel Method        │
│   - Random Forest        - Variational Classifier (VQC) │
│           │                      │                      │
│           └──────────┬───────────┘                      │
│                      ▼                                  │
│           [Evaluation & Comparison]                     │
│           - Accuracy / FPR / Precision / Recall         │
│           - Trade-off & feasibility analysis            │
│                                                         │
│   [Optional: Real-Time /proc Monitor] ──► [Live Alert]  │
└─────────────────────────────────────────────────────────┘
```

---

## 📦 Deliverables

### ✅ Core (Required)

**1. Threat Model + System Design Document**
- Formal definition of attack types and threat actors
- IDS architecture design
- Data flow and component interaction documentation

**2. Classical ML Anomaly Detection Pipeline**
- Preprocessed public dataset (UNSW-NB15 or CICIDS)
- Trained model: Isolation Forest / Random Forest
- Evaluation report: accuracy, false positive rate, precision, recall

**3. Quantum-Inspired Comparative Experiment**
- Quantum kernel method or Variational Quantum Classifier (VQC) via Qiskit
- Same dataset and pipeline as classical baseline
- Written comparative analysis: performance, trade-offs, hardware feasibility

### 🔵 Optional (If ahead of schedule)

- **Real-Time Memory Monitoring Module** — live `/proc` polling on Linux feeding the trained model for inference
- **Extended Benchmarking** — Autoencoder, One-Class SVM for a richer comparative study

---

## 📊 Dataset

This project uses one or both of the following public datasets:

| Dataset | Source | Description |
|---|---|---|
| **UNSW-NB15** | Australian Centre for Cyber Security | 2.5M records, 9 attack categories, 49 features |
| **CICIDS** | Canadian Institute for Cybersecurity | Realistic traffic with labeled attack types |

Both datasets are publicly available and widely used in academic IDS research.

- [UNSW-NB15 →](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- [CICIDS →](https://www.unb.ca/cic/datasets/ids-2017.html)

> ⚠️ Raw dataset files are not included in this repository due to size. See the links above to download them and place them in the `data/raw/` directory.

---

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- pip
- Linux environment (recommended)
- Git

### Clone the Repository

```bash
git clone https://github.com/your-username/memorywatch-ids.git
cd memorywatch-ids
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Requirements Overview

```
scikit-learn
pandas
numpy
matplotlib
seaborn
qiskit
qiskit-machine-learning
jupyter
imbalanced-learn
```

---

## 🚀 Usage

### 1. Preprocess the Dataset

```bash
python src/preprocessing/preprocess.py --dataset unsw-nb15 --input data/raw/ --output data/processed/
```

### 2. Train Classical ML Model

```bash
python src/classical/train.py --model isolation_forest --data data/processed/
```

### 3. Evaluate Classical Model

```bash
python src/classical/evaluate.py --model models/isolation_forest.pkl --data data/processed/test/
```

### 4. Run Quantum-Inspired Experiment

```bash
python src/quantum/train_qiskit.py --method quantum_kernel --data data/processed/
```

### 5. Compare Results

```bash
python src/evaluation/compare.py --classical models/isolation_forest.pkl --quantum models/quantum_kernel.pkl
```

### 6. (Optional) Real-Time Monitor

```bash
sudo python src/monitor/proc_monitor.py --model models/isolation_forest.pkl
```

> ⚠️ Real-time monitoring requires root privileges on Linux.

---

## 📈 Results

*Results will be populated after experimentation is complete.*

| Model | Accuracy | False Positive Rate | Precision | Recall | F1 Score |
|---|---|---|---|---|---|
| Isolation Forest | TBD | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD | TBD |
| Quantum Kernel (Qiskit) | TBD | TBD | TBD | TBD | TBD |
| VQC (Qiskit) | TBD | TBD | TBD | TBD | TBD |

---

## 📁 Project Structure

```
memorywatch-ids/
│
├── data/
│   ├── raw/                  # Raw dataset files (not tracked by git)
│   └── processed/            # Preprocessed splits
│
├── deliverables/
│   ├── proposal/             # Approved project proposal
│   ├── bibliography/         # Bibliography document
│   ├── literature_review/    # Literature review outline
│   └── thesis/               # Final thesis
│
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb
│   ├── 03_classical_ml.ipynb
│   └── 04_quantum_experiment.ipynb
│
├── src/
│   ├── preprocessing/        # Data cleaning and feature engineering
│   ├── classical/            # Classical ML pipeline
│   ├── quantum/              # Qiskit quantum-inspired models
│   ├── evaluation/           # Metrics and comparison scripts
│   └── monitor/              # Optional: real-time /proc monitor
│
├── models/                   # Saved trained models
├── results/                  # Evaluation outputs, plots, reports
├── requirements.txt
└── README.md
```

---

## 👥 Team

| Name | Role |
|---|---|
| **Vignesh MANI** | TBD |
| **Prachin TULADHAR** | TBD |
| **Esala WIJERATHNA** | TBD |
| **Ghita MANDRI** | TBD |

*Supervised by Professor Salman Nadeem — EPITA, Spring 2026*

---

##   Acknowledgements

- Professor Salman Nadeem for project guidance and feedback
- Australian Centre for Cyber Security for the UNSW-NB15 dataset
- Canadian Institute for Cybersecurity for the CICIDS dataset
- IBM / Qiskit open-source community
- EPITA for the Action Learning program framework

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.
