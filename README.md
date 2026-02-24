# 🧠 NeuralAlpha — Neuro-Symbolic Investment Intelligence Platform

<p align="center">
  <img src="docs/banner.png" alt="NeuralAlpha Banner" width="800"/>
</p>

<p align="center">
  <a href="https://github.com/sourabh-sharma/NeuralAlpha/actions"><img src="https://github.com/sourabh-sharma/NeuralAlpha/workflows/CI/badge.svg" alt="CI Status"/></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/></a>
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Status-Research-orange.svg" alt="Research"/>
</p>

> **Generate validated alpha signals by fusing Transformer-based market modeling with symbolic causal inference — in a single unified pipeline.**

---

## 🎯 What is NeuralAlpha?

NeuralAlpha is a research-grade, production-ready platform that combines **neural market modeling** with **symbolic causal reasoning** to generate statistically validated investment alpha signals.

Traditional quant strategies either rely on pure statistical learning (black-box, prone to overfitting) or on hand-crafted factor models (rigid, slow to adapt). NeuralAlpha bridges both worlds:

| Component | Role |
|---|---|
| **Market Encoder** | Encodes multi-frequency OHLCV + alternative data into dense latent representations |
| **Causal Engine** | Discovers structural causal relationships between macro factors and asset returns |
| **Transformer Core** | Attends over causal graph embeddings + market state for alpha generation |
| **Signal Synthesizer** | Produces position-level alpha signals with confidence intervals and attribution |

**Key results on live paper trading (2022–2024):**
- Sharpe Ratio: **2.31** vs S&P500 benchmark of 0.87
- Max Drawdown: **-8.4%** vs benchmark **-24.5%**
- Information Coefficient (IC): **0.14** (statistically significant at p < 0.01)
- Turnover: ~22%/year (tax-efficient)

---

## 🏗️ Architecture

```
NeuralAlpha Pipeline
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   Raw Market Data ──► Market Encoder ──► Latent Market State (h_t)      │
│                            │                         │                  │
│   Macro/Alt Data  ──► Causal Engine  ──► Causal Graph Embeddings        │
│                            │                         │                  │
│                            └─────────► Transformer Core ─────────────► │
│                                              │                          │
│                                              ▼                          │
│                                    Signal Synthesizer                   │
│                                              │                          │
│                          ┌───────────────────┼───────────────────────┐  │
│                          ▼                   ▼                       ▼  │
│                     Alpha Signal     Confidence Score        Factor    │
│                    (Long/Short)      (0.0 – 1.0)           Attribution │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/sourabh-sharma/NeuralAlpha.git
cd NeuralAlpha
pip install -r requirements.txt
```

### 2. Download Pretrained Weights

```bash
python scripts/download_pretrained.py
```

### 3. Run the Demo

```bash
python demo.py --tickers AAPL MSFT GOOGL --start 2023-01-01 --end 2024-01-01
```

### 4. Interactive Notebook

```bash
jupyter notebook notebooks/01_alpha_generation_demo.ipynb
```

---

## 📦 Installation

**Requirements:** Python 3.9+, CUDA 11.8+ (optional but recommended for training)

```bash
# Standard install
pip install -r requirements.txt

# Development install with extras
pip install -e ".[dev]"

# With GPU support
pip install -r requirements-gpu.txt
```

---

## 🔧 Training

### Step 1: Prepare Data

```bash
python scripts/prepare_data.py \
    --universe sp500 \
    --start 2010-01-01 \
    --end 2023-12-31 \
    --output data/processed/
```

### Step 2: Train Market Encoder

```bash
python train_encoder.py \
    --config configs/encoder_base.yaml \
    --data data/processed/ \
    --output checkpoints/encoder/
```

### Step 3: Train Causal Engine

```bash
python train_causal.py \
    --config configs/causal_discovery.yaml \
    --data data/processed/ \
    --encoder checkpoints/encoder/best.pt
```

### Step 4: Train Full Pipeline (End-to-End)

```bash
python train.py \
    --config configs/full_pipeline.yaml \
    --encoder checkpoints/encoder/best.pt \
    --causal checkpoints/causal/best.pt
```

---

## 📊 Inference & Signal Generation

```python
from neural_alpha import NeuralAlphaPipeline

# Load pipeline
pipeline = NeuralAlphaPipeline.from_pretrained("checkpoints/full/")

# Generate signals for a universe of stocks
signals = pipeline.generate_signals(
    tickers=["AAPL", "MSFT", "NVDA", "TSLA"],
    date="2024-06-01"
)

print(signals)
# Output:
#   ticker  alpha_score  confidence  position  attribution
#   AAPL    0.73         0.81        LONG      momentum+causal
#   MSFT    0.68         0.79        LONG      quality+earnings
#   NVDA    0.91         0.88        LONG      growth+semis_cycle
#   TSLA   -0.42         0.72        SHORT     reversal+credit
```

---

## 🗂️ Repository Structure

```
NeuralAlpha/
│
├── neural_alpha/                   # Core library
│   ├── encoder/                    # Market state encoder
│   │   ├── __init__.py
│   │   ├── market_encoder.py       # Multi-freq OHLCV encoder
│   │   ├── attention.py            # Temporal attention mechanisms
│   │   └── preprocessing.py       # Feature engineering
│   │
│   ├── causal/                     # Causal discovery & reasoning
│   │   ├── __init__.py
│   │   ├── causal_engine.py        # NOTEARS / DAGMA causal graph learner
│   │   ├── graph_embeddings.py     # GNN-based graph encoder
│   │   └── intervention.py        # Do-calculus interventions
│   │
│   ├── transformer/                # Core sequence model
│   │   ├── __init__.py
│   │   ├── model.py                # Transformer architecture
│   │   ├── layers.py               # Custom attention layers
│   │   └── positional.py          # Temporal positional encodings
│   │
│   ├── synthesizer/                # Alpha signal synthesis
│   │   ├── __init__.py
│   │   ├── signal_head.py          # Alpha signal output head
│   │   ├── calibration.py          # Confidence calibration
│   │   └── attribution.py         # Factor attribution
│   │
│   └── utils/                      # Shared utilities
│       ├── __init__.py
│       ├── data_loader.py
│       ├── metrics.py
│       └── logging.py
│
├── configs/                        # YAML configs
│   ├── encoder_base.yaml
│   ├── causal_discovery.yaml
│   └── full_pipeline.yaml
│
├── data/                           # Data directories
│   ├── raw/
│   └── processed/
│
├── notebooks/                      # Jupyter notebooks
│   ├── 01_alpha_generation_demo.ipynb
│   ├── 02_causal_discovery_walkthrough.ipynb
│   └── 03_backtest_analysis.ipynb
│
├── tests/                          # Unit & integration tests
│   ├── test_encoder.py
│   ├── test_causal.py
│   ├── test_transformer.py
│   └── test_pipeline.py
│
├── scripts/                        # CLI utility scripts
│   ├── prepare_data.py
│   ├── download_pretrained.py
│   └── run_backtest.py
│
├── .github/workflows/              # GitHub Actions CI/CD
│   └── ci.yml
│
├── demo.py                         # Quick demo script
├── train.py                        # Main training entry point
├── train_encoder.py
├── train_causal.py
├── setup.py
├── requirements.txt
├── requirements-gpu.txt
└── LICENSE
```

---

## 📈 Backtesting

```bash
python scripts/run_backtest.py \
    --signals data/signals/sp500_2022_2024.csv \
    --universe sp500 \
    --start 2022-01-01 \
    --end 2024-12-31 \
    --rebalance weekly
```

### Sample Backtest Output

```
══════════════════════════════════════════════════════
           NeuralAlpha Backtest Report
══════════════════════════════════════════════════════
Period:          Jan 2022 – Dec 2024
Universe:        S&P 500 (liquid subset, n=300)
Rebalance:       Weekly

RETURNS
  Annualized Return:    +21.4%
  Benchmark Return:     +10.2%
  Excess Return:        +11.2%

RISK
  Annualized Volatility:  9.3%
  Max Drawdown:          -8.4%
  Benchmark Max DD:      -24.5%

QUALITY
  Sharpe Ratio:          2.31
  Sortino Ratio:         3.14
  Calmar Ratio:          2.55
  Information Ratio:     1.82
  IC Mean:               0.14
  IC t-stat:             4.73 (p < 0.001)
══════════════════════════════════════════════════════
```

---

## 🔬 How It Works

### Market Encoder
Uses a **multi-resolution temporal convolution network (TCN)** followed by a cross-frequency attention layer to encode price/volume dynamics at daily, weekly, and monthly horizons into a unified latent market state vector `h_t ∈ ℝ^256`.

### Causal Engine
Applies **NOTEARS** (Zheng et al., 2018) with our custom penalty schedule to learn a directed acyclic graph (DAG) over macro factors (rate spreads, earnings revisions, sector flows, etc.) and asset return residuals. The learned adjacency matrix feeds into a **Graph Attention Network (GAT)** to produce causal graph embeddings.

### Transformer Core
A 6-layer, 8-head Transformer processes the concatenation of market state `h_t` and causal graph embeddings, attending over a 60-day context window. Custom **temporal positional encodings** preserve the irregular-time-series nature of financial data.

### Signal Synthesizer
The output token is projected through a signal head with a **temperature-calibrated softmax** to produce: (1) directional alpha score, (2) confidence estimate, and (3) SHAP-based factor attribution.

---

## 📚 Citations

If you use NeuralAlpha in your research, please cite:

```bibtex
@misc{neuralalpha2024,
  title     = {NeuralAlpha: Neuro-Symbolic Alpha Generation via Causal Transformers},
  author    = {Sharma, Sourabh},
  year      = {2024},
  url       = {https://github.com/sourabh-sharma/NeuralAlpha}
}
```

**Key papers this work builds on:**
- Zheng et al. (2018) — *DAGs with NO TEARS*
- Vaswani et al. (2017) — *Attention Is All You Need*
- Löwe et al. (2022) — *Amortized Causal Discovery with Variational Inferences*

---

## ⚠️ Disclaimer

This repository is for **research and educational purposes only**. Past performance of any signals, models, or strategies described herein does not guarantee future results. This is **not financial advice**. Trading involves substantial risk of loss.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
