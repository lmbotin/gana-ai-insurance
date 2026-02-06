# Gana — AI-Native Insurance Carrier

**CS224G Final Project**

**Team**
- Luis Botin — lmbotin@stanford.edu
- Jesus Santos — jsaranda@stanford.edu

---

## 1-Line Description
An AI-native insurance carrier delivering cheaper, faster, and more transparent insurance through end-to-end automation.

---

## Overview
Legacy insurance carriers rely on large, manual, and opaque processes across underwriting, policy management, and claims handling.
Gana explores how recent advances in AI enable a **super-lean insurance organization**, with humans in the loop only where judgment is truly required.

This project focuses on **Track & Trace / AI Operational Liability claims automation**, spanning:
- Multimodal First Notice of Loss (FNOL) for logistics incidents
- LLM-based extraction with explicit uncertainty handling
- Smart claims routing
- Fraud detection and risk scoring
- End-to-end claim state visibility

---

## Quick Start

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

**Option 1: Using uv (recommended)**
```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone <repo-url>
cd gana-ai-insurance
uv sync
```

**Option 2: Using pip**
```bash
git clone <repo-url>
cd gana-ai-insurance
pip install -e .
```

### Environment Setup

Create a `.env` file in the project root for API keys (optional - only needed for real LLM extraction):

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...    # For Claude
OPENAI_API_KEY=sk-...           # For OpenAI
```

### Running the CLI

```bash
# Using mock extractor (no API key needed - great for testing)
python -m src.fnol.cli --text "Shipment delayed 48 hours due to routing model failure at HUB-LAX-03" --pretty

# Using Claude for extraction
python -m src.fnol.cli --text "Package lost after AI misclassified priority" --llm-provider claude --pretty

# With claimant info
python -m src.fnol.cli --text "System outage caused $15,000 in damages" \
  --claimant-name "Acme Logistics" \
  --policy-number "POL-TT-12345" \
  --pretty

# Save output to file
python -m src.fnol.cli --text "Prediction failure led to misroute" --output claim.json
```

### Running Tests

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
pytest

# Run with coverage
pytest --cov=src
```

---

## Success Metrics
- Straight-Through Processing (STP) rate
- Claim resolution time
- Fraud precision / recall
- Human intervention rate

---

## Repository Structure
```
gana-ai-insurance/
├── src/
│   └── fnol/           # Core claim extraction pipeline
│       ├── schema.py       # Pydantic models (OperationalLiabilityClaim)
│       ├── pipeline.py     # Main orchestration
│       ├── text_extractor.py   # LLM-based extraction
│       ├── image_analyzer.py   # Document/log analysis
│       ├── fusion.py       # Multi-modal fusion
│       ├── checker.py      # Evidence & consistency checks
│       └── cli.py          # Command-line interface
├── data/
│   └── examples/       # Sample claim JSON files
├── tests/              # Unit tests
├── scripts/            # Pipeline execution and demos
├── docs/               # Architecture, proposal, metrics
└── pyproject.toml      # Dependencies (uv/pip compatible)
```

---

## Claim Schema

The system processes `OperationalLiabilityClaim` objects with:

- **Incident types**: misroute, delay, loss, data_error, prediction_failure, system_outage
- **Asset types**: shipment, package, container, ai_model, sensor, route, prediction, document
- **Provenance tracking**: Every extracted field includes source modality and confidence score
- **Uncertainty handling**: Explicit "unknown" values, confidence scores 0.0-1.0

See `data/examples/` for sample claim JSON files.

---

## Disclaimer
This project uses **synthetic or publicly available data only**.
No real personal or insurance data is used.
