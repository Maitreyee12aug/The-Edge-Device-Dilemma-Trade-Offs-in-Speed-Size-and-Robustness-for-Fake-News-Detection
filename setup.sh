#!/usr/bin/env bash
# ============================================================
# setup.sh — One-command environment setup
# ============================================================
set -e

echo "=========================================="
echo " Edge-Device Dilemma — Environment Setup"
echo "=========================================="

# 1. Create virtual environment
echo "[1/4] Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip
echo "[2/4] Upgrading pip..."
pip install --upgrade pip --quiet

# 3. Install dependencies
echo "[3/4] Installing dependencies from requirements.txt..."
pip install -r requirements.txt --quiet

# 4. Download NLTK data
echo "[4/4] Downloading NLTK resources..."
python3 -c "
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
print('NLTK resources downloaded.')
"

echo ""
echo "=========================================="
echo " Setup complete!"
echo " Activate your environment with:"
echo "   source .venv/bin/activate"
echo " Then run an experiment, e.g.:"
echo "   python scripts/01_phase1_efficiency.py"
echo "=========================================="
