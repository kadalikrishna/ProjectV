"""Vercel serverless entrypoint for the HDI Predictor Flask app."""

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1] / "HDI_Predictor"
sys.path.insert(0, str(PROJECT_DIR))

from app import app  # noqa: E402
