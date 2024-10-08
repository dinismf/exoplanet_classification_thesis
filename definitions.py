from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).resolve().parent

# Path to raw data, TCE table, and output directory for processed data
KEPLER_DATA_DIR = ROOT_DIR / 'data' / 'raw' / 'fits'
TCE_TABLE_DIR = ROOT_DIR / 'data' / 'raw' / 'q1_q17_dr24_tce.csv'
TCE_TABLE_DIR_TEST = ROOT_DIR / 'data' / 'raw' / 'q1_q17_dr24_tce_test.csv'
OUTPUT_DIR = ROOT_DIR / 'data' / 'processed'

MODELS_OUTPUT_DIR = ROOT_DIR / 'models' / 'trained_models'
