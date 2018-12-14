import os

# Project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


# Path to raw data, TCE table and output directory for processed data
KEPLER_DATA_DIR = 'E:\\MSC_PROJECT_DATA\\new_fits\\'
TCE_TABLE_DIR = 'E:\\MSC_PROJECT_DATA\\new_fits\\q1_q17_dr24_tce.csv'
OUTPUT_DIR = os.path.join(ROOT_DIR, "data\\processed\\")

