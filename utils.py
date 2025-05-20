# ======= utils.py =======
import pandas as pd

def get_generic_name(medicine_name):
    df = pd.read_csv("dataset/Training/training_labels.csv")
    row = df[df['MEDICINE_NAME'].str.lower() == medicine_name.lower()]
    return row.iloc[0]['GENERIC_NAME'] if not row.empty else "Not found"

def get_uses(generic_name):
    df = pd.read_csv("dataset/Medicine_Details.csv")
    row = df[df['Medicine Name'].str.lower().str.contains(generic_name.lower())]
    return row.iloc[0]['Uses'] if not row.empty else "No info found"