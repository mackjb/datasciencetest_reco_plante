import pandas as pd
import os

excel_path = "Streamlit/assets/architectures/perfo_archi.xlsx"

try:
    df = pd.read_excel(excel_path)
    # Select columns
    cols = ['Archi', 'Espèce-Macro_F1', 'Espèce-Accuracy', 'Maladie- Accuracy']
    subset = df[df['Archi'].isin([3, 9])][cols]
    print(subset)
except Exception as e:
    print(e)
