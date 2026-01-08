
import pandas as pd
import io
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def parse_classification_report(file_path):
    """Parses a sklearn classification report text file into a DataFrame."""
    try:
        with open(file_path, "r") as f:
            report_text = f.read()
        
        # Split lines and filter empty ones
        lines = [line.strip() for line in report_text.split('\n') if line.strip()]
        
        # Parse data
        data = []
        for line in lines[1:]: # Skip header
             parts = line.split()
             if len(parts) >= 2: # Check for valid line
                # Handle class names with spaces or special chars if any, though report usually aligns well
                # The last 3 are metrics, 4th from last is support, rest is name
                if parts[0] in ['accuracy', 'macro', 'weighted']: # Skip summary rows for species table
                     continue
                
                name = parts[0]
                precision = float(parts[1])
                recall = float(parts[2])
                f1 = float(parts[3])
                support = int(parts[4])
                data.append([name, precision, recall, f1, support])
                
        df = pd.DataFrame(data, columns=["Classe", "Précision", "Rappel", "F1-score", "Support"])
        return df
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def render_df_as_image(df, title=None):
    """Renders a DataFrame as a matplotlib figure image."""
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(5, len(df) * 0.25 + 1)) # Adjust height based on rows
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center', colColours=['#f2f2f2']*len(df.columns))
        
        # Style
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5) # Scale width, height
        
        if title:
            plt.title(title, fontweight="bold")
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)
        print("Image rendered successfully")
        return buf
    except Exception as e:
        print(f"Error rendering image: {e}")
        return None

# Test paths
model_paths = {
    "SVM (RBF)": {
        "report": "results/Machine_Learning/svm_rbf_baseline_features_selected/evaluation/baseline/classification_report.txt",
        "plot": "results/Machine_Learning/svm_rbf_baseline_features_selected/plots/baseline/confusion_matrix.png"
    }
}

for model, paths in model_paths.items():
    print(f"Testing {model}...")
    if os.path.exists(paths["report"]):
        df = parse_classification_report(paths["report"])
        if df is not None:
            print("Report parsed.")
            print(df.head())
            df_display = df[["Classe", "F1-score"]].copy() 
            df_display["F1-score"] = df_display["F1-score"].apply(lambda x: f"{x:.2f}")
            render_df_as_image(df_display)
        else:
            print("Failed to parse report.")
    else:
        print(f"File not found: {paths['report']}")
