
import pandas as pd

def load_data(path):
    df = pd.read_csv(path, encoding="latin1")
    
    # Keep only the relevant columns and rename them
    df = df[["v1", "v2"]]
    df.columns = ["label", "text"]
    
    return df
