from datasets import load_dataset
import pandas as pd

def load_data():
    dataset = load_dataset("mstz/heloc")["train"]
    df = pd.DataFrame(dataset)
    return df
