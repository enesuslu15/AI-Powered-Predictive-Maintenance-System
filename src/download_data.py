import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

def download_data():
    # Define folder path to save the dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    # Create folder if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    csv_path = os.path.join(data_dir, 'ai4i2020.csv')
    
    print("Downloading AI4I 2020 dataset from UCI Machine Learning Repository...")
    
    try:
        # fetch dataset (AI4I 2020 dataset id is 601)
        dataset = fetch_ucirepo(id=601) 
        
        # extract data
        X = dataset.data.features 
        y = dataset.data.targets 
        
        # Combine features and targets
        df = pd.concat([X, y], axis=1)
        
        # Save as CSV
        df.to_csv(csv_path, index=False)
        print(f"Success! Dataset downloaded and saved to {csv_path}.")
        print("-" * 30)
        print("Dataset Preview:")
        print(df.head())
        print("-" * 30)
        print("Dataset Info:")
        print(df.info())
        
    except Exception as e:
        print(f"An error occurred while downloading the data: {e}")

if __name__ == "__main__":
    download_data()
