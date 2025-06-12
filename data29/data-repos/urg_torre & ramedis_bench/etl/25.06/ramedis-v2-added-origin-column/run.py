from pathlib import Path
from typing import Tuple, List, Dict
import pandas as pd
from utils.bert import calculate_semantic_similarity
from utils.icd10 import ICD10Taxonomy

csvs = [
    Path("./test_RAMEDIS.csv"),
    Path("./test_HMS.csv"),
    Path("./test_MME.csv"),
    Path("./test_LIRICAL.csv"),
    Path("./test_PUMCH_ADM.csv")
]

def combine_csvs() -> pd.DataFrame:
    """
    Combine all CSV files into a single DataFrame with an origin column and renumbered IDs.
    """
    combined_data = []
    
    # Mapping from file names to origin values
    origin_mapping = {
        "test_RAMEDIS.csv": "RAMEDIS",
        "test_HMS.csv": "HMS", 
        "test_MME.csv": "MME",
        "test_LIRICAL.csv": "LIRICAL",
        "test_PUMCH_ADM.csv": "PUMCH_ADM"
    }
    
    for csv_path in csvs:
        if csv_path.exists():
            # Read CSV file
            df = pd.read_csv(csv_path)
            
            # Add origin column
            origin_value = origin_mapping[csv_path.name]
            df['origin'] = origin_value
            
            # Append to combined data
            combined_data.append(df)
            print(f"Loaded {len(df)} rows from {csv_path.name} with origin '{origin_value}'")
        else:
            print(f"Warning: File {csv_path} does not exist, skipping...")
    
    if combined_data:
        # Combine all DataFrames
        final_df = pd.concat(combined_data, ignore_index=True)
        
        # Drop existing 'id' column if it exists
        if 'id' in final_df.columns:
            final_df = final_df.drop(columns=['id'])
        
        # Renumber IDs starting from 0
        final_df = final_df.reset_index(drop=True)
        final_df.insert(0, 'id', range(len(final_df)))
        
        print(f"Combined {len(final_df)} total rows from {len(combined_data)} files")
        return final_df
    else:
        print("No CSV files found to combine")
        return pd.DataFrame()

def main():
    """
    Main function to execute the CSV combination process.
    """
    print("Starting CSV combination process...")
    
    # Combine all CSVs
    combined_df = combine_csvs()
    
    if not combined_df.empty:
        # Save combined CSV
        output_path = Path("./after.csv")
        combined_df.to_csv(output_path, index=False)
        print(f"Combined CSV saved to: {output_path}")
        print(f"Final shape: {combined_df.shape}")
        print(f"Origins found: {combined_df['origin'].unique().tolist()}")
    else:
        print("No data to save")

if __name__ == "__main__":
    main()
