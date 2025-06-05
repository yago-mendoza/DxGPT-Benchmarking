from pathlib import Path
from typing import Tuple, List, Dict
import pandas as pd
from utils.bert import calculate_semantic_similarity
from utils.icd10 import ICD10Taxonomy

csvs = [
    Path("./test_ramebench.csv"),
    Path("./test_HMS.csv"),
    Path("./test_MME.csv"),
    Path("./test_LIRICAL.csv"),
    Path("./test_PUMCH_ADM.csv")
]

# do not change the code above
# code below, properly implement the following:
# juntalos en un único CSV pero añadiendo una columna que ponga "origin"
# y en funcion de cual csv vengan pones de valor <ramedis_ramebench"> o <ramedis_HMS> etc, vale ?¿

def combine_csvs() -> pd.DataFrame:
    """
    Combine all CSV files into a single DataFrame with an origin column.
    """
    combined_data = []
    
    # Mapping from file names to origin values
    origin_mapping = {
        "test_ramebench.csv": "ramedis_ramebench",
        "test_HMS.csv": "ramedis_HMS", 
        "test_MME.csv": "ramedis_MME",
        "test_LIRICAL.csv": "ramedis_LIRICAL",
        "test_PUMCH_ADM.csv": "ramedis_PUMCH_ADM"
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
        output_path = Path("./combined_data.csv")
        combined_df.to_csv(output_path, index=False)
        print(f"Combined CSV saved to: {output_path}")
        print(f"Final shape: {combined_df.shape}")
        print(f"Origins found: {combined_df['origin'].unique().tolist()}")
    else:
        print("No data to save")

if __name__ == "__main__":
    main()

