import pandas as pd
from latitude import Latitude
import os
from dotenv import load_dotenv

load_dotenv()

sdk = Latitude(api_key=os.getenv("LATITUDE_API_KEY"))

def ensure_project():
    return sdk.projects.create_or_retrieve(name=os.getenv("PROJECT_NAME"))

def ensure_dataset(project_id: str):
    df = pd.read_csv(os.getenv("CSV_FILE_PATH"))
    ds = sdk.datasets.create_or_retrieve(
        project_id=project_id,
        name=os.getenv("DATASET_NAME")
    )
    for _, row in df.iterrows():
        input_ = {"CASO_CLINICO": str(row["case"])}
        sdk.test_cases.create_or_update(
            dataset_id=ds.id,
            external_id=str(row["id"]),
            input=input_,
            ground_truth=str(row["golden_diagnosis"])
        )
    return ds

if __name__ == "__main__":
    project = ensure_project()
    dataset = ensure_dataset(project.id)
    print(f"Dataset ready ⇒ {dataset.id}")
