# Creates the evaluation, uploads the prompt above and fires an experiment over the dataset

from latitude import Latitude
from pathlib import Path
import config as C

sdk = Latitude(api_key=C.LATITUDE_API_KEY)

def ensure_evaluation(project_id: str, model_id: str, dataset_id: str):
    prompt_text = Path("scorers/synonym_judge_prompt.txt").read_text()

    return sdk.evaluations.create_or_retrieve(
        project_id=project_id,
        name=C.EVAL_NAME,
        evaluation_type="LLM_JUDGE",          # ← doc § “LLM-as-Judges” 
        judge_provider=C.AZURE_PROVIDER_NAME, # you can also use a cheaper model
        judge_model="gpt-35-turbo",           # or gpt-4o-mini etc.
        judge_prompt=prompt_text,
        output_format="NUMERIC",
        min_score=0,
        max_score=5,
    )

def run_experiment(eval_id: str, model_id: str, dataset_id: str):
    sdk.evaluations.run(
        evaluation_id=eval_id,
        dataset_id=dataset_id,
        model_ids=[model_id]
    )

if __name__ == "__main__":
    project  = sdk.projects.get_by_name(C.PROJECT_NAME)
    dataset  = sdk.datasets.get_by_name(project.id, C.DATASET_NAME)
    model    = sdk.models.get_by_name(project.id, C.MODEL_NAME)
    eval_    = ensure_evaluation(project.id, model.id, dataset.id)
    run_experiment(eval_.id, model.id, dataset.id)
    print(f"Experiment launched – track it in the UI (Evaluations tab)")
