from latitude import Latitude
import os
from dotenv import load_dotenv

load_dotenv()
sdk = Latitude(api_key=os.getenv("LATITUDE_API_KEY"))

# Registers the Azure deployment as a Prompt in Latitude's terminology
def ensure_model(project_id: str):
    return sdk.models.create_or_retrieve(
        project_id=project_id,
        name=os.getenv("MODEL_NAME"),
        provider=os.getenv("AZURE_PROVIDER_NAME"),        # <-- provider alias from UI
        provider_model_id=os.getenv("AZURE_DEPLOYMENT_ID"),
        provider_args={                       # gets forwarded as headers/params
            "api-version": os.getenv("AZURE_API_VERSION")
        },
        prompt_template=os.getenv("PROMPT_TEMPLATE")
    )

if __name__ == "__main__":
    project = sdk.projects.get_by_name(os.getenv("PROJECT_NAME"))
    model   = ensure_model(project.id)
    print(f"Model ready ⇒ {model.id}")
