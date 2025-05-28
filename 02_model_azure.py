

from latitude import Latitude
import config as C

sdk = Latitude(api_key=C.LATITUDE_API_KEY)

# Registers the Azure deployment as a Prompt in Latitude’s terminology

def ensure_model(project_id: str):
    return sdk.models.create_or_retrieve(
        project_id=project_id,
        name=C.MODEL_NAME,
        provider=C.AZURE_PROVIDER_NAME,        # <-- provider alias from UI
        provider_model_id=C.AZURE_DEPLOYMENT_ID,
        provider_args={                       # gets forwarded as headers/params
            "api-version": C.AZURE_API_VERSION
        },
        prompt_template=C.PROMPT_TEMPLATE
    )

if __name__ == "__main__":
    project = sdk.projects.get_by_name(C.PROJECT_NAME)
    model   = ensure_model(project.id)
    print(f"Model ready ⇒ {model.id}")
