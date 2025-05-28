# All constants live in one place; the rest of the scripts just import config.

from dotenv import load_dotenv
import os
load_dotenv()        # reads .env

LATITUDE_API_KEY = os.getenv("LATITUDE_API_KEY")
if not LATITUDE_API_KEY:
    raise RuntimeError("Put LATITUDE_API_KEY in your .env")

# Azure OpenAI – the model your *prompt* will use
AZURE_PROVIDER_NAME   = "azure-prod"          # the alias you create in Latitude ▸ Settings ▸ Providers
AZURE_DEPLOYMENT_ID   = "gpt-35-turbo"        # whatever you called the model deployment
AZURE_API_VERSION     = "2024-04-01-preview"  # example

PROJECT_NAME          = "Benchmark Diagnóstico Médico"
DATASET_NAME          = "Casos Clínicos Mayo-2024"
MODEL_NAME            = "GPT-3.5 Turbo (Azure) – 5 opciones"
EVAL_NAME             = "Judge – Synonym rank"

CSV_FILE_PATH         = "clinical_dataset.csv"
PROMPT_TEMPLATE = (
    "Dado este caso clínico {{CASO_CLINICO}} "
    "genera un JSON con una lista llamada diagnoses y EXACTAMENTE "
    "5 diagnósticos probables (en español)."
)
