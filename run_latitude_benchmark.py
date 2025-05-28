import os
import pandas as pd
from latitude import Latitude
from latitude.models.project import CreateProjectRequestBody
from latitude.models.dataset import CreateDatasetRequestBody
from latitude.models.test_case import CreateTestCaseRequestBody
from latitude.models.model import CreateModelRequestBody
from latitude.models.evaluation import CreateEvaluationRequestBody
from dotenv import load_dotenv

# Cargar variables de entorno (para LATITUDE_API_KEY)
load_dotenv()

# --- CONFIGURACIÓN ---
LATITUDE_API_KEY = os.getenv("LATITUDE_API_KEY")
if not LATITUDE_API_KEY:
    raise ValueError("LATITUDE_API_KEY no encontrada. Asegúrate de que está en tu .env o configurada como variable de entorno.")

CSV_FILE_PATH = "clinical_dataset.csv" # Nombre de tu archivo CSV

# Nombres para los artefactos en Latitude
PROJECT_NAME = "Benchmark Diagnostico Medico"
DATASET_NAME = "Casos Clinicos Mayo 2024"
MODEL_NAME = "GPT3.5 Turbo - Diagnostico 5 Opciones"
EVALUATION_NAME = "Evaluacion Inicial Mayo 2024"

# El prompt que quieres probar
PROMPT_TEMPLATE = "Dado este caso clinico {{CASO_CLINICO}} genera una lista JSON de 5 posibles diagnosticos."
# La variable {{CASO_CLINICO}} será reemplazada por el contenido de la columna 'case' de tu CSV.

# El modelo LLM que quieres usar (asegúrate que Latitude lo soporta o configúralo en la UI de Latitude)
LLM_PROVIDER_MODEL_ID = "gpt-3.5-turbo" # Ejemplo, puedes cambiarlo por otro

# --- INICIALIZAR CLIENTE LATITUDE ---
client = Latitude(api_key=LATITUDE_API_KEY)
print("Cliente Latitude inicializado.")

# --- 1. CREAR O RECUPERAR PROYECTO ---
try:
    project = client.projects.create_or_retrieve(
        name=PROJECT_NAME,
        body=CreateProjectRequestBody(name=PROJECT_NAME)
    )
    print(f"Proyecto '{project.name}' (ID: {project.id}) obtenido/creado.")
except Exception as e:
    print(f"Error creando/obteniendo el proyecto: {e}")
    exit()

# --- 2. LEER DATOS DEL CSV Y CREAR/RECUPERAR DATASET CON TEST CASES ---
try:
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"Leído {len(df)} filas del CSV: {CSV_FILE_PATH}")

    dataset = client.datasets.create_or_retrieve(
        project_id=project.id,
        name=DATASET_NAME,
        body=CreateDatasetRequestBody(name=DATASET_NAME, project_id=project.id)
    )
    print(f"Dataset '{dataset.name}' (ID: {dataset.id}) obtenido/creado.")

    # Subir test cases (si no existen ya por 'external_id')
    print("Subiendo test cases al dataset...")
    for index, row in df.iterrows():
        case_id = str(row['id']) # Usamos el 'id' del CSV como 'external_id' para evitar duplicados
        clinical_case_text = str(row['case'])
        golden_diagnosis_text = str(row['golden_diagnosis'])

        try:
            # El input debe ser un diccionario donde la clave coincida con la variable en tu PROMPT_TEMPLATE
            input_data = {"CASO_CLINICO": clinical_case_text}

            test_case_body = CreateTestCaseRequestBody(
                dataset_id=dataset.id,
                input=input_data,
                ground_truth=golden_diagnosis_text,
                external_id=case_id # Ayuda a evitar duplicados si corres el script múltiples veces
            )
            # Usamos create_or_update para manejar casos donde ya existe el external_id
            # Nota: El SDK actual podría no tener create_or_update para test_cases directamente.
            # Si da error, tendrías que listar los test_cases y verificar si existe por external_id.
            # Por simplicidad, intentaremos crear. Si ya existe por external_id, Latitude podría manejarlo o dar un error.
            # La forma más robusta es hacer un client.test_cases.list() y chequear, o usar try-except.

            # Para simplificar, asumimos que create es suficiente y Latitude podría ignorar duplicados por external_id
            # o que necesitas limpiar el dataset antes si quieres re-subir con cambios.
            # OJO: El SDK puede que no tenga un `create_or_retrieve` para test_cases directamente.
            # Una forma es intentar obtenerlo por external_id, y si no existe, crearlo.
            # Para este ejemplo, vamos a crearlo directamente. Si el external_id ya existe,
            # Latitude podría dar un error o manejarlo. Consulta la documentación para el comportamiento exacto.
            # La documentación del SDK indica que external_id debe ser único para un dataset.
            
            # Primero, intentamos obtener el test case por external_id
            existing_test_cases = client.test_cases.list(dataset_id=dataset.id, external_id=case_id)
            if existing_test_cases and existing_test_cases.data and len(existing_test_cases.data) > 0:
                print(f"Test case con external_id '{case_id}' ya existe. Omitiendo creación.")
            else:
                created_tc = client.test_cases.create(body=test_case_body)
                print(f"Test case '{case_id}' creado (ID: {created_tc.id}).")

        except Exception as e_tc:
            print(f"Error creando/actualizando test case con external_id '{case_id}': {e_tc}")
            # Podrías querer continuar con los siguientes o parar el script
            # continue

    print("Finalizada la subida de test cases.")

except Exception as e:
    print(f"Error procesando el dataset: {e}")
    exit()

# --- 3. CREAR O RECUPERAR MODELO ---
# Aquí defines el LLM que quieres usar y el prompt.
try:
    model = client.models.create_or_retrieve(
        project_id=project.id,
        name=MODEL_NAME,
        body=CreateModelRequestBody(
            project_id=project.id,
            name=MODEL_NAME,
            provider_model_id=LLM_PROVIDER_MODEL_ID, # ej: "gpt-3.5-turbo", "claude-2", etc.
            prompt_template=PROMPT_TEMPLATE
            # Puedes añadir más configuraciones como 'temperature', 'max_tokens' aquí si es necesario
            # model_parameters={"temperature": 0.7, "max_tokens": 500}
        )
    )
    print(f"Modelo '{model.name}' (ID: {model.id}) obtenido/creado.")
except Exception as e:
    print(f"Error creando/obteniendo el modelo: {e}")
    exit()

# --- 4. CREAR Y EJECUTAR EVALUACIÓN ---
# Esto le dice a Latitude que corra el 'Modelo' contra el 'Dataset'.
try:
    evaluation_body = CreateEvaluationRequestBody(
        project_id=project.id,
        dataset_id=dataset.id,
        models_id=[model.id], # Puedes pasar una lista de IDs de modelos
        name=EVALUATION_NAME
        # Aquí podrías añadir 'scorers' si quieres que Latitude calcule métricas automáticamente
        # scorers=[{"type": "exact_match", "name": "Exact Match Diagnosis"}]
    )
    evaluation = client.evaluations.create(body=evaluation_body)
    print(f"Evaluación '{evaluation.name}' (ID: {evaluation.id}) creada y ejecución iniciada.")
    print(f"Puedes ver el progreso y resultados en: https://app.latitude.so/projects/{project.id}/evaluations/{evaluation.id}")

    # Opcional: Esperar a que la evaluación termine (puede tomar tiempo)
    # import time
    # while True:
    #     eval_status = client.evaluations.get(evaluation_id=evaluation.id)
    #     print(f"Estado de la evaluación: {eval_status.status}")
    #     if eval_status.status in ["completed", "failed", "error"]:
    #         break
    #     time.sleep(30) # Esperar 30 segundos
    # print(f"Evaluación finalizada con estado: {eval_status.status}")
    # if eval_status.status == "completed":
    #     # Aquí podrías obtener los resultados de la evaluación si el SDK lo permite directamente
    #     # o procesarlos desde la UI / API de Latitude.
    #     print("Resultados de la evaluación listos.")

except Exception as e:
    print(f"Error creando la evaluación: {e}")
    exit()

print("\n--- Script finalizado ---")
print(f"Revisa tu proyecto en Latitude: https://app.latitude.so/projects/{project.id}")