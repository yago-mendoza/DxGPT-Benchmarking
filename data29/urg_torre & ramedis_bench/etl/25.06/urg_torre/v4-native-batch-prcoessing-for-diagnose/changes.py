import pandas as pd
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import logging
import sys

# Asegúrate de que el módulo utils.llm sea accesible
sys.path.append(str(Path(__file__).resolve().parent.parent)) # Si 'utils' está en el directorio padre

try:
    from utils.llm import Azure
except ImportError:
    print("Error: No se pudo importar 'utils.llm.Azure'. Asegúrate de que el módulo está en la ruta correcta.")
    sys.exit(1)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('diagnosis_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DiagnosisBatchProcessor:
    def __init__(self,
                 input_file: str = "data.csv",
                 output_file: str = "data_processed.csv",
                 llm_batch_size: int = 100, # Número de items a enviar al LLM en una sola llamada
                 llm_model: str = "gpt-4o"):

        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.llm_batch_size = llm_batch_size
        
        try:
            # Temperatura un poco más alta puede ser necesaria si el modelo tiene que generalizar a través del lote
            self.llm = Azure(llm_model, temperature=0.1) 
            logger.info(f"LLM Azure con modelo '{llm_model}' inicializado.")
        except Exception as e:
            logger.error(f"Error al inicializar Azure LLM: {e}")
            logger.error("Asegúrate de que las variables de entorno están configuradas.")
            raise

        # Schema para que el LLM devuelva una lista de resultados, cada uno con ID y sus diagnósticos
        self.batch_diagnosis_schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "batch_results": {
                    "type": "array",
                    "description": "Una lista de resultados, uno por cada ítem de entrada procesado.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "item_id": {
                                "type": "string", # o integer si tus IDs son siempre numéricos
                                "description": "El identificador único del ítem de entrada original."
                            },
                            "extracted_diagnoses": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Lista de diagnósticos médicos concisos extraídos del texto para este ítem. "
                                    "Si no hay diagnósticos claros, la lista puede estar vacía."
                                )
                            }
                        },
                        "required": ["item_id", "extracted_diagnoses"]
                    }
                }
            },
            "required": ["batch_results"]
        }

        self.output_columns = [
            'id', 'case', 'diagnosis', 'diagnostic_code', 'death',
            'critical', 'pediatric', 'severity'
        ]
        self.input_columns_to_read = [
            'id', 'case', 'diagnosis', 'icd10_diagnosis', 'diagnostic_code',
            'death', 'critical', 'pediatric', 'severity'
        ]

    def _create_batch_extraction_prompt(self, batch_items: List[Dict[str, Any]]) -> str:
        """
        Crea el prompt para el LLM para extraer diagnósticos de un lote de items.
        Cada item en batch_items debe tener 'id' y 'text_to_process'.
        """
        prompt = """Eres un asistente experto en procesamiento de datos médicos.
Tu tarea es procesar un lote de textos de diagnóstico. Para CADA texto en el lote, debes:
1.  Extraer una lista de posibles diagnósticos o condiciones médicas.
2.  Devolver el resultado como parte de una estructura JSON general que coincida con el schema proporcionado.
    Cada resultado individual debe incluir el 'item_id' original y su lista de 'extracted_diagnoses'.

1. Maintain clear and concise diagnoses as they are
2. Translate medical jargon and codes into understandable diagnoses
3. Pay special attention to the beginning of the text, as it often contains the most relevant diagnoses
4. Formatting rules:
   - No parentheses, commas, or special syntax
   - Ignore line breaks and special characters between diagnoses
   - Use English language only
5. Precision guidelines:
   - Use exact medical terminology (e.g., "traumatism", "Acute nasopharyngitis" instead of "injury" or "common cold") [!]
   - Exclude vague terms like "unspecified", "other", "general", "nonspecific" [!!!]
   - Best are short concise diagnoses (many if possible)
   - Avoid redundant synonymous diagnoses (e.g., don't list both "external otitis" and "medial external otitis")
   - Do not discard any potentially relevant diagnosis unless it's clearly redundant
6. If no clear diagnoses can be extracted or the text is irrelevant, make an educated guess based on the available information

Examples:

Lesiones ampollosas en pie, sin datos de alarma en el momento actual Bullous disorder, unspecified  
>> Bullous lesions; Bullous disorder

Exantema inespecifico sin datos de alarma en el momento actual. Eritema infeccioso?, Rash and other nonspecific skin eruption  
>> Erythema infectiosum

CONTUSION GLUTEO DERECHO, Contusion of lower back and pelvis, initial encounter  
>> Gluteal contusion

Alteración de la agudeza visual a filiar. Desprendimiento vítreo?, Unspecified visual disturbance  
>> Vitreous detachment

AMIGDALITIS, Acute tonsillitis, unspecified  
>> Acute tonsillitis

Viriasis. Sin datos de alarma en el momento actual, Viral infection, unspecified  
>> Viral illness

Nasofaringitis Aguda Conjuntivitis Aguda, Acute nasopharyngitis [common cold]  
>> Acute nasopharyngitis; Acute conjunctivitis; Acute nasopharyngitis

Neumonia derecha sin datos de alarma en el momento actual, Pneumonia, unspecified organism  
>> Pneumonia

INFECCION URINARIA CRU ASOCIADO, Urinary tract infection, site not specified  
>> Urinary tract infection

Bronquiolitis Aguda / Otitis en mejoría, Acute bronchiolitis, unspecified  
>> Acute bronchiolitis

Exantema inespecifico sin datos de alarma en el momento actual. Eritema infeccioso?, Rash and other nonspecific skin eruption  
>> Erythema infectiosum

DESGARRO FIBRILAR GEMELO MEDIAL PIERNA D, Laceration of other muscle(s) and tendon(s) of posterior muscle group at lower leg level, right leg, initial encounter  
>> Medial gastrocnemius muscle tear

Dolor abdominal sin datos de alarma Litiasis ??, Unspecified abdominal pain  
>> Lithiasis

Traumatismo tobillo izq. a filiar?, Unspecified injury of left ankle, initial encounter  
>> Left ankle trauma

Irritación vaginal EPI ?, Other specified noninflammatory disorders of vagina  
>> Vaginal irritation

GONALGIA A ESTUDIO QUISTE DE BACKER?, Pain in unspecified knee  
>> Baker’s cyst; Knee pain

Faringitis Aguda, Acute pharyngitis, unspecified  
>> Acute pharyngitis

PARONIQUIA, Cellulitis of unspecified finger, Cellulitis and acute lymphangitis  
>> Paronychia; Cellulitis; Acute lymphangitis

Posible pielonefritis aguda Infección de vías respiratorias altas, Acute pyelonephritis  
>> Acute pyelonephritis

Dolor abdominal, con cuadro diarreico leve y sin datos de patología orgánica urgente en el momento actual   Ideación autolítica manifiesta en paciente con antecedentes psquiátricos, Unspecified abdominal pain  
>> Suicidal ideation; Abdominal pain

sn febril hiper reactividad bronquial catarro, Fever, unspecified  
>> Acute nasopharyngitis

A continuación, se presenta el lote de ítems a procesar. Cada ítem tiene un 'id' y un 'text'.
Debes proporcionar una respuesta JSON que contenga un campo 'batch_results', que es una lista de objetos,
donde cada objeto tiene 'item_id' y 'extracted_diagnoses'.

Lote de Textos de Entrada:
"""
        # Serializamos los ítems del lote como JSON para que el LLM los vea claramente estructurados.
        # Esto ayuda al modelo a entender que debe producir una salida estructurada similar.
        input_batch_json_representation = []
        for item in batch_items:
            input_batch_json_representation.append({
                "item_id": str(item['id']), # Asegurarse que el ID es string para el prompt y schema
                "text": item['text_to_process']
            })
        
        prompt += json.dumps(input_batch_json_representation, indent=2, ensure_ascii=False)
        prompt += "\n\nPor favor, genera la respuesta JSON estructurada según el schema."
        return prompt

    def _process_llm_batch(self, batch_df: pd.DataFrame) -> Dict[Any, str]:
        """
        Procesa un lote de filas del DataFrame con una ÚNICA llamada al LLM.
        Devuelve un diccionario mapeando ID de fila a su cadena de diagnóstico extraída.
        """
        items_to_process: List[Dict[str, Any]] = []
        for _, row in batch_df.iterrows():
            diag_text = str(row.get('diagnosis', '')).strip()
            icd_text = str(row.get('icd10_diagnosis', '')).strip()
            
            combined_text = f"{diag_text} | {icd_text}".strip()
            if combined_text == "|" or not combined_text:
                combined_text = "Información no disponible" 

            items_to_process.append({
                "id": str(row['id']), # Asegurar que el ID es string para el prompt
                "text_to_process": combined_text
            })

        if not items_to_process:
            return {}

        prompt = self._create_batch_extraction_prompt(items_to_process)
        
        # Estimar max_tokens. Es más difícil con lotes, pero una heurística podría ser:
        # (tokens por descripción de item * num_items) + tokens para la estructura JSON + tokens para prompt
        # ej: (50 tokens/item * num_items) + 200 para overhead JSON + 300 para instrucciones del prompt
        max_tokens_estimate = (len(items_to_process) * 70) + 500 
        # Asegúrate de que esto no exceda el límite máximo de tokens del modelo.
        # Para gpt-4o el contexto es grande, pero la salida también tiene límites.
        
        results_map: Dict[Any, str] = {item['id']: "Error: No procesado por LLM" for item in items_to_process}

        try:
            logger.debug(f"Prompt para lote (primeros 500 chars):\n{prompt[:500]}...")
            response_data = self.llm.generate(prompt, schema=self.batch_diagnosis_schema, max_tokens=min(max_tokens_estimate, 4000)) # Limitar a 4000 tokens de salida por si acaso
            
            if isinstance(response_data, dict) and "batch_results" in response_data:
                llm_batch_output = response_data["batch_results"]
                if isinstance(llm_batch_output, list):
                    for result_item in llm_batch_output:
                        if isinstance(result_item, dict) and "item_id" in result_item and "extracted_diagnoses" in result_item:
                            item_id = result_item["item_id"] # Este ID es string
                            diagnoses_list = result_item["extracted_diagnoses"]
                            
                            # Convertir la lista de diagnósticos en una cadena
                            formatted_dx_string = "; ".join(str(d).strip() for d in diagnoses_list if str(d).strip()) if diagnoses_list else "No se extrajeron diagnósticos"
                            results_map[item_id] = formatted_dx_string
                        else:
                            logger.warning(f"Ítem malformado en batch_results del LLM: {result_item}")
                else:
                    logger.warning(f"batch_results no es una lista en la respuesta del LLM: {llm_batch_output}")
            else:
                logger.warning(f"Respuesta inesperada o malformada del LLM para el lote: {response_data}")
                # Si la respuesta es solo texto, el LLM no siguió el schema.
                if isinstance(response_data, str):
                    logger.warning(f"LLM devolvió texto plano en lugar de JSON: {response_data[:300]}...")


        except Exception as e:
            logger.error(f"Error durante la llamada al LLM para un lote: {e}")
            logger.error(f"Prompt que causó el error (primeros 500 caracteres):\n{prompt[:500]}...")
            # Los IDs ya tienen un valor de error por defecto
        
        return results_map


    def run_processing(self):
        if not self.input_file.exists():
            logger.error(f"Archivo de entrada no encontrado: {self.input_file}")
            return

        try:
            df = pd.read_csv(self.input_file, usecols=lambda col: col in self.input_columns_to_read)
            logger.info(f"Cargadas {len(df)} filas desde {self.input_file} con columnas: {list(df.columns)}")
        except Exception as e:
            logger.error(f"Error al leer el CSV. Verifica que las columnas esperadas existen ({self.input_columns_to_read}): {e}")
            return

        total_rows = len(df)
        num_llm_batches = (total_rows + self.llm_batch_size - 1) // self.llm_batch_size
        logger.info(f"Total de filas a procesar: {total_rows}. Tamaño de lote LLM: {self.llm_batch_size}. Número de lotes LLM: {num_llm_batches}.")

        # Para el guardado incremental, necesitamos escribir el encabezado una vez.
        first_batch_written = False

        for i in range(num_llm_batches):
            start_idx = i * self.llm_batch_size
            end_idx = min((i + 1) * self.llm_batch_size, total_rows)
            llm_batch_df = df.iloc[start_idx:end_idx].copy() # Usar .copy() para evitar SettingWithCopyWarning

            logger.info(f"Procesando lote LLM {i + 1}/{num_llm_batches} (filas de CSV {start_idx + 1}-{end_idx})...")

            # Obtener el mapeo de {id_original (string): "dx1; dx2"} del LLM
            extracted_diagnoses_map = self._process_llm_batch(llm_batch_df)
            
            # Aplicar los diagnósticos extraídos al DataFrame del lote
            # Importante: los IDs en extracted_diagnoses_map son strings. El ID en llm_batch_df puede ser int.
            def get_dx(row_id):
                return extracted_diagnoses_map.get(str(row_id), "Error: ID no encontrado en respuesta LLM")

            llm_batch_df.loc[:, 'diagnosis'] = llm_batch_df['id'].apply(get_dx)
            
            # Seleccionar y reordenar columnas para el archivo de salida
            output_batch_df = llm_batch_df[self.output_columns]

            # Guardado incremental
            try:
                if not first_batch_written:
                    output_batch_df.to_csv(self.output_file, index=False, encoding='utf-8', mode='w', header=True)
                    first_batch_written = True
                    logger.info(f"Encabezado y primer lote ({len(output_batch_df)} filas) escritos en {self.output_file}")
                else:
                    output_batch_df.to_csv(self.output_file, index=False, encoding='utf-8', mode='a', header=False)
                    logger.info(f"Lote {i+1} ({len(output_batch_df)} filas) añadido a {self.output_file}")
            except Exception as e:
                logger.error(f"Error al escribir el lote {i+1} en {self.output_file}: {e}")
            
            if i < num_llm_batches - 1:
                logger.info(f"Pausa de 2 segundos antes del siguiente lote LLM...") # Un poco más de pausa para lotes más grandes
                time.sleep(2) 

        logger.info(f"Procesamiento completado. Todos los lotes procesados y guardados en {self.output_file}")

def main():
    logger.info("Iniciando el proceso de extracción de diagnósticos (versión con batch LLM y schema)...")

    processor = DiagnosisBatchProcessor(
        input_file="data.csv",
        output_file="data_processed.csv",
        llm_batch_size=60, # Número de ítems a enviar al LLM en una sola llamada. ¡Empieza PEQUEÑO (2-5)!
                           # Aumenta gradualmente si ves que funciona y el LLM lo maneja.
        llm_model="gpt-4o" 
    )
    
    try:
        processor.run_processing()
    except Exception as e:
        logger.critical(f"Ha ocurrido un error fatal durante el procesamiento: {e}", exc_info=True)

    logger.info("Proceso finalizado.")

if __name__ == "__main__":
    main()