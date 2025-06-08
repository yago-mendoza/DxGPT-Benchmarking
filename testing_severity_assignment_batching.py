from utils.llm import *
import json


def test_severity_assignment_batching():
    # Configurar LLM
    llm = Azure("gpt-4o", temperature=0.1)
    
    # Cargar el prompt y schema
    with open('bench/experiments/ramedis_baseline/eval-prompts/severity_assignment_batch_prompt.txt', 'r') as f:
        batch_prompt = f.read()
    
    with open('bench/experiments/ramedis_baseline/eval-prompts/severity_assignment_batch_schema.json', 'r') as f:
        batch_schema = json.load(f)
    
    # Datos de prueba
    batch = [
        "Common cold",
        "Pneumonia", 
        "Diabetes mellitus type 2",
        "Acute myocardial infarction",
        "Migraine",
        "Stage 4 pancreatic cancer"
    ]
    
    # Parámetros de generación
    generation_params = {
        "temperature": 0.1,
        "max_tokens": 1000
    }
    
    # Crear items con id y diagnosis
    items = [{"id": i+1, "diagnosis": diagnosis} for i, diagnosis in enumerate(batch)]
    
    print("Items: ############")
    print(items)
    print("############")
    
    # Hacer UNA SOLA llamada para todo el batch
    response = llm.generate(
        batch_prompt,
        batch_items=items,
        schema=batch_schema,
        **generation_params
    )
    
    print("\nResponse: ############")
    print(response)
    print("############")
    
    # Evaluar la respuesta
    try:
        if isinstance(response, str):
            response_data = json.loads(response)
        else:
            response_data = response
            
        # Manejar diferentes formatos de respuesta
        if isinstance(response_data, list):
            # Respuesta directa como lista
            severities = response_data
        elif isinstance(response_data, dict):
            # Respuesta con estructura {"severities": [...]}
            severities = response_data.get('severities', [])
        else:
            raise ValueError(f"Formato de respuesta inesperado: {type(response_data)}")
        
        print(f"\nEvaluación del batch:")
        print(f"- Diagnósticos enviados: {len(batch)}")
        print(f"- Severidades recibidas: {len(severities)}")
        
        if len(severities) == len(batch):
            print("✅ Cantidad correcta de respuestas")
            
            # Validar cada severidad
            for i, severity_item in enumerate(severities):
                diagnosis = severity_item.get('diagnosis', '')
                severity = severity_item.get('severity', '')
                expected_diagnosis = batch[i]
                
                print(f"\n{i+1}. {expected_diagnosis}")
                print(f"   Diagnóstico devuelto: {diagnosis}")
                print(f"   Severidad: {severity}")
                
                # Validar formato de severidad
                if severity.startswith('S') and len(severity) >= 2:
                    try:
                        sev_num = int(severity[1:])
                        if 0 <= sev_num <= 10:
                            print(f"   ✅ Formato de severidad válido")
                        else:
                            print(f"   ❌ Severidad fuera de rango: {severity}")
                    except ValueError:
                        print(f"   ❌ Formato de severidad inválido: {severity}")
                else:
                    print(f"   ❌ Formato de severidad inválido: {severity}")
        else:
            print("❌ Cantidad incorrecta de respuestas")
            
    except Exception as e:
        print(f"❌ Error al procesar la respuesta: {e}")
        print(f"Respuesta raw: {response}")


if __name__ == "__main__":
    test_severity_assignment_batching()