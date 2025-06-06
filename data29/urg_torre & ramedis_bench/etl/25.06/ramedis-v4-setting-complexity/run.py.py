#!/usr/bin/env python3
"""
Script para añadir etiquetas de complejidad a archivos CSV médicos
Procesa en lotes de 5 casos y guarda incrementalmente
"""

import sys
import pandas as pd
import json
from pathlib import Path
from utils.llm import AzureLLM
import csv

# Configuration
BATCH_SIZE = 5
MODEL = "gpt-4o"

# Initialize LLM
llm = AzureLLM(MODEL, temperature=0.2)

# Prompt optimizado para evaluación de complejidad
PROMPT_TEMPLATE = """Eres un evaluador experto de complejidad de casos médicos. 
Evalúa cada caso del P0 (mínima complejidad) al P10 (máxima complejidad).

Guía de complejidad:
- P0-P2: Casos simples y rutinarios
- P3-P4: Complejidad moderada 
- P5-P6: Complejidad significativa
- P7-P8: Alta complejidad
- P9-P10: Complejidad extrema

Analiza estos {num_cases} casos y responde con un objeto JSON que contenga un array "complexities":

{cases}

Formato esperado: {{"complexities": ["P3", "P7", "P2", "P5", "P8"]}}"""

def process_batch_cases(cases_batch):
    """Procesa un lote de casos y devuelve las etiquetas de complejidad"""
    # Formatear casos para el prompt
    cases_text = "\n\n---\n\n".join([
        f"Caso {i+1}:\n{case[:600]}"  # Limitar longitud para tokens
        for i, case in enumerate(cases_batch)
    ])
    
    prompt = PROMPT_TEMPLATE.format(
        num_cases=len(cases_batch),
        cases=cases_text
    )
    
    try:
        # Obtener respuesta con schema estructurado (debe ser objeto para Azure)
        response = llm.generate(
            prompt,
            schema={
                "type": "object",
                "properties": {
                    "complexities": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "pattern": "^P([0-9]|10)$"
                        },
                        "minItems": len(cases_batch),
                        "maxItems": len(cases_batch)
                    }
                },
                "required": ["complexities"],
                "additionalProperties": False
            }
        )
        
        # Extraer array de complejidades
        if isinstance(response, dict) and "complexities" in response:
            complexities = response["complexities"]
            if len(complexities) == len(cases_batch):
                return complexities
            else:
                raise ValueError(f"Se esperaban {len(cases_batch)} complejidades, se recibieron {len(complexities)}")
        else:
            raise ValueError("Respuesta no contiene 'complexities'")
            
    except Exception as e:
        print(f"Error procesando lote: {e}")
        # Devolver complejidades por defecto
        return [f"P5" for _ in cases_batch]

def process_csv_file(input_file):
    """Procesa un archivo CSV añadiendo columna de complejidad"""
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"Error: No se encuentra el archivo {input_file}")
        return False
    
    # Nombre del archivo de salida
    output_file = input_path.stem + "_with_complexity.csv"
    
    print(f"\nProcesando: {input_file}")
    print(f"Salida: {output_file}")
    
    # Leer el CSV original para obtener headers y detectar formato
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        first_row = next(reader)
    
    # Encontrar índice de la columna 'case'
    if 'case' not in headers:
        print("Error: No se encuentra la columna 'case' en el CSV")
        return False
    
    case_index = headers.index('case')
    
    # Crear nuevos headers con 'complexity' después de 'case'
    new_headers = headers[:case_index+1] + ['complexity'] + headers[case_index+1:]
    
    # Contar total de filas
    total_rows = sum(1 for line in open(input_file, 'r', encoding='utf-8')) - 1
    print(f"Total de casos a procesar: {total_rows}")
    
    # Abrir archivos para lectura y escritura
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=new_headers)
        
        # Escribir headers
        writer.writeheader()
        
        # Procesar en lotes
        batch = []
        batch_rows = []
        processed = 0
        
        for row in reader:
            batch.append(row['case'])
            batch_rows.append(row)
            
            # Cuando tenemos un lote completo o es el final del archivo
            if len(batch) == BATCH_SIZE or processed + len(batch) == total_rows:
                print(f"\nProcesando lote: casos {processed + 1}-{processed + len(batch)}...")
                
                # Obtener complejidades del LLM
                complexities = process_batch_cases(batch)
                
                # Escribir filas con complejidades
                for i, (row, complexity) in enumerate(zip(batch_rows, complexities)):
                    # Insertar complejidad después de 'case'
                    new_row = {}
                    for field in new_headers:
                        if field == 'complexity':
                            new_row[field] = complexity
                        else:
                            new_row[field] = row.get(field, '')
                    
                    writer.writerow(new_row)
                
                # Forzar escritura a disco
                outfile.flush()
                
                processed += len(batch)
                print(f"  ✓ Guardado: {processed}/{total_rows} casos completados")
                
                # Limpiar lotes
                batch = []
                batch_rows = []
    
    print(f"\n✓ Procesamiento completo!")
    print(f"Archivo guardado: {output_file}")
    
    # Mostrar distribución de complejidades
    df = pd.read_csv(output_file)
    print("\nDistribución de complejidades:")
    distribution = df['complexity'].value_counts().sort_index()
    for complexity, count in distribution.items():
        print(f"  {complexity}: {count} casos ({count/len(df)*100:.1f}%)")
    
    return True

def main():
    """Punto de entrada principal"""
    if len(sys.argv) < 2:
        print("Uso: python add_complexity_labels.py <archivo.csv>")
        print("Ejemplo: python add_complexity_labels.py ramebench.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    process_csv_file(input_file)

if __name__ == "__main__":
    main()