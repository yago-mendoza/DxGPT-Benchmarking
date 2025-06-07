#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ramedis Baseline Experiment Runner

Este script ejecuta el pipeline completo de evaluaci�n de LLMs para casos cl�nicos:
1. Genera diagn�sticos diferenciales (DDX) usando un LLM
2. Eval�a similitud sem�ntica entre DDX y diagn�sticos de referencia (GDX)
3. Asigna severidades a los DDX �nicos
4. Calcula scores de evaluaci�n de severidad
5. Genera visualizaciones de los resultados

Dise�ado para ser auto-contenido y espec�fico para este experimento.
"""

import json
import os
import sys
import yaml
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Importar utilidades del proyecto
from utils.llm import Azure
from utils.bert import calculate_semantic_similarity


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# 0. INITIAL SETUP
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

def setup_experiment_run() -> Tuple[Path, logging.Logger]:
    """
    Configura el directorio de resultados y el sistema de logging para la ejecuci�n.
    
    Returns:
        Tuple[Path, Logger]: Directorio de resultados y logger configurado
    """
    # Crear timestamp para esta ejecuci�n
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_name = f"run_{timestamp}"
    
    # Crear estructura de directorios
    base_dir = Path(__file__).parent
    results_dir = base_dir / "results" / run_name
    plots_dir = results_dir / "plots"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    
    # Configurar logging
    log_file = results_dir / "execution.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Copiar config.yaml a la carpeta de resultados
    config_source = base_dir / "config.yaml"
    config_dest = results_dir / "config.yaml"
    shutil.copy2(config_source, config_dest)
    
    logger.info(f"Experiment run initialized: {run_name}")
    logger.info(f"Results directory: {results_dir}")
    
    return results_dir, logger


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# 1. CONFIGURATION AND DATA LOADING
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Carga la configuraci�n del experimento desde el archivo YAML.
    
    Args:
        config_path: Ruta al archivo config.yaml
        
    Returns:
        Dict con la configuraci�n parseada
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_dataset(dataset_path: str, base_dir: Path) -> List[Dict[str, Any]]:
    """
    Carga el dataset de casos cl�nicos desde el archivo JSON.
    
    Args:
        dataset_path: Ruta relativa al dataset (desde bench/)
        base_dir: Directorio base del experimento
        
    Returns:
        Lista de casos cl�nicos con sus diagn�sticos de referencia
    """
    # Navegar desde experiments/ramedis_baseline hasta la ra�z del proyecto
    project_root = base_dir.parent.parent.parent
    full_path = project_root / dataset_path
    
    with open(full_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    return dataset


def create_llm_from_config(llm_config: Dict[str, Any], base_dir: Path) -> Tuple[Azure, str, Optional[Dict], Dict[str, Any]]:
    """
    Factory function para crear instancias de LLM desde la configuración.
    
    Args:
        llm_config: Configuración del LLM desde config.yaml
        base_dir: Directorio base del experimento
        
    Returns:
        Tuple[Azure, str, Optional[Dict], Dict]: Instancia LLM, template del prompt, esquema de salida, parámetros de generación
    """
    # Leer el prompt desde archivo
    prompt_path = base_dir / llm_config['prompt']
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    # Leer el esquema si existe
    schema = None
    if 'output_schema' in llm_config and llm_config['output_schema']:
        schema_path = base_dir / llm_config['output_schema']
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
    
    # Separar parámetros de configuración y de generación
    params = llm_config.get('params', {})
    config_params = {}
    generation_params = {}
    
    # Parámetros que van en la configuración del LLM
    if 'temperature' in params:
        config_params['temperature'] = params['temperature']
    
    # Parámetros que van en la llamada a generate()
    for key in ['max_tokens', 'top_p']:
        if key in params:
            generation_params[key] = params[key]
    
    # Crear instancia LLM
    llm = Azure(
        llm_config['model'],
        **config_params
    )
    
    return llm, prompt_template, schema, generation_params


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# 2. CANDIDATE DIAGNOSIS (DDX) GENERATION
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

# Constante del sistema: n�mero de DDX a generar
NUM_DDX_TO_GENERATE = 5

def generate_candidate_diagnoses(
    cases: List[Dict[str, Any]], 
    llm: Azure, 
    prompt_template: str,
    schema: Optional[Dict],
    generation_params: Dict[str, Any],
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """
    Genera diagn�sticos diferenciales candidatos para cada caso usando el LLM.
    
    Args:
        cases: Lista de casos cl�nicos
        llm: Instancia del LLM configurado
        prompt_template: Template del prompt con placeholder {case_description}
        schema: Esquema JSON para validar la respuesta
        logger: Logger para tracking
        
    Returns:
        Lista de respuestas con DDX por caso
    """
    responses = []
    
    logger.info(f"Generating DDX for {len(cases)} cases...")
    
    for i, case in enumerate(cases):
        logger.info(f"Processing case {i+1}/{len(cases)} (ID: {case['id']})")
        
        # Formatear el prompt con la descripci�n del caso
        prompt = prompt_template.replace("{case_description}", case["case"])
        
        try:
            # Generar respuesta del LLM
            if schema:
                response = llm.generate(prompt, schema=schema, **generation_params)
            else:
                response = llm.generate(prompt, **generation_params)
            
            # Extraer los diagn�sticos de la respuesta
            if isinstance(response, dict) and "diagnoses" in response:
                ddxs = response["diagnoses"][:NUM_DDX_TO_GENERATE]
            else:
                # Fallback si la respuesta no tiene el formato esperado
                logger.warning(f"Unexpected response format for case {case['id']}")
                ddxs = []
            
            # Asegurar que siempre tengamos exactamente 5 DDX
            while len(ddxs) < NUM_DDX_TO_GENERATE:
                ddxs.append(f"Unknown Diagnosis {len(ddxs) + 1}")
            
            responses.append({
                "case_id": case["id"],
                "ddxs": ddxs
            })
            
        except Exception as e:
            logger.error(f"Error generating DDX for case {case['id']}: {str(e)}")
            # Agregar respuesta vac�a en caso de error
            responses.append({
                "case_id": case["id"],
                "ddxs": [f"Error Diagnosis {j+1}" for j in range(NUM_DDX_TO_GENERATE)]
            })
    
    logger.info(f"DDX generation completed for all cases")
    return responses


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# 3. SEMANTIC EVALUATION (DDX vs. GDX)
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

# Batch size para procesamiento sem�ntico
SEMANTIC_BERT_BATCH_SIZE = 32

def calculate_semantic_scores_for_case(
    case_ddxs: List[str], 
    case_gdxs: List[str]
) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    """
    Calcula scores sem�nticos entre DDXs y GDXs de un caso.
    
    Args:
        case_ddxs: Lista de diagn�sticos diferenciales generados
        case_gdxs: Lista de diagn�sticos de referencia (golden standard)
        
    Returns:
        Tuple con scores por DDX y mejor match del caso
    """
    # Calcular similitud usando utils.bert
    result = calculate_semantic_similarity(case_ddxs, case_gdxs)
    
    # Formatear resultado para el JSON de salida
    ddx_semantic_scores = {}
    max_score = 0
    best_match = None
    
    for ddx in case_ddxs:
        scores = []
        for gdx in case_gdxs:
            score = result[ddx][gdx]
            scores.append(score if score is not None else 0.0)
            
            # Rastrear mejor match
            if score and score > max_score:
                max_score = score
                best_match = {"gdx": gdx, "ddx": ddx, "score": score}
        
        ddx_semantic_scores[ddx] = scores
    
    # Si no hay mejor match, crear uno por defecto
    if best_match is None:
        best_match = {"gdx": case_gdxs[0] if case_gdxs else "", "ddx": case_ddxs[0], "score": 0.0}
    
    return ddx_semantic_scores, best_match


def process_semantic_evaluation_parallel(
    cases: List[Dict[str, Any]], 
    candidate_responses: List[Dict[str, Any]],
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """
    Procesa evaluaci�n sem�ntica en paralelo para m�ltiples casos.
    
    Args:
        cases: Lista de casos con sus GDX
        candidate_responses: Respuestas con DDX generados
        logger: Logger para tracking
        
    Returns:
        Lista de evaluaciones sem�nticas por caso
    """
    # Crear mapeo de case_id a respuestas
    response_map = {resp["case_id"]: resp["ddxs"] for resp in candidate_responses}
    
    # Preparar datos para procesamiento
    cases_data = []
    for case in cases:
        case_id = case["id"]
        if case_id in response_map:
            gdx_names = [diag["name"] for diag in case.get("diagnoses", [])]
            cases_data.append({
                "id": case_id,
                "ddxs": response_map[case_id],
                "gdxs": gdx_names
            })
    
    results = []
    logger.info(f"Starting semantic evaluation for {len(cases_data)} cases...")
    
    with ThreadPoolExecutor(max_workers=min(SEMANTIC_BERT_BATCH_SIZE, len(cases_data))) as executor:
        # Enviar trabajos
        future_to_case = {
            executor.submit(
                calculate_semantic_scores_for_case,
                case_data['ddxs'],
                case_data['gdxs']
            ): case_data
            for case_data in cases_data
        }
        
        # Recolectar resultados
        for future in as_completed(future_to_case):
            case_data = future_to_case[future]
            try:
                ddx_scores, best_match = future.result()
                
                results.append({
                    "case_id": case_data['id'],
                    "gdx_set": case_data['gdxs'],
                    "best_match": best_match,
                    "ddx_semantic_scores": ddx_scores
                })
                
            except Exception as e:
                logger.error(f"Error in semantic evaluation for case {case_data['id']}: {str(e)}")
                # Agregar resultado vac�o en caso de error
                results.append({
                    "case_id": case_data['id'],
                    "gdx_set": case_data['gdxs'],
                    "best_match": {"gdx": "", "ddx": "", "score": 0.0},
                    "ddx_semantic_scores": {ddx: [0.0] * len(case_data['gdxs']) for ddx in case_data['ddxs']}
                })
    
    logger.info("Semantic evaluation completed")
    return results


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# 4. SEVERITY ASSIGNMENT TO UNIQUE DDX
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

# Batch size para asignaci�n de severidad
SEVERITY_LLM_BATCH_SIZE = 100

def assign_severities_batch(
    llm: Azure, 
    prompt_template: str, 
    unique_ddxs: List[str], 
    schema: Optional[Dict],
    generation_params: Dict[str, Any],
    logger: logging.Logger
) -> List[Dict[str, str]]:
    """
    Asigna severidades a DDX �nicos usando procesamiento por lotes.
    
    Args:
        llm: Instancia del LLM para asignaci�n de severidad
        prompt_template: Template del prompt con placeholder {diagnosis}
        unique_ddxs: Lista de diagn�sticos �nicos
        schema: Esquema JSON para validar respuesta
        logger: Logger para tracking
        
    Returns:
        Lista de diagn�sticos con sus severidades asignadas
    """
    results = []
    
    logger.info(f"Assigning severities to {len(unique_ddxs)} unique diagnoses...")
    
    # Procesar en batches
    for i in range(0, len(unique_ddxs), SEVERITY_LLM_BATCH_SIZE):
        batch = unique_ddxs[i:i+SEVERITY_LLM_BATCH_SIZE]
        logger.info(f"Processing severity batch {i//SEVERITY_LLM_BATCH_SIZE + 1}")
        
        for diagnosis in batch:
            # Formatear el prompt con el diagn�stico
            prompt = prompt_template.replace("{diagnosis}", diagnosis)
            
            try:
                # Generar respuesta del LLM
                if schema:
                    response = llm.generate(prompt, schema=schema, **generation_params)
                else:
                    response = llm.generate(prompt, **generation_params)
                
                # Extraer severidad de la respuesta
                if isinstance(response, dict) and "severity" in response:
                    severity = response["severity"]
                else:
                    logger.warning(f"Unexpected response format for diagnosis: {diagnosis}")
                    severity = "S5"  # Default medio si hay error
                
                results.append({
                    "ddx_unique_name": diagnosis,
                    "inferred_severity": severity
                })
                
            except Exception as e:
                logger.error(f"Error assigning severity to {diagnosis}: {str(e)}")
                results.append({
                    "ddx_unique_name": diagnosis,
                    "inferred_severity": "S5"  # Default medio en caso de error
                })
    
    logger.info("Severity assignment completed")
    return results


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# 5. CASE-LEVEL SEVERITY EVALUATION
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

# Mapeo de etiquetas de severidad a valores num�ricos
SEVERITY_MAPPING = {f"S{i}": i for i in range(11)}  # S0->0, S1->1, ..., S10->10
MAX_SEVERITY_VALUE = 10
MIN_SEVERITY_VALUE = 0
MAX_SEVERITY_DIFF_SQUARED = (MAX_SEVERITY_VALUE - MIN_SEVERITY_VALUE) ** 2

def evaluate_case_severity(
    case_id: str,
    ddxs: List[str],
    gdxs: List[Dict[str, Any]],
    severity_assignments: Dict[str, str],
    semantic_scores: Dict[str, List[float]],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Eval�a el score de severidad para un caso espec�fico.
    
    Args:
        case_id: ID del caso
        ddxs: Lista de DDX generados para el caso
        gdxs: Lista de GDX con sus severidades
        severity_assignments: Mapeo de DDX a severidades asignadas
        semantic_scores: Scores sem�nticos DDX-GDX del caso
        logger: Logger para debugging
        
    Returns:
        Dict con score de severidad final y penalizaci�n promedio
    """
    penalties = []
    
    for ddx_idx, ddx in enumerate(ddxs):
        # Obtener severidad num�rica del DDX
        ddx_severity_label = severity_assignments.get(ddx, "S5")  # Default S5 si no se encuentra
        sev_ddx = SEVERITY_MAPPING[ddx_severity_label]
        
        # Encontrar el GDX con mayor similitud sem�ntica
        ddx_scores = semantic_scores.get(ddx, [0.0] * len(gdxs))
        if not ddx_scores or not gdxs:
            penalties.append(MAX_SEVERITY_DIFF_SQUARED)
            continue
        
        best_gdx_idx = np.argmax(ddx_scores)
        best_gdx = gdxs[best_gdx_idx]
        
        # Obtener severidad num�rica del GDX
        gdx_severity_label = best_gdx.get("severity", "S5")
        sev_gdx = SEVERITY_MAPPING.get(gdx_severity_label, 5)
        
        # Calcular penalizaci�n cuadr�tica
        penalty = (sev_ddx - sev_gdx) ** 2
        penalties.append(penalty)
    
    # Calcular penalizaci�n promedio
    avg_penalty = sum(penalties) / len(penalties) if penalties else 0
    
    # Calcular score final (0-1, donde 1 es mejor)
    final_score = 1 - (avg_penalty / MAX_SEVERITY_DIFF_SQUARED)
    final_score = max(0, min(1, final_score))  # Acotar entre 0 y 1
    
    return {
        "case_id": case_id,
        "final_severity_score": round(final_score, 4),
        "avg_penalty_raw": round(avg_penalty, 2)
    }


def process_severity_evaluation(
    cases: List[Dict[str, Any]],
    candidate_responses: List[Dict[str, Any]],
    semantic_evaluations: List[Dict[str, Any]],
    severity_assignments: List[Dict[str, str]],
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """
    Procesa la evaluaci�n de severidad para todos los casos.
    
    Args:
        cases: Lista de casos con sus GDX
        candidate_responses: Respuestas con DDX generados
        semantic_evaluations: Evaluaciones sem�nticas por caso
        severity_assignments: Asignaciones de severidad para DDX �nicos
        logger: Logger para tracking
        
    Returns:
        Lista de evaluaciones de severidad por caso
    """
    # Crear mapeos para acceso r�pido
    response_map = {resp["case_id"]: resp["ddxs"] for resp in candidate_responses}
    semantic_map = {eval["case_id"]: eval for eval in semantic_evaluations}
    severity_map = {assign["ddx_unique_name"]: assign["inferred_severity"] 
                    for assign in severity_assignments}
    
    results = []
    logger.info(f"Evaluating severity scores for {len(cases)} cases...")
    
    for case in cases:
        case_id = case["id"]
        
        if case_id not in response_map or case_id not in semantic_map:
            logger.warning(f"Missing data for case {case_id}")
            continue
        
        ddxs = response_map[case_id]
        gdxs = case.get("diagnoses", [])
        semantic_eval = semantic_map[case_id]
        
        result = evaluate_case_severity(
            case_id=case_id,
            ddxs=ddxs,
            gdxs=gdxs,
            severity_assignments=severity_map,
            semantic_scores=semantic_eval["ddx_semantic_scores"],
            logger=logger
        )
        
        results.append(result)
    
    logger.info("Severity evaluation completed")
    return results


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# 6. VISUALIZATION GENERATION
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

def generate_visualizations(
    semantic_evaluations: List[Dict[str, Any]],
    severity_evaluations: List[Dict[str, Any]],
    plotting_config: Dict[str, bool],
    plots_dir: Path,
    logger: logging.Logger
):
    """
    Genera las visualizaciones configuradas en config.yaml.
    
    Args:
        semantic_evaluations: Evaluaciones sem�nticas por caso
        severity_evaluations: Evaluaciones de severidad por caso
        plotting_config: Configuraci�n de qu� gr�ficos generar
        plots_dir: Directorio donde guardar los gr�ficos
        logger: Logger para tracking
    """
    # Extraer datos para visualizaci�n
    semantic_scores = [eval["best_match"]["score"] for eval in semantic_evaluations]
    
    # Crear mapeo de case_id a severity score
    severity_map = {eval["case_id"]: eval["final_severity_score"] for eval in severity_evaluations}
    severity_scores = [severity_map.get(eval["case_id"], 0) for eval in semantic_evaluations]
    
    # Configurar estilo de matplotlib
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Scatter Plot Detallado: Todos los casos
    if plotting_config.get("plot_semantic_vs_severity_all_cases", False):
        plt.figure(figsize=(10, 8))
        plt.scatter(severity_scores, semantic_scores, alpha=0.6, s=100)
        plt.xlabel("Final Severity Score", fontsize=12)
        plt.ylabel("Best Semantic Match Score", fontsize=12)
        plt.title("Semantic vs Severity Scores - All Cases", fontsize=14)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.3)
        
        # Agregar l�nea de tendencia
        z = np.polyfit(severity_scores, semantic_scores, 1)
        p = np.poly1d(z)
        plt.plot([0, 1], p([0, 1]), "r--", alpha=0.8, label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / "semantic_vs_severity_all_cases.png", dpi=300)
        plt.close()
        logger.info("Generated: semantic_vs_severity_all_cases.png")
    
    # 2. Scatter Plot Agregado: Promedio
    if plotting_config.get("plot_semantic_vs_severity_average", False):
        avg_semantic = np.mean(semantic_scores)
        avg_severity = np.mean(severity_scores)
        
        plt.figure(figsize=(8, 8))
        plt.scatter([avg_severity], [avg_semantic], s=200, c='red', marker='D')
        plt.xlabel("Average Final Severity Score", fontsize=12)
        plt.ylabel("Average Best Semantic Match Score", fontsize=12)
        plt.title("Average Semantic vs Severity Scores", fontsize=14)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        # Agregar anotaci�n con valores
        plt.annotate(f"({avg_severity:.3f}, {avg_semantic:.3f})", 
                    xy=(avg_severity, avg_semantic), 
                    xytext=(avg_severity + 0.05, avg_semantic + 0.05),
                    fontsize=10)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "semantic_vs_severity_average.png", dpi=300)
        plt.close()
        logger.info("Generated: semantic_vs_severity_average.png")
    
    # 3. Histograma de distribuci�n sem�ntica
    if plotting_config.get("plot_semantic_distribution", False):
        plt.figure(figsize=(10, 6))
        plt.hist(semantic_scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel("Best Semantic Match Score", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title("Distribution of Semantic Similarity Scores", fontsize=14)
        plt.axvline(np.mean(semantic_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(semantic_scores):.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "semantic_distribution.png", dpi=300)
        plt.close()
        logger.info("Generated: semantic_distribution.png")
    
    # 4. Histograma de distribuci�n de severidad
    if plotting_config.get("plot_severity_distribution", False):
        plt.figure(figsize=(10, 6))
        plt.hist(severity_scores, bins=20, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel("Final Severity Score", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title("Distribution of Severity Evaluation Scores", fontsize=14)
        plt.axvline(np.mean(severity_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(severity_scores):.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "severity_distribution.png", dpi=300)
        plt.close()
        logger.info("Generated: severity_distribution.png")


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# MAIN EXECUTION PIPELINE
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

def main():
    """
    Ejecuta el pipeline completo del experimento Ramedis Baseline.
    """
    # 0. Initial Setup
    results_dir, logger = setup_experiment_run()
    base_dir = Path(__file__).parent
    
    try:
        # 1. Configuration and Data Loading
        logger.info("=" * 80)
        logger.info("STEP 1: Configuration and Data Loading")
        logger.info("=" * 80)
        
        config = load_config(base_dir / "config.yaml")
        dataset = load_dataset(config["dataset_path"], base_dir)
        logger.info(f"Loaded {len(dataset)} cases from dataset")
        
        # Crear instancias de LLM
        candidate_llm, candidate_prompt, candidate_schema, candidate_gen_params = create_llm_from_config(
            config["llm_configs"]["candidate_dx_gpt"], base_dir
        )
        severity_llm, severity_prompt, severity_schema, severity_gen_params = create_llm_from_config(
            config["llm_configs"]["severity_assigner_llm"], base_dir
        )
        logger.info("LLM instances created successfully")
        
        # 2. Candidate Diagnosis (DDX) Generation
        logger.info("=" * 80)
        logger.info("STEP 2: Candidate Diagnosis Generation")
        logger.info("=" * 80)
        
        candidate_responses = generate_candidate_diagnoses(
            dataset, candidate_llm, candidate_prompt, candidate_schema, candidate_gen_params, logger
        )
        
        # Guardar respuestas candidatas
        candidate_output = {
            "metadata": {
                "experiment_name": config["experiment_name"],
                "llm_config_alias": "candidate_dx_gpt",
                "model_used": config["llm_configs"]["candidate_dx_gpt"]["model"],
                "prompt_path": f"experiments/ramedis_baseline/{config['llm_configs']['candidate_dx_gpt']['prompt']}",
                "dataset_path": config["dataset_path"],
                "timestamp": datetime.now().isoformat()
            },
            "responses": candidate_responses
        }
        
        with open(results_dir / "candidate_responses.json", 'w', encoding='utf-8') as f:
            json.dump(candidate_output, f, indent=2, ensure_ascii=False)
        logger.info("Saved candidate_responses.json")
        
        # 3. Semantic Evaluation
        logger.info("=" * 80)
        logger.info("STEP 3: Semantic Evaluation")
        logger.info("=" * 80)
        
        semantic_evaluations = process_semantic_evaluation_parallel(
            dataset, candidate_responses, logger
        )
        
        # Guardar evaluaciones sem�nticas
        semantic_output = {
            "metadata": {
                "experiment_name": config["experiment_name"],
                "timestamp": datetime.now().isoformat()
            },
            "evaluations": semantic_evaluations
        }
        
        with open(results_dir / "semantic_evaluation.json", 'w', encoding='utf-8') as f:
            json.dump(semantic_output, f, indent=2, ensure_ascii=False)
        logger.info("Saved semantic_evaluation.json")
        
        # 4. Severity Assignment to Unique DDX
        logger.info("=" * 80)
        logger.info("STEP 4: Severity Assignment")
        logger.info("=" * 80)
        
        # Recopilar DDX �nicos
        unique_ddxs = set()
        for response in candidate_responses:
            unique_ddxs.update(response["ddxs"])
        unique_ddxs = list(unique_ddxs)
        logger.info(f"Found {len(unique_ddxs)} unique diagnoses")
        
        severity_assignments = assign_severities_batch(
            severity_llm, severity_prompt, unique_ddxs, severity_schema, severity_gen_params, logger
        )
        
        # Guardar asignaciones de severidad
        severity_assignment_output = {
            "metadata": {
                "experiment_name": config["experiment_name"],
                "llm_config_alias": "severity_assigner_llm",
                "model_used": config["llm_configs"]["severity_assigner_llm"]["model"],
                "prompt_path": f"experiments/ramedis_baseline/{config['llm_configs']['severity_assigner_llm']['prompt']}",
                "timestamp": datetime.now().isoformat()
            },
            "assigned_severities": severity_assignments
        }
        
        with open(results_dir / "severity_assignments.json", 'w', encoding='utf-8') as f:
            json.dump(severity_assignment_output, f, indent=2, ensure_ascii=False)
        logger.info("Saved severity_assignments.json")
        
        # 5. Case-Level Severity Evaluation
        logger.info("=" * 80)
        logger.info("STEP 5: Severity Evaluation")
        logger.info("=" * 80)
        
        severity_evaluations = process_severity_evaluation(
            dataset, candidate_responses, semantic_evaluations, 
            severity_assignments, logger
        )
        
        # Guardar evaluaciones de severidad
        severity_eval_output = {
            "metadata": {
                "experiment_name": config["experiment_name"],
                "timestamp": datetime.now().isoformat()
            },
            "evaluations": severity_evaluations
        }
        
        with open(results_dir / "severity_evaluation.json", 'w', encoding='utf-8') as f:
            json.dump(severity_eval_output, f, indent=2, ensure_ascii=False)
        logger.info("Saved severity_evaluation.json")
        
        # 6. Visualization Generation
        logger.info("=" * 80)
        logger.info("STEP 6: Visualization Generation")
        logger.info("=" * 80)
        
        generate_visualizations(
            semantic_evaluations, severity_evaluations, 
            config.get("plotting", {}), results_dir / "plots", logger
        )
        
        logger.info("=" * 80)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info(f"Results saved in: {results_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"FATAL ERROR: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()