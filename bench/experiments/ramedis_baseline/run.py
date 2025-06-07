import json
import asyncio
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import yaml
import shutil
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import numpy as np

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from config_parser.llm_factory import create_llms_from_config, get_llm_prompt, get_llm_schema
from utils.bert import calculate_semantic_similarity


# Constants for severity evaluation
UNDERDIAGNOSIS_WEIGHT = 2.0
OVERDIAGNOSIS_WEIGHT = 1.5


def setup_logging(log_path: Path):
    """Setup logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load dataset from JSON file."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both formats: array directly or object with 'cases' key
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'cases' in data:
        return data['cases']
    else:
        raise ValueError(f"Unexpected dataset format. Expected list or dict with 'cases' key, got {type(data)}")


def calculate_severity_score(ddx_severities: List[str], gdx_severities: List[str]) -> float:
    """
    Calculate severity score with asymmetric penalization.
    
    Args:
        ddx_severities: List of severities from differential diagnosis (e.g., ["S3", "S5"])
        gdx_severities: List of severities from golden diagnoses (e.g., ["S4", "S7"])
    
    Returns:
        Score between 0 and 1, where 1 is perfect match
    """
    if not ddx_severities or not gdx_severities:
        return 0.0
    
    penalties = []
    
    for ddx_sev in ddx_severities:
        ddx_num = int(ddx_sev[1:])  # Extract number from "S3" -> 3
        
        for gdx_sev in gdx_severities:
            gdx_num = int(gdx_sev[1:])
            diff = ddx_num - gdx_num
            
            if diff < 0:  # Underdiagnosis
                penalty = abs(diff) * UNDERDIAGNOSIS_WEIGHT
            else:  # Overdiagnosis
                penalty = diff * OVERDIAGNOSIS_WEIGHT
            
            penalties.append(penalty)
    
    # Take the minimum penalty (best match)
    min_penalty = min(penalties) if penalties else 10
    
    # Convert to score 0-1
    score = 1 - (min_penalty / 10)
    return max(0, min(1, score))


def calculate_semantic_score_with_details(
    ddx_list: List[str], 
    gdx_list: List[str]
) -> Tuple[float, Dict[str, List[float]]]:
    """
    Calculate semantic similarity score with detailed results.
    
    Returns:
        Tuple of (max_score, scores_detail)
    """
    if not ddx_list or not gdx_list:
        return 0.0, {}
    
    # Get similarities for all combinations
    similarities = calculate_semantic_similarity(ddx_list, gdx_list)
    
    max_score = 0.0
    scores_detail = {}
    
    for ddx in ddx_list:
        ddx_scores = []
        for gdx in gdx_list:
            score = similarities.get(ddx, {}).get(gdx, 0.0)
            if score is None:
                score = 0.0
            ddx_scores.append(score)
            
            # Update max score
            if score > max_score:
                max_score = score
            
            # Early stopping if perfect match
            if score >= 1.0:
                scores_detail[ddx] = ddx_scores
                return 1.0, scores_detail
        
        scores_detail[ddx] = ddx_scores
    
    return max_score, scores_detail


async def generate_candidate_responses(
    cases: List[Dict[str, Any]], 
    candidate_llm, 
    prompt_template: str,
    schema: Dict[str, Any],
    results_dir: Path
) -> Dict[str, Any]:
    """Generate differential diagnoses for all cases."""
    logging.info("Generating candidate diagnoses...")
    
    responses = []
    
    for i, case in enumerate(cases):
        logging.info(f"Processing case {i+1}/{len(cases)}: {case['id']}")
        
        # Format prompt with case description
        prompt = prompt_template.format(case=case['case'])
        
        try:
            # Get runtime parameters from LLM config
            runtime_params = candidate_llm._experiment_config.get('runtime_params', {})
            
            # Generate response
            response = candidate_llm.generate(
                prompt,
                schema=schema,
                **runtime_params  # Pass runtime params like max_tokens
            )
            
            # Extract diagnoses list
            if isinstance(response, dict) and 'diagnoses' in response:
                ddx = response['diagnoses']
            else:
                logging.error(f"Invalid response format for case {case['id']}")
                ddx = []
            
            responses.append({
                "case_id": case['id'],
                "ddx": ddx
            })
            
        except Exception as e:
            logging.error(f"Error processing case {case['id']}: {e}")
            responses.append({
                "case_id": case['id'],
                "ddx": []
            })
    
    # Save responses
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": candidate_llm.config.deployment_name,
            "dataset": "ramedis-45.json",
            "prompt": "candidate_prompt.txt"
        },
        "responses": responses
    }
    
    output_path = results_dir / "candidate_responses.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Saved candidate responses to {output_path}")
    return output


def evaluate_semantic_similarity(
    cases: List[Dict[str, Any]],
    candidate_responses: Dict[str, Any],
    results_dir: Path
) -> Dict[str, Any]:
    """Evaluate semantic similarity between DDX and GDX."""
    logging.info("Evaluating semantic similarity...")
    
    evaluations = []
    all_scores = []
    
    # Create mapping of case_id to responses
    response_map = {r['case_id']: r['ddx'] for r in candidate_responses['responses']}
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        
        for case in cases:
            case_id = case['id']
            ddx = response_map.get(case_id, [])
            gdx = [diag['name'] for diag in case.get('diagnoses', [])]
            
            future = executor.submit(
                calculate_semantic_score_with_details,
                ddx,
                gdx
            )
            futures.append((case_id, ddx, gdx, future))
        
        # Collect results
        for case_id, ddx, gdx, future in futures:
            score, scores_detail = future.result()
            
            evaluations.append({
                "case_id": case_id,
                "score": score,
                "gdx": gdx,
                "ddx_scores": scores_detail
            })
            all_scores.append(score)
            
            logging.info(f"Case {case_id}: semantic score = {score:.3f}")
    
    # Save evaluations
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "average_score": sum(all_scores) / len(all_scores) if all_scores else 0.0
        },
        "evaluations": evaluations
    }
    
    output_path = results_dir / "semantic_evaluation.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Saved semantic evaluation to {output_path}")
    logging.info(f"Average semantic score: {output['metadata']['average_score']:.3f}")
    
    return output


async def assign_severities_in_batches(
    diagnoses: List[str],
    judge_llm,
    prompt_template: str,
    schema: Dict[str, Any],
    batch_size: int = 100
) -> Dict[str, str]:
    """Process diagnoses in batches to assign severities."""
    logging.info(f"Assigning severities to {len(diagnoses)} unique diagnoses...")
    
    results = {}
    
    for i in range(0, len(diagnoses), batch_size):
        batch = diagnoses[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(diagnoses) + batch_size - 1) // batch_size
        
        logging.info(f"Processing batch {batch_num}/{total_batches}")
        
        # Format diagnoses as JSON list for prompt
        diagnoses_json = json.dumps(batch, ensure_ascii=False, indent=2)
        prompt = prompt_template.format(diagnoses=diagnoses_json)
        
        try:
            # Get runtime parameters from LLM config
            runtime_params = judge_llm._experiment_config.get('runtime_params', {})
            
            response = judge_llm.generate(prompt, schema=schema, **runtime_params)
            
            if isinstance(response, dict) and 'severities' in response:
                results.update(response['severities'])
            else:
                logging.error(f"Invalid response format for batch {batch_num}")
                
        except Exception as e:
            logging.error(f"Error processing batch {batch_num}: {e}")
    
    return results


async def evaluate_severities(
    cases: List[Dict[str, Any]],
    candidate_responses: Dict[str, Any],
    severity_assignments: Dict[str, str],
    config: Dict[str, Any],
    results_dir: Path
) -> Dict[str, Any]:
    """Evaluate severity matching between DDX and GDX."""
    logging.info("Evaluating severity scores...")
    
    evaluations = []
    all_scores = []
    
    # Get weights from config
    global UNDERDIAGNOSIS_WEIGHT, OVERDIAGNOSIS_WEIGHT
    UNDERDIAGNOSIS_WEIGHT = config['evaluation']['underdiagnosis_weight']
    OVERDIAGNOSIS_WEIGHT = config['evaluation']['overdiagnosis_weight']
    
    # Create mapping of case_id to responses
    response_map = {r['case_id']: r['ddx'] for r in candidate_responses['responses']}
    
    for case in cases:
        case_id = case['id']
        ddx = response_map.get(case_id, [])
        
        # Get severities for DDX
        ddx_severities = []
        for diagnosis in ddx:
            if diagnosis in severity_assignments:
                ddx_severities.append(severity_assignments[diagnosis])
        
        # Get severities for GDX
        gdx_severities = [diag['severity'] for diag in case.get('diagnoses', [])]
        
        # Calculate score
        score = calculate_severity_score(ddx_severities, gdx_severities)
        
        evaluations.append({
            "case_id": case_id,
            "score": score
        })
        all_scores.append(score)
        
        logging.info(f"Case {case_id}: severity score = {score:.3f}")
    
    # Save evaluations
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "average_score": sum(all_scores) / len(all_scores) if all_scores else 0.0
        },
        "evaluations": evaluations
    }
    
    output_path = results_dir / "severity_evaluation.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Saved severity evaluation to {output_path}")
    logging.info(f"Average severity score: {output['metadata']['average_score']:.3f}")
    
    return output


def generate_scatter_plot(
    semantic_eval: Dict[str, Any],
    severity_eval: Dict[str, Any],
    results_dir: Path,
    config: Dict[str, Any]
):
    """Generate scatter plot of severity vs semantic scores."""
    logging.info("Generating scatter plot...")
    
    # Extract scores
    semantic_scores = []
    severity_scores = []
    
    # Create mapping of case_id to scores
    semantic_map = {e['case_id']: e['score'] for e in semantic_eval['evaluations']}
    severity_map = {e['case_id']: e['score'] for e in severity_eval['evaluations']}
    
    # Get aligned scores
    for case_id in semantic_map:
        if case_id in severity_map:
            semantic_scores.append(semantic_map[case_id])
            severity_scores.append(severity_map[case_id])
    
    if not semantic_scores:
        logging.warning("No scores to plot")
        return
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot all points
    if config['plots']['scatter_all_points']:
        plt.scatter(severity_scores, semantic_scores, alpha=0.6, s=50, label='Cases')
    
    # Plot average
    if config['plots']['scatter_average']:
        avg_severity = sum(severity_scores) / len(severity_scores)
        avg_semantic = sum(semantic_scores) / len(semantic_scores)
        plt.scatter(avg_severity, avg_semantic, 
                   color='red', s=200, marker='*', zorder=5,
                   label=f'Average ({avg_severity:.2f}, {avg_semantic:.2f})')
    
    plt.xlabel('Severity Score', fontsize=12)
    plt.ylabel('Semantic Score', fontsize=12)
    plt.title('Diagnostic Evaluation: Severity vs Semantic Accuracy', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Add diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    plot_path = plots_dir / f"severity_vs_semantic.{config['plots']['save_format']}"
    plt.savefig(plot_path, dpi=config['plots']['dpi'])
    plt.close()
    
    logging.info(f"Saved scatter plot to {plot_path}")


def generate_distribution_histograms(
    semantic_eval: Dict[str, Any],
    severity_eval: Dict[str, Any],
    results_dir: Path,
    config: Dict[str, Any]
):
    """Generate histograms of score distributions."""
    logging.info("Generating distribution histograms...")
    
    # Extract scores
    semantic_scores = [e['score'] for e in semantic_eval['evaluations']]
    severity_scores = [e['score'] for e in severity_eval['evaluations']]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Semantic scores histogram
    ax1.hist(semantic_scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(np.mean(semantic_scores), color='red', linestyle='dashed', linewidth=2)
    ax1.set_xlabel('Semantic Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Semantic Score Distribution\nMean: {np.mean(semantic_scores):.3f}')
    ax1.set_xlim(0, 1)
    
    # Severity scores histogram
    ax2.hist(severity_scores, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(np.mean(severity_scores), color='red', linestyle='dashed', linewidth=2)
    ax2.set_xlabel('Severity Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Severity Score Distribution\nMean: {np.mean(severity_scores):.3f}')
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    plot_path = plots_dir / f"score_distributions.{config['plots']['save_format']}"
    plt.savefig(plot_path, dpi=config['plots']['dpi'])
    plt.close()
    
    logging.info(f"Saved distribution histograms to {plot_path}")


def warm_up_sapbert_model(max_attempts: int = 20, wait_seconds: int = 5):
    """
    Warm up the SapBERT model by making test calls until it responds.
    Simple and clean - no emojis, no verbose errors.
    """
    print("\nWarming up SapBERT model...")
    print("The model needs to wake up from HuggingFace's scaled-to-zero state.")
    print("This usually takes 10-30 seconds. Pinging every 5 seconds...\n")
    
    test_diagnoses_1 = ["Fever", "Headache"]
    test_diagnoses_2 = ["Pyrexia", "Cephalgia"]
    
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"Ping {attempt}...", end='', flush=True)
            
            # Try a simple semantic similarity calculation
            result = calculate_semantic_similarity(test_diagnoses_1, test_diagnoses_2)
            
            # If we get here without exception, the model is ready
            if result and len(result) > 0:
                print(" OK!")
                print(f"\nSapBERT is ready! (took {attempt} attempts)\n")
                return True
            else:
                print(" empty response")
                
        except Exception:
            # Don't print the error, just indicate it failed
            print(" failed")
            
        if attempt < max_attempts:
            time.sleep(wait_seconds)
        else:
            print("\nFailed to warm up SapBERT after all attempts.")
            raise Exception("SapBERT warm-up timeout")
    
    return False


async def main():

    print('Hello')
    """Main execution function."""
    # Load configuration
    config_path = Path("config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_dir = Path(f'results/{timestamp}')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_path = results_dir / 'execution.log'
    setup_logging(log_path)
    
    logging.info("Starting diagnostic evaluation experiment")
    logging.info(f"Results directory: {results_dir}")
    
    # Copy config to results directory
    shutil.copy2(config_path, results_dir / 'config_copy.yaml')
    
    # Create LLMs
    llms = create_llms_from_config('config.yaml')
    candidate_llm = llms['candidate']
    judge_llm = llms['judge']
    
    # Load prompts and schemas
    candidate_prompt = get_llm_prompt(candidate_llm)
    candidate_schema = get_llm_schema(candidate_llm)

    print(candidate_prompt) ///
    print(candidate_schema) ///

    print(🔶) ///

    judge_prompt = get_llm_prompt(judge_llm)
    judge_schema = get_llm_schema(judge_llm)
    
    # Process each dataset
    for dataset_path in config['datasets']:
        dataset_name = Path(dataset_path).stem
        logging.info(f"\nProcessing dataset: {dataset_name}")
        
        # Load dataset
        full_dataset_path = Path(__file__).parent.parent.parent / dataset_path
        cases = load_dataset(full_dataset_path)
        logging.info(f"Loaded {len(cases)} cases")
        
        # Generate candidate responses
        candidate_responses = await generate_candidate_responses(
            cases, candidate_llm, candidate_prompt, candidate_schema, results_dir
        )
        
        # Warm up SapBERT model right before semantic evaluation
        warm_up_sapbert_model()
        
        # Evaluate semantic similarity
        semantic_eval = evaluate_semantic_similarity(
            cases, candidate_responses, results_dir
        )
        
        # Get unique diagnoses for severity assignment
        unique_diagnoses = set()
        for response in candidate_responses['responses']:
            unique_diagnoses.update(response['ddx'])
        
        # Also add golden diagnoses
        for case in cases:
            for diag in case.get('diagnoses', []):
                unique_diagnoses.add(diag['name'])
        
        unique_diagnoses = sorted(list(unique_diagnoses))
        logging.info(f"Found {len(unique_diagnoses)} unique diagnoses")
        
        # Assign severities
        severity_assignments = await assign_severities_in_batches(
            unique_diagnoses,
            judge_llm,
            judge_prompt,
            judge_schema,
            config['evaluation']['batch_size']
        )
        
        # Save severity assignments
        severity_output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": judge_llm.config.deployment_name,
                "total_diagnoses": len(severity_assignments)
            },
            "severities": severity_assignments
        }
        
        severity_path = results_dir / "severity_assignments.json"
        with open(severity_path, 'w', encoding='utf-8') as f:
            json.dump(severity_output, f, ensure_ascii=False, indent=2)
        
        # Evaluate severities
        severity_eval = await evaluate_severities(
            cases, candidate_responses, severity_assignments, config, results_dir
        )
        
        # Generate visualizations
        if config['plots']['scatter_all_points'] or config['plots']['scatter_average']:
            generate_scatter_plot(semantic_eval, severity_eval, results_dir, config)
        
        if config['plots']['distribution_histograms']:
            generate_distribution_histograms(semantic_eval, severity_eval, results_dir, config)
    
    logging.info("\nExperiment completed successfully!")
    logging.info(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    asyncio.run(main())