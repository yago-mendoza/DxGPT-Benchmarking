"""
Semantic Similarity Evaluator

Evaluates diagnostic predictions using semantic similarity between model outputs
and golden standard diagnoses.
"""

import json
import csv
import re
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.llm import AzureLLM
from utils.bert import calculate_semantic_similarity
from core.registry import BaseEvaluator


class SemanticEvaluator(BaseEvaluator):
    """Evaluates predictions using semantic similarity metrics."""
    
    # Expected CSV columns
    EXPECTED_COLUMNS = ['id', 'case', 'diagnosis']
    COLUMN_DESCRIPTIONS = {
        'id': 'Unique identifier for each case',
        'case': 'Patient case description/symptoms',
        'diagnosis': 'Golden standard diagnosis(es) separated by semicolon (;)'
    }
    
    def get_name(self) -> str:
        """Get evaluator name."""
        return "semantic"
    
    def get_description(self) -> str:
        """Get evaluator description."""
        return "Evaluates diagnostic accuracy using BERT-based semantic similarity between predictions and golden diagnoses"
    
    def evaluate(self, dataset_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run semantic evaluation on dataset.
        
        Config options:
        - batch_size: Number of cases to process together (default: 5)
        - n_diagnoses: Number of diagnoses to generate per case (default: 5)
        - llm_model: LLM deployment name (default: "gpt-4o")
        - temperature: LLM temperature (default: 0.3)
        - prompt_path: Path to prompt template (default: "src/dxgpt-prompt/dxgpt-prompt.txt")
        """
        # Validate dataset
        if not self.validate_dataset(dataset_path):
            raise ValueError("Dataset validation failed")
        
        # Extract config with defaults
        batch_size = config.get('batch_size', 5)
        n_diagnoses = config.get('n_diagnoses', 5)
        llm_model = config.get('llm_model', 'gpt-4o')
        temperature = config.get('temperature', 0.3)
        prompt_path = config.get('prompt_path', 'src/dxgpt-prompt/dxgpt-prompt.txt')
        
        # Load dataset
        dataset = self._load_dataset(dataset_path)
        
        # Run evaluation
        results = self._evaluate_dataset(
            dataset, 
            batch_size=batch_size,
            n_diagnoses=n_diagnoses,
            llm_model=llm_model,
            temperature=temperature,
            prompt_path=prompt_path
        )
        
        # Calculate summary statistics
        summary = self._calculate_summary(results)
        
        return {
            'evaluator': self.get_name(),
            'dataset': dataset_path,
            'config': config,
            'summary': summary,
            'detailed_results': results
        }
    
    def _load_dataset(self, csv_path: str) -> List[Dict[str, Any]]:
        """Load dataset from CSV file."""
        dataset = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset.append({
                    'id': row['id'],
                    'case': row['case'],
                    'diagnosis': row['diagnosis'],
                    # Include other columns if present
                    **{k: v for k, v in row.items() if k not in ['id', 'case', 'diagnosis']}
                })
        return dataset
    
    def _load_prompt_template(self, prompt_path: str) -> str:
        """Load the DxGPT prompt template from file."""
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _parse_diagnosis_output(self, llm_response: str) -> List[Dict[str, Any]]:
        """Extract diagnosis list from LLM response containing XML tags."""
        try:
            # Find the diagnosis_output XML content
            match = re.search(r'<diagnosis_output>(.*?)</diagnosis_output>', llm_response, re.DOTALL)
            if not match:
                print("WARNING: No <diagnosis_output> tags found in response")
                return []
            
            json_content = match.group(1).strip()
            return json.loads(json_content)
        except (json.JSONDecodeError, Exception) as e:
            print(f"ERROR: Failed to parse diagnosis output: {e}")
            return []
    
    def _process_batch_with_dxgpt(
        self, 
        llm: AzureLLM, 
        prompt_template: str, 
        batch_cases: List[Dict[str, Any]],
        n_diagnoses: int
    ) -> List[List[Dict[str, Any]]]:
        """Process a batch of cases with DxGPT and return N diagnoses for each."""
        # Prepare batch items for the LLM
        batch_items = []
        for case in batch_cases:
            batch_items.append({
                "id": case['id'],
                "description": case['case']
            })
        
        # Create the prompt requesting N diagnoses per case
        batch_prompt = f"""Process each patient case and provide exactly {n_diagnoses} potential diagnoses for each.

For each case, follow the format specified in the template and ensure you return exactly {n_diagnoses} diagnoses.

{prompt_template}

Process each case independently and return the results as a JSON array where each element corresponds to one input case."""
        
        try:
            # Use batch processing
            response = llm.generate(
                batch_prompt,
                batch_items=batch_items,
                max_tokens=4000
            )
            
            # Response should be a list of results
            if isinstance(response, list):
                # Parse each result to extract diagnoses
                all_diagnoses = []
                for i, result in enumerate(response):
                    if isinstance(result, str):
                        diagnoses = self._parse_diagnosis_output(result)
                    else:
                        # If it's already parsed JSON
                        diagnoses = result if isinstance(result, list) else []
                    
                    # Ensure we have exactly n_diagnoses
                    if len(diagnoses) > n_diagnoses:
                        diagnoses = diagnoses[:n_diagnoses]
                    elif len(diagnoses) < n_diagnoses:
                        # Pad with empty diagnoses
                        while len(diagnoses) < n_diagnoses:
                            diagnoses.append({
                                "diagnosis": "No additional diagnosis",
                                "description": "No additional diagnosis available",
                                "symptoms_in_common": [],
                                "symptoms_not_in_common": []
                            })
                    
                    all_diagnoses.append(diagnoses)
                
                return all_diagnoses
            else:
                print("ERROR: Unexpected response format from LLM")
                return [[] for _ in batch_cases]
                
        except Exception as e:
            print(f"ERROR: Failed to process batch: {e}")
            return [[] for _ in batch_cases]
    
    def _calculate_best_similarity_score(
        self, 
        predicted_diagnoses: List[Dict[str, Any]], 
        golden_diagnoses: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate the best similarity score between predicted and golden diagnoses.
        
        Returns a dict with the best score and detailed similarity results.
        """
        # Extract diagnosis names from predictions
        predicted_names = [dx.get('diagnosis', '') for dx in predicted_diagnoses if dx.get('diagnosis')]
        
        if not predicted_names or not golden_diagnoses:
            return {
                'best_score': 0.0,
                'predicted_diagnoses': predicted_names,
                'similarity_scores': []
            }
        
        # Calculate semantic similarity for all pairs
        try:
            similarity_results = calculate_semantic_similarity(predicted_names, golden_diagnoses)
        except Exception as e:
            print(f"WARNING: Semantic similarity calculation failed: {e}")
            print("Using fallback: exact string matching")
            # Fallback to exact string matching
            similarity_results = {}
            for pred in predicted_names:
                similarity_results[pred] = {}
                for gold in golden_diagnoses:
                    # Simple normalization and comparison
                    pred_normalized = pred.lower().strip()
                    gold_normalized = gold.lower().strip()
                    if pred_normalized == gold_normalized:
                        similarity_results[pred][gold] = 1.0
                    elif pred_normalized in gold_normalized or gold_normalized in pred_normalized:
                        similarity_results[pred][gold] = 0.5
                    else:
                        similarity_results[pred][gold] = 0.0
        
        # Find the maximum similarity score
        best_score = 0.0
        all_scores = []
        
        for pred_name in predicted_names:
            pred_scores = []
            for golden_name in golden_diagnoses:
                score = similarity_results.get(pred_name, {}).get(golden_name, 0.0)
                if score is None:
                    score = 0.0
                pred_scores.append(score)
                best_score = max(best_score, score)
            all_scores.append(pred_scores)
        
        return {
            'best_score': best_score,
            'predicted_diagnoses': predicted_names,
            'similarity_scores': all_scores
        }
    
    def _evaluate_dataset(
        self, 
        dataset: List[Dict[str, Any]], 
        batch_size: int,
        n_diagnoses: int,
        llm_model: str,
        temperature: float,
        prompt_path: str
    ) -> List[Dict[str, Any]]:
        """Evaluate the entire dataset using batched processing."""
        # Initialize LLM
        llm = AzureLLM(llm_model, temperature=temperature)
        
        # Load prompt template
        prompt_template = self._load_prompt_template(prompt_path)
        
        # Process results
        results = []
        
        # Process in batches
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(dataset) + batch_size - 1)//batch_size}...")
            
            # Get predictions for the batch
            batch_predictions = self._process_batch_with_dxgpt(llm, prompt_template, batch, n_diagnoses)
            
            # Calculate similarity scores for each case
            for j, case in enumerate(batch):
                # Split golden diagnoses by semicolon
                golden_diagnoses = [d.strip() for d in case['diagnosis'].split(';') if d.strip()]
                
                # Get predicted diagnoses for this case
                predicted_diagnoses = batch_predictions[j] if j < len(batch_predictions) else []
                
                # Calculate best similarity score
                similarity_result = self._calculate_best_similarity_score(predicted_diagnoses, golden_diagnoses)
                
                # Create result entry
                result = {
                    'uid': case['id'],
                    'score': similarity_result['best_score'],
                    'diagnosis': case['diagnosis'],  # Original diagnosis with semicolons
                    'details': {
                        'ddx': similarity_result['predicted_diagnoses'][:n_diagnoses],
                        'similarity_matrix': similarity_result['similarity_scores']
                    }
                }
                
                results.append(result)
                print(f"  Case {case['id']}: Best score = {result['score']:.3f}")
        
        return results
    
    def _calculate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics from results."""
        scores = [r['score'] for r in results]
        
        if not scores:
            return {
                'total_cases': 0,
                'avg_score': 0.0,
                'min_score': 0.0,
                'max_score': 0.0,
                'perfect_matches': 0
            }
        
        return {
            'total_cases': len(scores),
            'avg_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'perfect_matches': sum(1 for s in scores if s >= 0.95),
            'score_distribution': {
                '0.0-0.2': sum(1 for s in scores if 0.0 <= s < 0.2),
                '0.2-0.4': sum(1 for s in scores if 0.2 <= s < 0.4),
                '0.4-0.6': sum(1 for s in scores if 0.4 <= s < 0.6),
                '0.6-0.8': sum(1 for s in scores if 0.6 <= s < 0.8),
                '0.8-1.0': sum(1 for s in scores if 0.8 <= s <= 1.0),
            }
        }