#!/usr/bin/env python3
"""
DxGPT Evaluation Framework CLI

Main entry point for running evaluations with various configurations.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from core.session import EvaluationSession
from core.registry import EvaluatorRegistry
from core.config import ConfigManager
from visualizations import visualize_results


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]  # Console handler
    
    if log_file:
        # Create file handler
        file_handler = logging.FileHandler(log_file, mode='a')
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )


def list_evaluators(registry: EvaluatorRegistry):
    """List available evaluators."""
    print("\n=== Available Evaluators ===\n")
    
    evaluators = registry.list_evaluators()
    if not evaluators:
        print("No evaluators found.")
        return
    
    for evaluator in evaluators:
        print(f"Name: {evaluator['name']}")
        print(f"Description: {evaluator['description']}")
        print(f"Expected columns: {', '.join(evaluator['expected_columns'])}")
        print("Column descriptions:")
        for col, desc in evaluator['column_descriptions'].items():
            print(f"  - {col}: {desc}")
        print()


def list_datasets(dataset_dirs: List[str]):
    """List available datasets in specified directories."""
    default_dirs = ["src/golden-dataset", "src/datasets"]
    
    if not dataset_dirs:
        dataset_dirs = default_dirs
    
    for dataset_dir in dataset_dirs:
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            continue
            
        csv_files = list(dataset_path.glob("*.csv"))
        if csv_files:
            print(f"\n=== Datasets in {dataset_dir} ===")
            for csv_file in csv_files:
                print(f"- {csv_file.name}")
                
                # Try to get row count
                try:
                    import csv
                    with open(csv_file, 'r') as f:
                        row_count = sum(1 for _ in csv.reader(f)) - 1
                    print(f"  Rows: {row_count}")
                except:
                    pass


def resolve_dataset_paths(dataset_args: List[str]) -> List[str]:
    """
    Resolve dataset arguments to full paths.
    If the argument is a simple filename, look for it in src/datasets.
    If it's already a path, use it as-is.
    """
    resolved_paths = []
    datasets_dir = Path("src/datasets")
    
    for dataset_arg in dataset_args:
        dataset_path = Path(dataset_arg)
        
        # If it's an absolute path or contains path separators, use as-is
        if dataset_path.is_absolute() or "/" in dataset_arg or "\\" in dataset_arg:
            resolved_paths.append(dataset_arg)
        else:
            # Simple filename - look in src/datasets
            candidate_path = datasets_dir / dataset_arg
            
            # If filename doesn't have .csv extension, try adding it
            if not dataset_arg.endswith('.csv'):
                candidate_with_csv = datasets_dir / f"{dataset_arg}.csv"
                if candidate_with_csv.exists():
                    resolved_paths.append(str(candidate_with_csv))
                    continue
            
            # Use the path in src/datasets
            resolved_paths.append(str(candidate_path))
    
    return resolved_paths


def run_evaluation(
    evaluator_names: List[str],
    dataset_paths: List[str],
    config_manager: ConfigManager,
    session: EvaluationSession,
    registry: EvaluatorRegistry
):
    """Run evaluation with specified evaluators and datasets."""
    logger = logging.getLogger(__name__)
    
    # Filter enabled evaluators
    enabled_evaluators = [
        name for name in evaluator_names 
        if config_manager.is_evaluator_enabled(name)
    ]
    
    if not enabled_evaluators:
        logger.error("No enabled evaluators found.")
        return
    
    logger.info(f"Running evaluators: {', '.join(enabled_evaluators)}")
    logger.info(f"On datasets: {', '.join(dataset_paths)}")
    
    # Run each evaluator on each dataset
    all_results = {}
    
    for evaluator_name in enabled_evaluators:
        evaluator = registry.get_evaluator(evaluator_name)
        if not evaluator:
            logger.error(f"Evaluator '{evaluator_name}' not found in registry.")
            continue
        
        evaluator_config = config_manager.get_evaluator_config(evaluator_name)
        
        # Register evaluator in session
        session.register_evaluator(
            evaluator_name,
            f"src/evaluators/{evaluator_name}.py",
            evaluator_config
        )
        
        for dataset_path in dataset_paths:
            dataset_name = Path(dataset_path).stem
            logger.info(f"\nRunning {evaluator_name} on {dataset_name}...")
            
            try:
                # Register dataset
                session.register_dataset(dataset_name, dataset_path)
                
                # Run evaluation
                results = evaluator.evaluate(dataset_path, evaluator_config)
                
                # Log results
                session.log_result(evaluator_name, dataset_name, results)
                
                # Store for visualization
                result_key = f"{evaluator_name}_{dataset_name}"
                all_results[result_key] = results
                
                # Print summary
                if 'summary' in results:
                    print(f"\n=== {evaluator_name} Results on {dataset_name} ===")
                    for key, value in results['summary'].items():
                        print(f"{key}: {value}")
                
            except Exception as e:
                logger.error(f"Error running {evaluator_name} on {dataset_path}: {e}")
                import traceback
                traceback.print_exc()
    
    return all_results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="DxGPT Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available evaluators
  py evaluate.py --list-evaluators
  
  # List available datasets
  py evaluate.py --list-datasets
  
  # Run semantic evaluation on mini-test dataset (searches in src/datasets)
  py evaluate.py --evaluators semantic --datasets mini-test
  
  # Run with .csv extension (searches in src/datasets)
  py evaluate.py --evaluators semantic --datasets mini-test.csv
  
  # Run with full path (uses exact path)
  py evaluate.py --evaluators semantic --datasets src/golden-dataset/mini-test.csv
  
  # Run with custom config
  py evaluate.py --config my_config.yaml --evaluators semantic --datasets mini-test
  
  # Run multiple evaluators on multiple datasets
  py evaluate.py --evaluators semantic severity --datasets mini-test other-dataset
  
  # Disable visualizations
  py evaluate.py --evaluators semantic --datasets mini-test --no-viz
        """
    )
    
    # Action arguments
    parser.add_argument('--list-evaluators', action='store_true',
                        help='List available evaluators')
    parser.add_argument('--list-datasets', action='store_true',
                        help='List available datasets in default directory')
    
    # Evaluation arguments
    parser.add_argument('--evaluators', nargs='+', metavar='NAME',
                        help='Evaluators to run')
    parser.add_argument('--datasets', nargs='+', metavar='DATASET',
                        help='Dataset names (searched in src/datasets) or full paths')
    
    # Configuration arguments
    parser.add_argument('--config', metavar='PATH',
                        help='Custom configuration file (YAML or JSON)')
    parser.add_argument('--set', nargs=2, action='append', metavar=('KEY', 'VALUE'),
                        help='Override config values (e.g., --set evaluators.semantic.config.batch_size 10)')
    
    # Output arguments
    parser.add_argument('--output-dir', metavar='PATH',
                        help='Output directory for results')
    parser.add_argument('--no-viz', action='store_true',
                        help='Disable visualizations')
    
    # Other arguments
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    args = parser.parse_args()
    
    # Initial setup without file logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Initialize components
    registry = EvaluatorRegistry()
    config_manager = ConfigManager()
    
    # Load custom config if provided
    if args.config:
        logger.info(f"Loading custom config from {args.config}")
        config_manager.load_custom_config(args.config)
    
    # Apply config overrides
    if args.set:
        for key, value in args.set:
            # Try to parse value as JSON first (for complex types)
            try:
                value = json.loads(value)
            except:
                # Try to parse as number
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except:
                    # Keep as string
                    pass
            
            logger.debug(f"Setting {key} = {value}")
            config_manager.set(key, value)
    
    # Handle output directory
    if args.output_dir:
        config_manager.set('general.output_dir', args.output_dir)
    
    # Handle list actions
    if args.list_evaluators:
        list_evaluators(registry)
        return 0
    
    if args.list_datasets:
        list_datasets([])  # Will use default directories
        return 0
    
    # Check if evaluation requested
    if not args.evaluators or not args.datasets:
        parser.print_help()
        return 1
    
    # Resolve dataset paths
    resolved_dataset_paths = resolve_dataset_paths(args.datasets)
    logger.info(f"Resolved datasets: {resolved_dataset_paths}")
    
    # Disable visualizations if requested
    if args.no_viz:
        config_manager.set('visualizations.enabled', False)
    
    # Create evaluation session
    session = EvaluationSession()
    session.update_config(config_manager.to_dict())
    
    # Setup file logging in session directory
    log_file = session.results_dir / "evaluation.log"
    setup_logging(args.log_level, log_file)
    logger.info(f"Logging to: {log_file}")
    
    try:
        # Run evaluations
        results = run_evaluation(
            args.evaluators,
            resolved_dataset_paths,
            config_manager,
            session,
            registry
        )
        
        # Generate visualizations if enabled
        if config_manager.get('visualizations.enabled', True) and results:
            logger.info("\nGenerating visualizations...")
            try:
                visualize_results(results, session.results_dir, config_manager)
            except Exception as e:
                logger.error(f"Error generating visualizations: {e}")
        
        # Finish session
        session.finish()
        
        # Print summary
        print(f"\n=== Evaluation Complete ===")
        print(f"Session ID: {session.session_id}")
        print(f"Results saved to: {session.results_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())