"""
Main visualization orchestrator

Coordinates visualization generation based on configuration.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
import json

from .semantic_viz import SemanticVisualizer
from .combined_viz import CombinedVisualizer


logger = logging.getLogger(__name__)


def visualize_results(
    results: Dict[str, Any],
    output_dir: Path,
    config_manager: Any
):
    """
    Generate visualizations based on results and configuration.
    
    Args:
        results: Dictionary of evaluation results
        output_dir: Directory to save visualizations
        config_manager: Configuration manager instance
    """
    # Create visualizations directory
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Get visualization settings
    output_formats = config_manager.get('visualizations.output_format', ['png'])
    
    # Process semantic evaluator visualizations
    semantic_results = {k: v for k, v in results.items() if 'semantic' in k}
    if semantic_results:
        logger.info("Generating semantic visualizations...")
        semantic_viz = SemanticVisualizer()
        
        semantic_configs = config_manager.get('visualizations.semantic', [])
        for viz_config in semantic_configs:
            if viz_config.get('enabled', True):
                try:
                    viz_type = viz_config['type']
                    logger.info(f"  Creating {viz_type}...")
                    
                    for result_key, result_data in semantic_results.items():
                        output_prefix = viz_dir / f"{result_key}_{viz_type}"
                        
                        if viz_type == 'score_distribution':
                            semantic_viz.plot_score_distribution(
                                result_data['detailed_results'],
                                str(output_prefix),
                                output_formats
                            )
                        elif viz_type == 'confusion_matrix':
                            semantic_viz.plot_confusion_matrix(
                                result_data['detailed_results'],
                                str(output_prefix),
                                output_formats
                            )
                        elif viz_type == 'top_errors':
                            semantic_viz.plot_top_errors(
                                result_data['detailed_results'],
                                str(output_prefix),
                                output_formats
                            )
                            
                except Exception as e:
                    logger.error(f"Error creating {viz_type} visualization: {e}")
    
    # Process combined visualizations (when multiple evaluators are available)
    combined_configs = config_manager.get('visualizations.combined', [])
    if len(results) > 1 and any(cfg.get('enabled', False) for cfg in combined_configs):
        logger.info("Generating combined visualizations...")
        combined_viz = CombinedVisualizer()
        
        for viz_config in combined_configs:
            if viz_config.get('enabled', False):
                try:
                    viz_type = viz_config['type']
                    logger.info(f"  Creating {viz_type}...")
                    
                    if viz_type == '2d_scatter':
                        # This would be used when both semantic and severity are available
                        output_path = viz_dir / "combined_2d_scatter"
                        combined_viz.plot_2d_scatter(
                            results,
                            viz_config['x_axis'],
                            viz_config['y_axis'],
                            str(output_path),
                            output_formats
                        )
                        
                except Exception as e:
                    logger.error(f"Error creating {viz_type} visualization: {e}")
    
    # Save visualization metadata
    viz_metadata = {
        'generated_visualizations': [str(p) for p in viz_dir.glob('*')],
        'config': {
            'output_formats': output_formats,
            'semantic_configs': semantic_configs if semantic_results else [],
            'combined_configs': combined_configs if len(results) > 1 else []
        }
    }
    
    with open(viz_dir / 'visualization_metadata.json', 'w') as f:
        json.dump(
            {k: str(v) if isinstance(v, Path) else v for k, v in viz_metadata.items()},
            f,
            indent=2
        )
    
    logger.info(f"Visualizations saved to: {viz_dir}")