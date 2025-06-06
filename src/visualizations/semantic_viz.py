"""
Visualization functions for semantic evaluation results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any
import pandas as pd
from pathlib import Path


class SemanticVisualizer:
    """Visualizations specific to semantic evaluation results."""
    
    def plot_score_distribution(
        self, 
        results: List[Dict[str, Any]], 
        output_prefix: str,
        formats: List[str] = ['png']
    ):
        """Plot distribution of similarity scores."""
        scores = [r['score'] for r in results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(scores, bins=20, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Similarity Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Similarity Scores')
        ax1.axvline(np.mean(scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(scores):.3f}')
        ax1.legend()
        
        # Box plot
        ax2.boxplot(scores, vert=True)
        ax2.set_ylabel('Similarity Score')
        ax2.set_title('Score Distribution Summary')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        for fmt in formats:
            if fmt != 'html':  # Skip HTML for matplotlib plots
                plt.savefig(f"{output_prefix}.{fmt}", dpi=300, bbox_inches='tight')
        
        plt.close()
        
        # Also create a detailed histogram by score ranges
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        hist, _ = np.histogram(scores, bins=bins)
        
        ax.bar(range(len(hist)), hist, tick_label=[f'{bins[i]:.1f}-{bins[i+1]:.1f}' 
                                                   for i in range(len(bins)-1)])
        ax.set_xlabel('Score Range')
        ax.set_ylabel('Number of Cases')
        ax.set_title('Cases by Score Range')
        
        # Add value labels on bars
        for i, v in enumerate(hist):
            ax.text(i, v + 0.5, str(v), ha='center')
        
        plt.tight_layout()
        
        for fmt in formats:
            if fmt != 'html':  # Skip HTML for matplotlib plots
                plt.savefig(f"{output_prefix}_ranges.{fmt}", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_confusion_matrix(
        self,
        results: List[Dict[str, Any]],
        output_prefix: str,
        formats: List[str] = ['png']
    ):
        """Plot confusion matrix of predictions vs golden diagnoses."""
        # Extract top prediction and golden diagnosis for each case
        data = []
        for r in results:
            golden = r['diagnosis'].split(';')[0].strip()  # First golden diagnosis
            if r['details']['ddx']:
                predicted = r['details']['ddx'][0]  # Top prediction
            else:
                predicted = "No prediction"
            
            data.append({
                'golden': golden[:30] + '...' if len(golden) > 30 else golden,
                'predicted': predicted[:30] + '...' if len(predicted) > 30 else predicted,
                'score': r['score']
            })
        
        df = pd.DataFrame(data)
        
        # Create confusion matrix
        confusion_df = pd.crosstab(df['golden'], df['predicted'])
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(confusion_df, annot=True, fmt='d', cmap='Blues', 
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix: Golden vs Predicted Diagnoses')
        plt.xlabel('Predicted Diagnosis')
        plt.ylabel('Golden Diagnosis')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        for fmt in formats:
            if fmt != 'html':  # Skip HTML for matplotlib plots
                plt.savefig(f"{output_prefix}.{fmt}", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_top_errors(
        self,
        results: List[Dict[str, Any]],
        output_prefix: str,
        formats: List[str] = ['png'],
        n_errors: int = 10
    ):
        """Plot top N cases with lowest similarity scores."""
        # Sort by score (ascending)
        sorted_results = sorted(results, key=lambda x: x['score'])
        
        # Get top N errors
        top_errors = sorted_results[:n_errors]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        case_ids = [r['uid'] for r in top_errors]
        scores = [r['score'] for r in top_errors]
        diagnoses = [r['diagnosis'].split(';')[0][:40] + '...' 
                    if len(r['diagnosis'].split(';')[0]) > 40 
                    else r['diagnosis'].split(';')[0] 
                    for r in top_errors]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(case_ids))
        bars = ax.barh(y_pos, scores)
        
        # Color bars based on score
        colors = ['red' if s < 0.2 else 'orange' if s < 0.5 else 'yellow' 
                 for s in scores]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"Case {cid}\n{diag}" for cid, diag in zip(case_ids, diagnoses)])
        ax.set_xlabel('Similarity Score')
        ax.set_title(f'Top {n_errors} Cases with Lowest Similarity Scores')
        ax.set_xlim(0, 1)
        
        # Add score values on bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{score:.3f}', va='center')
        
        plt.tight_layout()
        
        for fmt in formats:
            if fmt != 'html':  # Skip HTML for matplotlib plots
                plt.savefig(f"{output_prefix}.{fmt}", dpi=300, bbox_inches='tight')
        
        plt.close()
        
        # Also save detailed error analysis as HTML
        if 'html' in formats:
            self._save_error_analysis_html(top_errors, f"{output_prefix}.html")
    
    def _save_error_analysis_html(self, errors: List[Dict[str, Any]], output_path: str):
        """Save detailed error analysis as HTML."""
        html_content = """
        <html>
        <head>
            <title>Error Analysis</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .case { margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; }
                .score { font-size: 18px; font-weight: bold; }
                .low { color: red; }
                .medium { color: orange; }
                .high { color: green; }
                table { border-collapse: collapse; width: 100%; margin-top: 10px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Detailed Error Analysis</h1>
        """
        
        for error in errors:
            score_class = 'low' if error['score'] < 0.2 else 'medium' if error['score'] < 0.5 else 'high'
            
            html_content += f"""
            <div class="case">
                <h2>Case {error['uid']}</h2>
                <p class="score {score_class}">Similarity Score: {error['score']:.3f}</p>
                <p><strong>Golden Diagnosis:</strong> {error['diagnosis']}</p>
                <h3>Predicted Diagnoses:</h3>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Diagnosis</th>
                        <th>Similarity Scores</th>
                    </tr>
            """
            
            for i, ddx in enumerate(error['details']['ddx']):
                scores = error['details']['similarity_matrix'][i] if i < len(error['details']['similarity_matrix']) else []
                scores_str = ', '.join([f'{s:.3f}' for s in scores])
                
                html_content += f"""
                    <tr>
                        <td>{i+1}</td>
                        <td>{ddx}</td>
                        <td>{scores_str}</td>
                    </tr>
                """
            
            html_content += """
                </table>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)