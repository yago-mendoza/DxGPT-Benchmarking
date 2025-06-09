import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import seaborn as sns
from collections import defaultdict
import argparse

def load_json(filepath):
    """Load JSON data from file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_semantic_bias_evaluation(semantic_data, output_dir):
    """
    Plot 1: Semantic Evaluation by Bias
    Y-axis: Semantic evaluation levels (0-10)
    X-axis: Distance from center (negative=pessimist, positive=optimist)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Collect data points
    semantic_levels = []
    optimist_counts = []
    pessimist_counts = []
    
    # Process each case
    for evaluation in semantic_data['evaluations']:
        best_score = evaluation['best_match']['score']
        semantic_level = int(best_score * 10)  # Convert to 0-10 scale
        
        # Count optimist/pessimist evaluators based on DDX scores
        optimist_count = 0
        pessimist_count = 0
        
        for ddx_name, scores in evaluation['ddx_semantic_scores'].items():
            avg_score = np.mean(scores)
            if avg_score > best_score:
                optimist_count += 1
            elif avg_score < best_score:
                pessimist_count += 1
        
        semantic_levels.append(semantic_level)
        optimist_counts.append(optimist_count)
        pessimist_counts.append(pessimist_count)
    
    # Create scatter plot
    for i in range(len(semantic_levels)):
        level = semantic_levels[i]
        # Plot pessimist point (negative x)
        if pessimist_counts[i] > 0:
            ax.scatter(-pessimist_counts[i], level, color='red', alpha=0.6, s=100)
        # Plot optimist point (positive x)
        if optimist_counts[i] > 0:
            ax.scatter(optimist_counts[i], level, color='blue', alpha=0.6, s=100)
    
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Distance from Center\n← Pessimists | Optimists →', fontsize=12)
    ax.set_ylabel('Semantic Evaluation Level (0-10)', fontsize=12)
    ax.set_title('Semantic Evaluation by Evaluator Bias', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.6, label='Pessimist Evaluators'),
        Patch(facecolor='blue', alpha=0.6, label='Optimist Evaluators')
    ]
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'semantic_bias_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_gdx_ddx_severity(severity_data, output_dir):
    """
    Plot 2: GDX vs DDX Severity
    Y-axis: DDX severity average (separated by optimist/pessimist)
    X-axis: Associated GDX severity
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    gdx_severities = []
    ddx_optimist_severities = []
    ddx_pessimist_severities = []
    
    for evaluation in severity_data['evaluations']:
        gdx_severity = int(evaluation['gdx']['severity'][1:])  # Extract number from S9, S8, etc.
        
        # Get optimist and pessimist DDX severities
        optimist_sevs = []
        pessimist_sevs = []
        
        for ddx in evaluation['ddx_list']:
            ddx_severity = int(ddx['severity'][1:])
            if ddx['distance'] < 0:  # Pessimist (lower severity than GDX)
                pessimist_sevs.append(ddx_severity)
            elif ddx['distance'] > 0:  # Optimist (higher severity than GDX)
                optimist_sevs.append(ddx_severity)
        
        gdx_severities.append(gdx_severity)
        ddx_optimist_severities.append(np.mean(optimist_sevs) if optimist_sevs else gdx_severity)
        ddx_pessimist_severities.append(np.mean(pessimist_sevs) if pessimist_sevs else gdx_severity)
    
    # Create scatter plots
    ax.scatter(gdx_severities, ddx_optimist_severities, color='blue', alpha=0.6, s=100, label='Optimist DDX')
    ax.scatter(gdx_severities, ddx_pessimist_severities, color='red', alpha=0.6, s=100, label='Pessimist DDX')
    
    # Add diagonal reference line
    min_sev = min(min(gdx_severities), 3)
    max_sev = max(max(gdx_severities), 10)
    ax.plot([min_sev, max_sev], [min_sev, max_sev], 'k--', alpha=0.5, label='Perfect Match')
    
    ax.set_xlabel('GDX Severity', fontsize=12)
    ax.set_ylabel('Average DDX Severity', fontsize=12)
    ax.set_title('GDX vs DDX Severity Comparison\n(Optimist vs Pessimist Evaluators)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gdx_ddx_severity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_semantic_distribution(semantic_data, output_dir):
    """Plot distribution of semantic similarity scores"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Collect all scores
    all_scores = []
    best_scores = []
    
    for evaluation in semantic_data['evaluations']:
        best_scores.append(evaluation['best_match']['score'])
        for scores in evaluation['ddx_semantic_scores'].values():
            all_scores.extend(scores)
    
    # Plot 1: Distribution of all semantic scores
    ax1.hist(all_scores, bins=30, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Semantic Similarity Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of All Semantic Similarity Scores', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distribution of best match scores
    ax2.hist(best_scores, bins=20, edgecolor='black', alpha=0.7, color='green')
    ax2.set_xlabel('Best Match Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Best Match Scores', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'semantic_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_severity_distribution(severity_data, output_dir):
    """Plot distribution of severity scores"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Collect severity data
    gdx_severities = []
    ddx_severities = []
    severity_distances = []
    
    for evaluation in severity_data['evaluations']:
        gdx_sev = int(evaluation['gdx']['severity'][1:])
        gdx_severities.append(gdx_sev)
        
        for ddx in evaluation['ddx_list']:
            ddx_sev = int(ddx['severity'][1:])
            ddx_severities.append(ddx_sev)
            severity_distances.append(ddx['distance'])
    
    # Plot 1: GDX vs DDX severity distributions
    ax1.hist([gdx_severities, ddx_severities], bins=range(3, 11), 
             label=['GDX', 'DDX'], alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Severity Level', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Severity Levels', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distribution of severity distances
    ax2.hist(severity_distances, bins=range(-5, 6), edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Severity Distance (DDX - GDX)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Severity Distances', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'severity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_semantic_vs_severity_correlation(semantic_data, severity_data, output_dir):
    """Plot correlation between semantic similarity and severity accuracy"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Match cases between semantic and severity data
    semantic_scores = []
    severity_scores = []
    
    for sem_eval in semantic_data['evaluations']:
        case_id = sem_eval['case_id']
        
        # Find corresponding severity evaluation
        for sev_eval in severity_data['evaluations']:
            if sev_eval['id'] == case_id:
                semantic_scores.append(sem_eval['best_match']['score'])
                severity_scores.append(sev_eval['final_score'])
                break
    
    # Create scatter plot
    ax.scatter(semantic_scores, severity_scores, alpha=0.6, s=100)
    
    # Add regression line
    z = np.polyfit(semantic_scores, severity_scores, 1)
    p = np.poly1d(z)
    ax.plot(semantic_scores, p(semantic_scores), "r--", alpha=0.8, 
            label=f'y={z[0]:.3f}x+{z[1]:.3f}')
    
    # Calculate correlation
    correlation = np.corrcoef(semantic_scores, severity_scores)[0, 1]
    
    ax.set_xlabel('Semantic Similarity Score', fontsize=12)
    ax.set_ylabel('Severity Accuracy Score', fontsize=12)
    ax.set_title(f'Semantic Similarity vs Severity Accuracy\n(Correlation: {correlation:.3f})', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'semantic_vs_severity_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_case_performance_heatmap(semantic_data, severity_data, output_dir):
    """Create heatmap of performance by case"""
    # Create matrix for heatmap
    cases = []
    semantic_scores = []
    severity_scores = []
    
    for sem_eval in semantic_data['evaluations']:
        case_id = sem_eval['case_id']
        cases.append(f"Case {case_id}")
        semantic_scores.append(sem_eval['best_match']['score'])
        
        # Find corresponding severity score
        for sev_eval in severity_data['evaluations']:
            if sev_eval['id'] == case_id:
                severity_scores.append(sev_eval['final_score'])
                break
    
    # Create DataFrame for heatmap
    data = pd.DataFrame({
        'Semantic': semantic_scores,
        'Severity': severity_scores
    }, index=cases)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(6, 15))
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', 
                cbar_kws={'label': 'Score'}, ax=ax, vmin=0, vmax=1)
    
    ax.set_title('Performance Heatmap by Case', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'case_performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_optimist_pessimist_balance(severity_data, output_dir):
    """Plot balance between optimist and pessimist evaluations"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    optimist_counts = []
    pessimist_counts = []
    case_ids = []
    
    for evaluation in severity_data['evaluations']:
        case_ids.append(evaluation['id'])
        optimist_counts.append(evaluation['optimist']['n'])
        pessimist_counts.append(evaluation['pessimist']['n'])
    
    # Plot 1: Stacked bar chart
    x = np.arange(len(case_ids))
    width = 0.8
    
    ax1.bar(x, optimist_counts, width, label='Optimist', color='blue', alpha=0.7)
    ax1.bar(x, pessimist_counts, width, bottom=optimist_counts, 
            label='Pessimist', color='red', alpha=0.7)
    
    ax1.set_xlabel('Case ID', fontsize=12)
    ax1.set_ylabel('Number of Evaluators', fontsize=12)
    ax1.set_title('Optimist vs Pessimist Evaluator Distribution', fontsize=14)
    ax1.set_xticks(x[::5])
    ax1.set_xticklabels(case_ids[::5])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Overall distribution pie chart
    total_optimist = sum(optimist_counts)
    total_pessimist = sum(pessimist_counts)
    
    ax2.pie([total_optimist, total_pessimist], labels=['Optimist', 'Pessimist'],
            colors=['blue', 'red'], autopct='%1.1f%%', alpha=0.7)
    ax2.set_title('Overall Optimist vs Pessimist Distribution', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'optimist_pessimist_balance.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate visualizations for Ramedis baseline experiment results')
    parser.add_argument('--run-dir', type=str, help='Specific run directory to visualize')
    args = parser.parse_args()
    
    # Find the most recent run directory if not specified
    results_dir = Path(__file__).parent / 'results'
    if args.run_dir:
        run_dir = results_dir / args.run_dir
    else:
        run_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('run_')])
        if not run_dirs:
            print("No run directories found!")
            return
        run_dir = run_dirs[-1]  # Get most recent
    
    print(f"Visualizing results from: {run_dir}")
    
    # Create plots directory
    plots_dir = run_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Load data
    semantic_file = run_dir / 'semantic_evaluation.json'
    severity_file = run_dir / 'severity_evaluation.json'
    
    if not semantic_file.exists() or not severity_file.exists():
        print("Required JSON files not found!")
        return
    
    semantic_data = load_json(semantic_file)
    severity_data = load_json(severity_file)
    
    # Generate all plots
    print("Generating semantic bias evaluation plot...")
    plot_semantic_bias_evaluation(semantic_data, plots_dir)
    
    print("Generating GDX vs DDX severity comparison plot...")
    plot_gdx_ddx_severity(severity_data, plots_dir)
    
    print("Generating semantic distribution plots...")
    plot_semantic_distribution(semantic_data, plots_dir)
    
    print("Generating severity distribution plots...")
    plot_severity_distribution(severity_data, plots_dir)
    
    print("Generating semantic vs severity correlation plot...")
    plot_semantic_vs_severity_correlation(semantic_data, severity_data, plots_dir)
    
    print("Generating case performance heatmap...")
    plot_case_performance_heatmap(semantic_data, severity_data, plots_dir)
    
    print("Generating optimist/pessimist balance plots...")
    plot_optimist_pessimist_balance(severity_data, plots_dir)
    
    print(f"\nAll visualizations saved to: {plots_dir}")

if __name__ == "__main__":
    main()