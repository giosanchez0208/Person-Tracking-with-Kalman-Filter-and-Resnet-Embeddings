"""
Visualization and Analysis for MOT17 Benchmark Results
Generates comprehensive plots and reports.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path


def load_results(results_file):
    """Load benchmark results from JSON."""
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_metrics_comparison(results, output_dir='benchmark_results'):
    """Create bar plots comparing metrics across sequences."""
    sequences = results['sequences']
    seq_names = list(sequences.keys())
    
    # Extract metrics
    metrics_to_plot = ['MOTA', 'MOTP', 'Precision', 'Recall', 'F1']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # MOTA
    ax = axes[0]
    mota_values = [sequences[s]['MOTA'] for s in seq_names]
    bars = ax.bar(range(len(seq_names)), mota_values, color='steelblue', alpha=0.8)
    ax.axhline(y=results['overall']['MOTA'], color='red', linestyle='--', linewidth=2, label='Average')
    ax.set_ylabel('MOTA (%)', fontsize=12)
    ax.set_title('Multiple Object Tracking Accuracy (MOTA)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(seq_names)))
    ax.set_xticklabels([s.split('-')[1] for s in seq_names], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # MOTP
    ax = axes[1]
    motp_values = [sequences[s]['MOTP'] for s in seq_names]
    bars = ax.bar(range(len(seq_names)), motp_values, color='coral', alpha=0.8)
    ax.axhline(y=results['overall']['MOTP'], color='red', linestyle='--', linewidth=2, label='Average')
    ax.set_ylabel('MOTP (pixels)', fontsize=12)
    ax.set_title('Multiple Object Tracking Precision (MOTP)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(seq_names)))
    ax.set_xticklabels([s.split('-')[1] for s in seq_names], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Precision
    ax = axes[2]
    precision_values = [sequences[s]['Precision'] for s in seq_names]
    bars = ax.bar(range(len(seq_names)), precision_values, color='lightgreen', alpha=0.8)
    ax.axhline(y=results['overall']['Precision'], color='red', linestyle='--', linewidth=2, label='Average')
    ax.set_ylabel('Precision (%)', fontsize=12)
    ax.set_title('Detection Precision', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(seq_names)))
    ax.set_xticklabels([s.split('-')[1] for s in seq_names], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Recall
    ax = axes[3]
    recall_values = [sequences[s]['Recall'] for s in seq_names]
    bars = ax.bar(range(len(seq_names)), recall_values, color='plum', alpha=0.8)
    ax.axhline(y=results['overall']['Recall'], color='red', linestyle='--', linewidth=2, label='Average')
    ax.set_ylabel('Recall (%)', fontsize=12)
    ax.set_title('Detection Recall', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(seq_names)))
    ax.set_xticklabels([s.split('-')[1] for s in seq_names], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # F1 Score
    ax = axes[4]
    f1_values = [sequences[s]['F1'] for s in seq_names]
    bars = ax.bar(range(len(seq_names)), f1_values, color='gold', alpha=0.8)
    ax.axhline(y=results['overall']['F1'], color='red', linestyle='--', linewidth=2, label='Average')
    ax.set_ylabel('F1 Score (%)', fontsize=12)
    ax.set_title('F1 Score', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(seq_names)))
    ax.set_xticklabels([s.split('-')[1] for s in seq_names], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # ID Switches
    ax = axes[5]
    id_switches = [sequences[s]['ID_Switches'] for s in seq_names]
    bars = ax.bar(range(len(seq_names)), id_switches, color='salmon', alpha=0.8)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('ID Switches (Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(seq_names)))
    ax.set_xticklabels([s.split('-')[1] for s in seq_names], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'metrics_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved metrics comparison plot to: {output_path}")
    plt.close()


def plot_tracking_quality(results, output_dir='benchmark_results'):
    """Plot tracking quality metrics (ID switches, fragmentations)."""
    sequences = results['sequences']
    seq_names = list(sequences.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ID Switches
    ax = axes[0]
    id_switches = [sequences[s]['ID_Switches'] for s in seq_names]
    bars = ax.barh(range(len(seq_names)), id_switches, color='crimson', alpha=0.7)
    ax.set_xlabel('Number of ID Switches', fontsize=12)
    ax.set_title('ID Switches per Sequence', fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(seq_names)))
    ax.set_yticklabels([s.split('-')[1] for s in seq_names])
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Add values on bars
    for i, v in enumerate(id_switches):
        ax.text(v + 0.5, i, str(v), va='center', fontsize=10)
    
    # Fragmentations
    ax = axes[1]
    fragmentations = [sequences[s]['Fragmentations'] for s in seq_names]
    bars = ax.barh(range(len(seq_names)), fragmentations, color='orange', alpha=0.7)
    ax.set_xlabel('Number of Fragmentations', fontsize=12)
    ax.set_title('Track Fragmentations per Sequence', fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(seq_names)))
    ax.set_yticklabels([s.split('-')[1] for s in seq_names])
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Add values on bars
    for i, v in enumerate(fragmentations):
        ax.text(v + 0.5, i, str(v), va='center', fontsize=10)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'tracking_quality.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved tracking quality plot to: {output_path}")
    plt.close()


def plot_detection_errors(results, output_dir='benchmark_results'):
    """Plot false positives and misses."""
    sequences = results['sequences']
    seq_names = list(sequences.keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(seq_names))
    width = 0.35
    
    false_positives = [sequences[s]['False_Positives'] for s in seq_names]
    misses = [sequences[s]['Misses'] for s in seq_names]
    
    bars1 = ax.bar(x - width/2, false_positives, width, label='False Positives', color='red', alpha=0.7)
    bars2 = ax.bar(x + width/2, misses, width, label='Misses', color='blue', alpha=0.7)
    
    ax.set_xlabel('Sequence', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Detection Errors: False Positives vs Misses', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.split('-')[1] for s in seq_names], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'detection_errors.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved detection errors plot to: {output_path}")
    plt.close()


def plot_fps_performance(results, output_dir='benchmark_results'):
    """Plot FPS performance across sequences."""
    sequences = results['sequences']
    seq_names = list(sequences.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # FPS bar chart
    ax = axes[0]
    fps_values = [sequences[s]['Avg_FPS'] for s in seq_names]
    bars = ax.bar(range(len(seq_names)), fps_values, color='teal', alpha=0.8)
    ax.axhline(y=results['overall']['Avg_FPS'], color='red', linestyle='--', linewidth=2, label='Average')
    ax.set_ylabel('FPS', fontsize=12)
    ax.set_title('Average FPS per Sequence', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(seq_names)))
    ax.set_xticklabels([s.split('-')[1] for s in seq_names], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Frame time
    ax = axes[1]
    frame_times = [sequences[s]['Avg_Frame_Time_ms'] for s in seq_names]
    bars = ax.bar(range(len(seq_names)), frame_times, color='purple', alpha=0.8)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Average Frame Processing Time', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(seq_names)))
    ax.set_xticklabels([s.split('-')[1] for s in seq_names], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'fps_performance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved FPS performance plot to: {output_path}")
    plt.close()


def plot_overall_summary(results, output_dir='benchmark_results'):
    """Create a summary dashboard with key metrics."""
    overall = results['overall']
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    # Large MOTA display
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.axis('off')
    ax1.text(0.5, 0.5, f"{overall['MOTA']:.2f}%", 
             ha='center', va='center', fontsize=72, fontweight='bold', color='steelblue')
    ax1.text(0.5, 0.15, "MOTA", ha='center', va='center', fontsize=24, color='gray')
    
    # Key metrics boxes
    metrics_data = [
        ('MOTP', f"{overall['MOTP']:.2f}px", 'coral'),
        ('Precision', f"{overall['Precision']:.2f}%", 'lightgreen'),
        ('Recall', f"{overall['Recall']:.2f}%", 'plum'),
        ('F1 Score', f"{overall['F1']:.2f}%", 'gold'),
        ('Avg FPS', f"{overall['Avg_FPS']:.2f}", 'teal'),
    ]
    
    positions = [
        gs[0, 2], gs[1, 0], gs[1, 1], gs[1, 2], gs[2, 0]
    ]
    
    for (metric_name, value, color), pos in zip(metrics_data, positions):
        ax = fig.add_subplot(pos)
        ax.axis('off')
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.3))
        ax.text(0.5, 0.6, value, ha='center', va='center', fontsize=28, fontweight='bold')
        ax.text(0.5, 0.25, metric_name, ha='center', va='center', fontsize=14, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    # Error statistics
    ax_errors = fig.add_subplot(gs[2, 1:])
    error_data = [
        overall['ID_Switches'],
        overall['Fragmentations'],
        overall['False_Positives'],
        overall['Misses']
    ]
    error_labels = ['ID\nSwitches', 'Fragmen-\ntations', 'False\nPositives', 'Misses']
    colors = ['crimson', 'orange', 'red', 'blue']
    
    bars = ax_errors.bar(error_labels, error_data, color=colors, alpha=0.7)
    ax_errors.set_ylabel('Count', fontsize=11)
    ax_errors.set_title('Error Statistics', fontsize=12, fontweight='bold')
    ax_errors.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        ax_errors.text(bar.get_x() + bar.get_width()/2., height,
                      f'{int(height)}',
                      ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('MOT17 Benchmark - Overall Performance Summary', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    output_path = Path(output_dir) / 'overall_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved overall summary to: {output_path}")
    plt.close()


def create_results_table(results, output_dir='benchmark_results'):
    """Create a detailed table of results."""
    sequences = results['sequences']
    
    # Build dataframe
    data = []
    for seq_name, metrics in sequences.items():
        data.append({
            'Sequence': seq_name.split('-')[1],
            'MOTA (%)': f"{metrics['MOTA']:.2f}",
            'MOTP (px)': f"{metrics['MOTP']:.2f}",
            'Precision (%)': f"{metrics['Precision']:.2f}",
            'Recall (%)': f"{metrics['Recall']:.2f}",
            'F1 (%)': f"{metrics['F1']:.2f}",
            'ID Switches': metrics['ID_Switches'],
            'Fragmentations': metrics['Fragmentations'],
            'FPS': f"{metrics['Avg_FPS']:.2f}"
        })
    
    df = pd.DataFrame(data)
    
    # Add overall row
    overall = results['overall']
    overall_row = {
        'Sequence': 'OVERALL',
        'MOTA (%)': f"{overall['MOTA']:.2f}",
        'MOTP (px)': f"{overall['MOTP']:.2f}",
        'Precision (%)': f"{overall['Precision']:.2f}",
        'Recall (%)': f"{overall['Recall']:.2f}",
        'F1 (%)': f"{overall['F1']:.2f}",
        'ID Switches': overall['ID_Switches'],
        'Fragmentations': overall['Fragmentations'],
        'FPS': f"{overall['Avg_FPS']:.2f}"
    }
    df = pd.concat([df, pd.DataFrame([overall_row])], ignore_index=True)
    
    # Save as CSV
    output_path = Path(output_dir) / 'results_table.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved results table to: {output_path}")
    
    # Create figure table
    fig, ax = plt.subplots(figsize=(14, len(df) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center',
                     colColours=['lightgray'] * len(df.columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Highlight overall row
    for i in range(len(df.columns)):
        table[(len(df), i)].set_facecolor('#FFD700')
        table[(len(df), i)].set_text_props(weight='bold')
    
    plt.title('MOT17 Benchmark Results - Detailed Table', 
              fontsize=16, fontweight='bold', pad=20)
    
    output_path = Path(output_dir) / 'results_table.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved results table image to: {output_path}")
    plt.close()
    
    return df


def generate_all_visualizations(results_file, output_dir='benchmark_results'):
    """Generate all plots and reports."""
    print(f"\nGenerating visualizations from: {results_file}")
    
    results = load_results(results_file)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("Creating visualizations...")
    print("="*60)
    
    plot_overall_summary(results, output_dir)
    plot_metrics_comparison(results, output_dir)
    plot_tracking_quality(results, output_dir)
    plot_detection_errors(results, output_dir)
    plot_fps_performance(results, output_dir)
    create_results_table(results, output_dir)
    
    print("\n" + "="*60)
    print("All visualizations generated successfully!")
    print(f"Output directory: {output_path.absolute()}")
    print("="*60 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate MOT17 benchmark visualizations')
    parser.add_argument('--results-file', type=str, required=True,
                        help='Path to results JSON file')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                        help='Output directory for plots')
    
    args = parser.parse_args()
    
    generate_all_visualizations(args.results_file, args.output_dir)
