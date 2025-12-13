#!/usr/bin/env python3
"""
Visualize robustness evaluation results from JSON file.
Creates bar plots and summary statistics for different adaptation methods.

Usage:
    python visualize_robustness_results.py --input robustness_results_sev5.json --output_dir ./plots/robustness_analysis
"""

import os
import json
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    # Use matplotlib style instead
    plt.style.use('seaborn-v0_8-whitegrid' if hasattr(plt.style, 'seaborn-v0_8-whitegrid') else 'default')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Define corruption categories (ImageNet-C standard categories)
CORRUPTION_CATEGORIES = {
    'Noise': ['gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise'],
    'Blur': ['gaussian_blur', 'defocus_blur'],
    'Weather': ['fog', 'frost', 'brightness'],
    'Digital': ['jpeg_compression', 'contrast', 'pixelate', 'elastic_transform']
}

# Method display names (for better labels)
METHOD_DISPLAY_NAMES = {
    'unadapted': 'Unadapted',
    'tent': 'Tent',
    'eata': 'EATA',
    'sar': 'SAR',
    'prototta': 'ProtoTTA'
}

# Color palette for methods
METHOD_COLORS = {
    'unadapted': '#1f77b4',           # Blue
    'tent': '#ff7f0e',                 # Orange
    'eata': '#e377c2',                 # Pink
    'sar': '#7f7f7f',                  # Gray
    'prototta': '#2ca02c'              # Green
}


def load_results(json_file):
    """Load results from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def load_prototta_from_csv(csv_file, experiment_name, configuration, severity='5'):
    """Load ProtoTTA results from CSV file.
    
    Args:
        csv_file: Path to CSV file
        experiment_name: Name of experiment (e.g., "Experiment 1: Consensus Strategy")
        configuration: Configuration name (e.g., "top_k_mean")
        severity: Severity level (default: '5')
    
    Returns:
        Dictionary with corruption types as keys and accuracies as values (in decimal format)
    """
    results = {}
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row['Experiment'] == experiment_name and 
                row['Configuration'] == configuration):
                corruption = row['Corruption']
                # CSV has accuracy as percentage (e.g., 51.9676), convert to decimal
                accuracy = float(row['Accuracy']) / 100.0
                results[corruption] = {severity: accuracy}
    
    return results


def merge_results(json_file, csv_file, severity='5'):
    """Merge baseline results from JSON and ProtoTTA results from CSV.
    
    Args:
        json_file: Path to JSON file with baseline results
        csv_file: Path to CSV file with ProtoTTA results
        severity: Severity level to extract
    
    Returns:
        Dictionary with merged results in the same format as JSON results
    """
    # Load JSON results
    json_data = load_results(json_file)
    json_results = json_data['results']
    
    # Extract baseline methods (tent, eata, sar, normal)
    merged_results = {}
    
    # Rename 'normal' to 'unadapted' and add baseline methods
    baseline_methods = ['tent', 'eata', 'sar', 'normal']
    for method in baseline_methods:
        if method in json_results:
            if method == 'normal':
                merged_results['unadapted'] = json_results[method]
            else:
                merged_results[method] = json_results[method]
    
    # Load ProtoTTA results from CSV
    prototta_results = load_prototta_from_csv(
        csv_file, 
        experiment_name='Experiment 1: Consensus Strategy',
        configuration='top_k_mean',
        severity=severity
    )
    
    if prototta_results:
        merged_results['prototta'] = prototta_results
    
    return merged_results


def get_method_averages(results_dict, severity='5', exclude_list=None):
    """Calculate overall average accuracy for each method.
    
    Args:
        results_dict: Dictionary of results
        severity: Severity level to analyze
        exclude_list: List of corruption types to exclude (default: None)
    """
    if exclude_list is None:
        exclude_list = []
    
    method_averages = {}
    
    for method_name, corruptions in results_dict.items():
        accuracies = []
        for corruption_type, severities in corruptions.items():
            # Skip excluded corruptions
            if corruption_type in exclude_list:
                continue
            if severity in severities and severities[severity] is not None:
                accuracies.append(severities[severity])
        
        if accuracies:
            method_averages[method_name] = {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'min': np.min(accuracies),
                'max': np.max(accuracies),
                'count': len(accuracies)
            }
        else:
            method_averages[method_name] = None
    
    return method_averages


def get_category_averages(results_dict, severity='5', exclude_list=None):
    """Calculate average accuracy per category for each method.
    
    Args:
        results_dict: Dictionary of results
        severity: Severity level to analyze
        exclude_list: List of corruption types to exclude (default: None)
    """
    if exclude_list is None:
        exclude_list = []
    
    category_averages = defaultdict(dict)
    
    for method_name, corruptions in results_dict.items():
        for category, corruption_list in CORRUPTION_CATEGORIES.items():
            accuracies = []
            for corruption_type in corruption_list:
                # Skip excluded corruptions
                if corruption_type in exclude_list:
                    continue
                if corruption_type in corruptions:
                    if severity in corruptions[corruption_type]:
                        acc = corruptions[corruption_type][severity]
                        if acc is not None:
                            accuracies.append(acc)
            
            if accuracies:
                category_averages[method_name][category] = {
                    'mean': np.mean(accuracies),
                    'std': np.std(accuracies),
                    'count': len(accuracies)
                }
            else:
                category_averages[method_name][category] = None
    
    return dict(category_averages)


def plot_per_corruption_bar(results_dict, severity='5', output_dir='./plots', exclude_list=None):
    """Create bar plot showing performance per corruption type.
    
    Args:
        results_dict: Dictionary of results
        severity: Severity level to analyze
        output_dir: Directory to save plots
        exclude_list: List of corruption types to exclude (default: None)
    """
    if exclude_list is None:
        exclude_list = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all corruption types and methods
    all_corruptions = set()
    all_methods = list(results_dict.keys())
    
    for method_name, corruptions in results_dict.items():
        all_corruptions.update(corruptions.keys())
    
    # Filter out excluded corruptions
    all_corruptions = sorted([c for c in all_corruptions if c not in exclude_list])
    
    # Prepare data
    x = np.arange(len(all_corruptions))
    width = 0.8 / len(all_methods)
    
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # Plot bars for each method
    for i, method_name in enumerate(all_methods):
        accuracies = []
        for corruption in all_corruptions:
            if (corruption in results_dict[method_name] and 
                severity in results_dict[method_name][corruption]):
                acc = results_dict[method_name][corruption][severity]
                accuracies.append(acc * 100 if acc is not None else 0)
            else:
                accuracies.append(0)
        
        offset = (i - len(all_methods)/2 + 0.5) * width
        display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
        color = METHOD_COLORS.get(method_name, '#808080')
        
        ax.bar(x + offset, accuracies, width, 
               label=display_name, color=color, alpha=0.8)
    
    ax.set_xlabel('Corruption Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Across Different Corruptions (Severity 5)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in all_corruptions], 
                       rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'per_corruption_barplot.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved per-corruption bar plot to: {output_path}")


def plot_method_comparison_bar(method_averages, output_dir='./plots'):
    """Create bar plot comparing overall averages of all methods."""
    os.makedirs(output_dir, exist_ok=True)
    
    methods = []
    means = []
    stds = []
    colors_list = []
    
    for method_name, stats in method_averages.items():
        if stats is not None:
            methods.append(METHOD_DISPLAY_NAMES.get(method_name, method_name))
            means.append(stats['mean'] * 100)
            stds.append(stats['std'] * 100)
            colors_list.append(METHOD_COLORS.get(method_name, '#808080'))
    
    x = np.arange(len(methods))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(x, means, yerr=stds, capsize=5, 
                  color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for i, (bar, mean_val) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + stds[i] + 1,
                f'{mean_val:.2f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Adaptation Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Overall Performance Comparison (All Corruptions, Severity 5)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(means) + max(stds) + 5])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'method_comparison_barplot.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved method comparison bar plot to: {output_path}")


def plot_category_averages(category_averages, output_dir='./plots', exclude_list=None):
    """Create grouped bar plot showing performance per category.
    
    Args:
        category_averages: Dictionary of category averages
        output_dir: Directory to save plots
        exclude_list: List of corruption types to exclude (default: None)
    """
    if exclude_list is None:
        exclude_list = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter categories that have no valid corruptions (all excluded)
    categories = []
    for category, corruption_list in CORRUPTION_CATEGORIES.items():
        # Check if any corruption in this category is not excluded
        if any(c not in exclude_list for c in corruption_list):
            categories.append(category)
    
    methods = list(category_averages.keys())
    
    # Prepare data
    x = np.arange(len(categories))
    width = 0.8 / len(methods)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot bars for each method
    for i, method_name in enumerate(methods):
        means = []
        for category in categories:
            if (category in category_averages[method_name] and 
                category_averages[method_name][category] is not None):
                means.append(category_averages[method_name][category]['mean'] * 100)
            else:
                means.append(0)
        
        offset = (i - len(methods)/2 + 0.5) * width
        display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
        color = METHOD_COLORS.get(method_name, '#808080')
        
        ax.bar(x + offset, means, width, 
               label=display_name, color=color, alpha=0.8)
    
    ax.set_xlabel('Corruption Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance by Corruption Category (Severity 5)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'category_averages_barplot.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved category averages bar plot to: {output_path}")


def plot_category_heatmap(category_averages, output_dir='./plots', exclude_list=None):
    """Create heatmap showing performance per category for each method.
    
    Args:
        category_averages: Dictionary of category averages
        output_dir: Directory to save plots
        exclude_list: List of corruption types to exclude (default: None)
    """
    if exclude_list is None:
        exclude_list = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter categories that have no valid corruptions (all excluded)
    categories = []
    for category, corruption_list in CORRUPTION_CATEGORIES.items():
        # Check if any corruption in this category is not excluded
        if any(c not in exclude_list for c in corruption_list):
            categories.append(category)
    
    methods = list(category_averages.keys())
    
    # Prepare data matrix
    data_matrix = []
    method_labels = []
    
    for method_name in methods:
        row = []
        for category in categories:
            if (category in category_averages[method_name] and 
                category_averages[method_name][category] is not None):
                row.append(category_averages[method_name][category]['mean'] * 100)
            else:
                row.append(np.nan)
        data_matrix.append(row)
        method_labels.append(METHOD_DISPLAY_NAMES.get(method_name, method_name))
    
    data_matrix = np.array(data_matrix)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Set ticks
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(method_labels)))
    ax.set_xticklabels(categories)
    ax.set_yticklabels(method_labels)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(method_labels)):
        for j in range(len(categories)):
            if not np.isnan(data_matrix[i, j]):
                text = ax.text(j, i, f'{data_matrix[i, j]:.1f}%',
                             ha="center", va="center", color="black", fontsize=9, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy (%)', rotation=270, labelpad=20, fontsize=11)
    
    ax.set_title('Performance Heatmap by Category and Method (Severity 5)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'category_heatmap.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved category heatmap to: {output_path}")


def print_summary_table(method_averages, category_averages, exclude_list=None):
    """Print summary statistics to console.
    
    Args:
        method_averages: Dictionary of method averages
        category_averages: Dictionary of category averages
        exclude_list: List of corruption types to exclude (default: None)
    """
    if exclude_list is None:
        exclude_list = []
    
    if exclude_list:
        print(f"\nNOTE: Excluding corruptions: {', '.join(exclude_list)}")
    
    print("\n" + "="*80)
    print("OVERALL PERFORMANCE SUMMARY")
    print("="*80)
    print(f"{'Method':<30s} {'Mean Acc (%)':>12s} {'Std (%)':>10s} {'Min (%)':>10s} {'Max (%)':>10s}")
    print("-"*80)
    
    # Sort by mean accuracy (descending)
    sorted_methods = sorted(method_averages.items(), 
                           key=lambda x: x[1]['mean'] if x[1] else 0, 
                           reverse=True)
    
    for method_name, stats in sorted_methods:
        if stats is not None:
            display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
            print(f"{display_name:<30s} {stats['mean']*100:11.2f}% "
                  f"{stats['std']*100:9.2f}% {stats['min']*100:9.2f}% "
                  f"{stats['max']*100:9.2f}%")
    
    print("\n" + "="*80)
    print("PER-CATEGORY PERFORMANCE SUMMARY")
    print("="*80)
    
    # Filter categories that have no valid corruptions (all excluded)
    categories = []
    for category, corruption_list in CORRUPTION_CATEGORIES.items():
        # Check if any corruption in this category is not excluded
        if any(c not in exclude_list for c in corruption_list):
            categories.append(category)
    
    methods = list(category_averages.keys())
    
    # Header
    header = f"{'Method':<30s}"
    for category in categories:
        header += f" {category:>12s}"
    print(header)
    print("-"*80)
    
    # Data rows
    for method_name in methods:
        display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
        row = f"{display_name:<30s}"
        for category in categories:
            if (category in category_averages[method_name] and 
                category_averages[method_name][category] is not None):
                mean_acc = category_averages[method_name][category]['mean'] * 100
                row += f" {mean_acc:11.2f}%"
            else:
                row += f" {'N/A':>12s}"
        print(row)
    
    print("="*80)


def print_per_corruption_table(results_dict, severity='5', exclude_list=None):
    """Print per-corruption performance table.
    
    Args:
        results_dict: Dictionary of results
        severity: Severity level to analyze
        exclude_list: List of corruption types to exclude (default: None)
    """
    if exclude_list is None:
        exclude_list = []
    
    # Get all corruption types and methods
    all_corruptions = set()
    all_methods = list(results_dict.keys())
    
    for method_name, corruptions in results_dict.items():
        all_corruptions.update(corruptions.keys())
    
    # Filter out excluded corruptions
    all_corruptions = sorted([c for c in all_corruptions if c not in exclude_list])
    
    if not all_corruptions:
        print("\nNo corruptions to display (all excluded).")
        return
    
    print("\n" + "="*100)
    print("PER-CORRUPTION PERFORMANCE SUMMARY")
    print("="*100)
    
    # Header
    header = f"{'Corruption':<25s}"
    for method_name in all_methods:
        display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
        # Truncate if too long
        if len(display_name) > 12:
            display_name = display_name[:9] + "..."
        header += f" {display_name:>12s}"
    print(header)
    print("-"*100)
    
    # Data rows
    for corruption in all_corruptions:
        # Format corruption name nicely
        corruption_display = corruption.replace('_', ' ').title()
        if len(corruption_display) > 24:
            corruption_display = corruption_display[:21] + "..."
        
        row = f"{corruption_display:<25s}"
        for method_name in all_methods:
            if (corruption in results_dict[method_name] and 
                severity in results_dict[method_name][corruption]):
                acc = results_dict[method_name][corruption][severity]
                if acc is not None:
                    row += f" {acc*100:11.2f}%"
                else:
                    row += f" {'N/A':>12s}"
            else:
                row += f" {'N/A':>12s}"
        print(row)
    
    print("="*100)


def save_summary_json(method_averages, category_averages, output_file, exclude_list=None):
    """Save summary statistics to JSON file.
    
    Args:
        method_averages: Dictionary of method averages
        category_averages: Dictionary of category averages
        output_file: Path to save JSON file
        exclude_list: List of corruption types to exclude (default: None)
    """
    if exclude_list is None:
        exclude_list = []
    
    summary = {
        'excluded_corruptions': exclude_list,
        'overall_averages': {},
        'category_averages': {}
    }
    
    for method_name, stats in method_averages.items():
        if stats is not None:
            summary['overall_averages'][method_name] = {
                'mean': float(stats['mean'] * 100),
                'std': float(stats['std'] * 100),
                'min': float(stats['min'] * 100),
                'max': float(stats['max'] * 100),
                'count': stats['count']
            }
    
    for method_name, categories in category_averages.items():
        summary['category_averages'][method_name] = {}
        for category, stats in categories.items():
            if stats is not None:
                summary['category_averages'][method_name][category] = {
                    'mean': float(stats['mean'] * 100),
                    'std': float(stats['std'] * 100),
                    'count': stats['count']
                }
            else:
                summary['category_averages'][method_name][category] = None
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved summary statistics to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize robustness evaluation results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input JSON results file (baseline methods)')
    parser.add_argument('--csv_input', type=str, 
                       default='./ablation_studies/visualizations/detailed_results.csv',
                       help='Path to CSV file with ProtoTTA results')
    parser.add_argument('--output_dir', type=str, default='./plots/robustness_analysis',
                       help='Directory to save output plots')
    parser.add_argument('--severity', type=str, default='5',
                       help='Severity level to analyze (default: 5)')
    parser.add_argument('--save_summary', type=str, default=None,
                       help='Path to save summary JSON (optional)')
    parser.add_argument('--exclude', nargs='+', default=[],
                       help='Corruption types to exclude from analysis (e.g., --exclude spatter saturate)')
    
    args = parser.parse_args()
    
    # Process exclude list
    exclude_list = [c.strip().lower() for c in args.exclude] if args.exclude else []
    
    if exclude_list:
        print(f"Excluding corruptions: {', '.join(exclude_list)}")
    
    # Load and merge results
    print(f"Loading baseline results from: {args.input}")
    print(f"Loading ProtoTTA results from: {args.csv_input}")
    results_dict = merge_results(args.input, args.csv_input, severity=args.severity)
    print(f"Loaded {len(results_dict)} methods: {list(results_dict.keys())}")
    
    # Calculate statistics
    print("Calculating statistics...")
    method_averages = get_method_averages(results_dict, severity=args.severity, exclude_list=exclude_list)
    category_averages = get_category_averages(results_dict, severity=args.severity, exclude_list=exclude_list)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_per_corruption_bar(results_dict, severity=args.severity, output_dir=args.output_dir, exclude_list=exclude_list)
    plot_method_comparison_bar(method_averages, output_dir=args.output_dir)
    plot_category_averages(category_averages, output_dir=args.output_dir, exclude_list=exclude_list)
    plot_category_heatmap(category_averages, output_dir=args.output_dir, exclude_list=exclude_list)
    
    # Print summary
    print_summary_table(method_averages, category_averages, exclude_list=exclude_list)
    print_per_corruption_table(results_dict, severity=args.severity, exclude_list=exclude_list)
    
    # Save summary if requested
    if args.save_summary:
        save_summary_json(method_averages, category_averages, args.save_summary, exclude_list=exclude_list)
    
    print(f"\nAll visualizations saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

