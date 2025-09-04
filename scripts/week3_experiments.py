#!/usr/bin/env python3
"""
Week 3 Experiment Pipeline
Automated multi-dataset, multi-model evaluation for low-rank SHAP.

Usage:
    python scripts/week3_experiments.py --datasets all --models all --ranks 10,30,50
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lowrank_shap.lowrank_shap import LowRankSHAP, benchmark_comparison
from lowrank_shap.baseline import KernelSHAPBaseline
from lowrank_shap.data_utils import load_wine_quality, load_bike_sharing, load_adult, load_compas

class ExperimentRunner:
    """Automated experiment runner for Week 3."""
    
    def __init__(self, results_dir: str = "../results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Define datasets and models
        self.datasets = {
            'wine': load_wine_quality,
            'bike': load_bike_sharing,
            'adult': load_adult,
            'compas': load_compas
        }
        
        self.models = {
            'logreg': lambda: LogisticRegression(random_state=42, max_iter=1000),
            'rf': lambda: RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': lambda: SVC(kernel='rbf', random_state=42, probability=True),
            'mlp': lambda: MLPClassifier(random_state=42, max_iter=1000)
        }
        
        self.results = []
    
    def run_dataset_model_experiment(self, dataset_name: str, model_name: str, 
                                   ranks: List[int] = [10, 30, 50]) -> Dict[str, Any]:
        """Run complete experiment for a dataset-model pair."""
        
        print(f"\n{'='*60}")
        print(f"Running: {dataset_name} × {model_name}")
        print(f"{'='*60}")
        
        # Load data
        try:
            print(f"Loading {dataset_name} dataset...")
            loader = self.datasets[dataset_name]
            X, y, _ = loader()  # Unpack 3 values: X, y, columns
            print(f"  Loaded: {X.shape[0]} samples, {X.shape[1]} features")
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features for certain models
        if model_name in ['logreg', 'svm', 'mlp']:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        # Train model
        model = self.models[model_name]()
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model accuracy: {accuracy:.3f}")
        print(f"Training time: {training_time:.2f}s")
        
        # Prepare for SHAP experiments
        # Use smaller subsets for computational efficiency
        background_size = min(100, len(X_train))
        test_size = min(20, len(X_test))
        
        X_background = X_train[:background_size]
        X_test_small = X_test[:test_size]
        
        # Run benchmark comparison
        results = {
            'dataset': dataset_name,
            'model': model_name,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'accuracy': accuracy,
            'training_time': training_time,
            'benchmark_results': {}
        }
        
        try:
            benchmark_results = benchmark_comparison(
                model, X_background, X_test_small, ranks=ranks
            )
            results['benchmark_results'] = benchmark_results
            
            # Extract key metrics
            exact_runtime = benchmark_results['exact']['metadata']['total_runtime']
            exact_memory = benchmark_results['exact']['metadata']['max_memory']
            
            for rank in ranks:
                if str(rank) in benchmark_results['low_rank']:
                    lr_results = benchmark_results['low_rank'][str(rank)]
                    speedup = exact_runtime / lr_results['metadata']['total_runtime']
                    memory_ratio = exact_memory / lr_results['metadata']['max_memory']
                    
                    results[f'rank_{rank}_speedup'] = speedup
                    results[f'rank_{rank}_memory_ratio'] = memory_ratio
                    results[f'rank_{rank}_error'] = lr_results['mean_relative_error']
            
            print(f"✓ Benchmark completed successfully")
            
        except Exception as e:
            print(f"✗ Error in benchmark: {e}")
            results['error'] = str(e)
        
        return results
    
    def run_all_experiments(self, datasets: List[str], models: List[str], 
                          ranks: List[int] = [10, 30, 50]) -> pd.DataFrame:
        """Run all experiments across datasets and models."""
        
        print(f"\n{'='*80}")
        print("WEEK 3 EXPERIMENT PIPELINE")
        print(f"{'='*80}")
        print(f"Datasets: {datasets}")
        print(f"Models: {models}")
        print(f"Ranks: {ranks}")
        print(f"Total experiments: {len(datasets) * len(models)}")
        print(f"{'='*80}")
        
        start_time = time.time()
        all_results = []
        
        for dataset_name in datasets:
            for model_name in models:
                result = self.run_dataset_model_experiment(
                    dataset_name, model_name, ranks
                )
                if result:
                    all_results.append(result)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df = pd.DataFrame(all_results)
        
        # Save detailed results
        results_file = f"{self.results_dir}/week3_experiments_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        
        # Save detailed JSON with benchmark data
        json_file = f"{self.results_dir}/week3_detailed_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print("EXPERIMENTS COMPLETE")
        print(f"{'='*80}")
        print(f"Total experiments run: {len(all_results)}")
        print(f"Total runtime: {total_time:.2f}s ({total_time/60:.1f} min)")
        print(f"Results saved to:")
        print(f"  CSV: {results_file}")
        print(f"  JSON: {json_file}")
        print(f"{'='*80}")
        
        return results_df
    
    def create_summary_report(self, results_df: pd.DataFrame) -> str:
        """Create a markdown summary report."""
        report = []
        report.append("# Week 3 Experiment Summary")
        report.append("")
        
        if results_df.empty:
            report.append("## ⚠️ No Successful Experiments")
            report.append("All experiments failed to complete. Check the error messages above.")
            report.append("")
            return "\n".join(report)
        
        # Overall statistics
        report.append("## Overall Statistics")
        report.append(f"- Total experiments: {len(results_df)}")
        report.append(f"- Datasets: {results_df['dataset'].nunique()}")
        report.append(f"- Models: {results_df['model'].nunique()}")
        report.append(f"- Ranks tested: [5, 10, 20]")
        report.append("")
        
        # Performance summary
        report.append("## Performance Summary")
        # Extract performance metrics from benchmark_results JSON
        try:
            import json
            speedups = []
            memory_ratios = []
            errors = []
            
            for _, row in results_df.iterrows():
                benchmark_data = json.loads(row['benchmark_results'].replace("'", '"'))
                if 'low_rank' in benchmark_data:
                    for rank in [5, 10, 20]:
                        if rank in benchmark_data['low_rank']:
                            lr_data = benchmark_data['low_rank'][rank]
                            exact_data = benchmark_data['exact']
                            
                            # Calculate speedup
                            if 'avg_runtime' in lr_data and 'avg_runtime' in exact_data:
                                speedup = exact_data['avg_runtime'] / lr_data['avg_runtime']
                                speedups.append(speedup)
                            
                            # Calculate memory ratio
                            if 'max_memory' in lr_data and 'max_memory' in exact_data:
                                memory_ratio = lr_data['max_memory'] / exact_data['max_memory']
                                memory_ratios.append(memory_ratio)
                            
                            # Get error
                            if 'mean_relative_error' in lr_data:
                                errors.append(lr_data['mean_relative_error'])
            
            avg_speedup = sum(speedups) / len(speedups) if speedups else 0
            avg_memory_ratio = sum(memory_ratios) / len(memory_ratios) if memory_ratios else 0
            avg_error = sum(errors) / len(errors) if errors else 0
            
        except Exception as e:
            print(f"Warning: Could not parse performance metrics: {e}")
            avg_speedup = avg_memory_ratio = avg_error = 0
        
        report.append(f"- Average speedup: {avg_speedup:.2f}x")
        report.append(f"- Average memory reduction: {avg_memory_ratio:.2f}x")
        report.append(f"- Average relative error: {avg_error:.4f}")
        report.append("")
        
        # Dataset performance summary
        report.append("## Dataset Performance Summary")
        for dataset in results_df['dataset'].unique():
            dataset_data = results_df[results_df['dataset'] == dataset]
            report.append(f"\n### {dataset.title()} Dataset")
            report.append(f"- Samples: {dataset_data['n_samples'].iloc[0]:,}")
            report.append(f"- Features: {dataset_data['n_features'].iloc[0]}")
            report.append(f"- Models tested: {', '.join(dataset_data['model'].unique())}")
            avg_acc = dataset_data['accuracy'].mean()
            report.append(f"- Average model accuracy: {avg_acc:.3f}")
        
        report.append("")
        
        return "\n".join(report)

def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description="Week 3 Experiment Pipeline")
    parser.add_argument("--datasets", default="all", 
                       help="Comma-separated datasets or 'all'")
    parser.add_argument("--models", default="all",
                       help="Comma-separated models or 'all'")
    parser.add_argument("--ranks", default="10,30,50",
                       help="Comma-separated ranks to test")
    parser.add_argument("--results-dir", default="../results",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    # Parse arguments
    runner = ExperimentRunner(args.results_dir)
    
    datasets = list(runner.datasets.keys()) if args.datasets == "all" else args.datasets.split(",")
    models = list(runner.models.keys()) if args.models == "all" else args.models.split(",")
    ranks = [int(r) for r in args.ranks.split(",")]
    
    # Validate inputs
    invalid_datasets = [d for d in datasets if d not in runner.datasets]
    invalid_models = [m for m in models if m not in runner.models]
    
    if invalid_datasets:
        print(f"Invalid datasets: {invalid_datasets}")
        return
    if invalid_models:
        print(f"Invalid models: {invalid_models}")
        return
    
    # Run experiments
    results_df = runner.run_all_experiments(datasets, models, ranks)
    
    # Generate summary report
    report = runner.create_summary_report(results_df)
    report_file = f"{args.results_dir}/week3_summary.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nSummary report saved to: {report_file}")

if __name__ == "__main__":
    main()
