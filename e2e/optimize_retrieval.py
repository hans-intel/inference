#!/usr/bin/env python3
"""
Universal Retrieval Parameter Optimization using Optuna

This script systematically explores retrieval parameters to find optimal configurations
for both BM25 and Vector retrieval methods using Bayesian optimization.

Supports optimizing:
- BM25 parameters (k1, b, method, delta, stemmer, stopwords)
- Vector parameters (index_method, ivf_nprobe)
- Retrieval strategy parameters (top_p, relative_ratio)
- Reranking parameters (top_k_retriever, top_k_reranking)
"""

import optuna
import subprocess
import os
import sys
import shutil
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse

# Import centralized parameter definitions
from params import (
    suggest_param,
    format_params_for_cli,
    get_optimizable_params,
    get_default_params,
    PARAM_BY_NAME,
    OPTIMIZABLE_PARAM_NAMES
)

def get_default_threads() -> int:
    """
    Get default number of threads: half of available cores, 
    rounded down to nearest smaller power of 2.
    """
    import os
    cpu_count = os.cpu_count() or 4
    half_cores = cpu_count // 2
    
    if half_cores <= 0:
        return 1
    
    # Find nearest smaller or equal power of 2
    power = 1
    while power * 2 <= half_cores:
        power *= 2
    
    return max(1, power)

class RetrievalOptimizer:
    def __init__(self, 
                 retrieval_method: str = "vector",
                 ingest_path: str = "passages/test_small.json",
                 eval_queries: int = 100,
                 dataset: str = "data/frames_dataset.tsv",
                 output_dir: str = "optimization_results",
                 num_threads: int = None,
                 metric: str = "f1@10",
                 device: str = "auto",
                 optimize_params: Optional[List[str]] = None,
                 fixed_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the retrieval optimizer.
        
        Args:
            retrieval_method: 'bm25' or 'vector'
            ingest_path: Path to passages JSON file
            eval_queries: Number of queries to evaluate
            dataset: Path to evaluation dataset
            output_dir: Directory for optimization results
            num_threads: Number of threads for BM25 (None = auto)
            metric: Metric to optimize (e.g., 'f1@10', 'precision@10')
            device: Device for vector retrieval ('auto', 'xpu', 'cuda', 'cpu')
            optimize_params: List of parameters to optimize (None = use defaults)
            fixed_params: Parameters not in optimize_params will be fixed at these values
        """
        self.retrieval_method = retrieval_method
        self.ingest_path = ingest_path
        self.eval_queries = eval_queries
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.metric = metric
        self.device = device
        self.num_threads = num_threads if num_threads is not None else get_default_threads()
        self.fixed_params = fixed_params or {}
        
        # Set default parameters to optimize based on retrieval method
        if optimize_params is None:
            if retrieval_method == "bm25":
                self.optimize_params = ["bm25_k1", "bm25_b", "bm25_method", "bm25_stemmer"]
            elif retrieval_method == "vector":
                self.optimize_params = ["retrieval_strategy", "top_p", "top_k_retriever"]
            else:
                raise ValueError(f"Unknown retrieval method: {retrieval_method}")
        else:
            self.optimize_params = optimize_params
        
        # Validate parameters exist
        for param_name in self.optimize_params:
            if param_name not in PARAM_BY_NAME:
                raise ValueError(f"Unknown parameter: {param_name}. Available: {', '.join(OPTIMIZABLE_PARAM_NAMES)}")
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        print(f"Optimization outputs will be saved to: {self.output_dir}")
        print(f"Retrieval method: {retrieval_method.upper()}")
        print(f"Optimizing parameters: {', '.join(self.optimize_params)}")
        print(f"Fixed parameters: {self.fixed_params}")
        print(f"Target metric: {self.metric}")
        
        # Set up file paths
        self.results_log = self.output_dir / "optimization_results.json"
        self.optuna_db = self.output_dir / "optuna_study.db"
        self.best_params_file = self.output_dir / "best_parameters.json"
        
    def cleanup_trial_files(self, db_name: str):
        """Clean up any leftover files from previous trial."""
        files_to_clean = [
            f"{db_name}.db",
            f"{db_name}.json", 
            "results.json"
        ]
        
        data_dir = f"{db_name}_data"
        
        for file_path in files_to_clean:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass
        
        if os.path.exists(data_dir):
            try:
                shutil.rmtree(data_dir)
            except Exception:
                pass
    
    def suggest_param_value(self, trial, param_name: str) -> Any:
        """Suggest parameter value using centralized definitions."""
        return suggest_param(trial, param_name)
    
    def build_command(self, trial_id: int, params: Dict[str, Any]) -> tuple:
        """Build command with parameters using centralized formatting."""
        timestamp = int(time.time() * 1000000)
        db_name = f"{self.retrieval_method}_trial_{timestamp}"
        
        # Get the directory where this script is located
        script_dir = Path(__file__).parent.absolute()
        retrieval_script = script_dir / "single_shot_retrieval.py"
        
        # Build base command with absolute path to script
        cmd = ["python3", str(retrieval_script)]
        
        # Add all parameters using centralized formatting (skip defaults to reduce clutter)
        param_args = format_params_for_cli(params, skip_defaults=True)
        cmd.extend(param_args)
        
        return cmd, db_name
    
    def objective(self, trial):
        """Optuna objective function to maximize the specified metric."""
        trial_id = trial.number
        
        try:
            # Start with framework parameters
            timestamp = int(time.time() * 1000000)
            db_name = f"{self.retrieval_method}_trial_{timestamp}"
            
            params = {
                'ingest': self.ingest_path,
                'database': db_name,
                'retrieval_method': self.retrieval_method,
                'eval': self.eval_queries,
                'dataset': self.dataset,
                'device': self.device,
                'no_save': True,
            }
            
            # Add BM25-specific parameters
            if self.retrieval_method == "bm25":
                params['threads'] = self.num_threads
            
            # Add fixed parameters (can override defaults)
            params.update(self.fixed_params)
            
            # Sample and add parameters to optimize
            for param_name in self.optimize_params:
                params[param_name] = self.suggest_param_value(trial, param_name)
            
            # Build command
            cmd, db_name = self.build_command(trial_id, params)
            
            # Clean up any existing files
            self.cleanup_trial_files(db_name)
            
            print(f"\nTrial {trial_id}: Testing parameters:")
            for key, value in params.items():
                print(f"  {key}={value}")
            print(f"  Command: {' '.join(cmd)}")
            
            # Run experiment (use script directory as cwd)
            script_dir = Path(__file__).parent.absolute()
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(script_dir), timeout=600)
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"  Execution time: {duration:.1f}s, Return code: {result.returncode}")
            
            if result.returncode != 0:
                print(f"Trial {trial_id} failed with return code {result.returncode}")
                print("STDERR:", result.stderr[-1000:] if result.stderr else "")
                return 0.0
            
            # Parse results (use script directory path)
            script_dir = Path(__file__).parent.absolute()
            results_path = script_dir / "results.json"
            
            if results_path.exists():
                with open(results_path, "r") as f:
                    results = json.load(f)
                
                # Get the specified metric
                if self.metric == "legacy":
                    score = results.get("accuracy", 0.0)
                else:
                    score = results.get("metrics", {}).get(self.metric, 0.0)
                
                accuracy = results.get("accuracy", 0.0)
                all_metrics = results.get("metrics", {})
            else:
                print("Warning: results.json not found")
                score = 0.0
                accuracy = 0.0
                all_metrics = {}
            
            print(f"  Result: {self.metric}={score:.4f}, accuracy={accuracy:.4f}")
            
            # Log the result
            self.log_result(trial_id, {
                "params": params,
                "score": score,
                "metric": self.metric,
                "accuracy": accuracy,
                "all_metrics": all_metrics,
                "duration": duration,
                "retrieval_method": self.retrieval_method
            })
            
            return score
            
        except subprocess.TimeoutExpired:
            print(f"Trial {trial_id}: Timeout after 600 seconds")
            return 0.0
        except Exception as e:
            print(f"Trial {trial_id} error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
        finally:
            # Always clean up
            if 'db_name' in locals():
                self.cleanup_trial_files(db_name)
    
    def log_result(self, trial_id: int, result: Dict[str, Any]):
        """Log trial result to JSON file."""
        try:
            if self.results_log.exists():
                with open(self.results_log, 'r') as f:
                    results = json.load(f)
            else:
                results = []
            
            result["trial_id"] = trial_id
            result["timestamp"] = time.time()
            results.append(result)
            
            with open(self.results_log, 'w') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not log result: {e}")
    
    def run_optimization(self, n_trials: int = 50, timeout: int = None, clean_start: bool = True):
        """Run the optimization process."""
        
        print(f"\n{'='*60}")
        print(f"Starting {self.retrieval_method.upper()} parameter optimization")
        print(f"{'='*60}")
        print(f"Trials: {n_trials}")
        print(f"Queries: {self.eval_queries}")
        print(f"Dataset: {self.dataset}")
        print(f"Metric: {self.metric}")
        print(f"Results: {self.results_log}")
        print(f"{'='*60}\n")
        
        # Optionally clean up previous optimization
        if clean_start:
            for file_path in [self.optuna_db, self.results_log, self.best_params_file]:
                if file_path.exists():
                    try:
                        file_path.unlink()
                        print(f"Cleaned up: {file_path}")
                    except Exception:
                        pass
        
        # Create sampler with better configuration for faster convergence
        # Use TPE sampler with reduced startup trials for faster exploitation
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=min(10, n_trials // 2),  # Random exploration phase
            n_ei_candidates=24,  # Number of candidates for Expected Improvement
            multivariate=True,  # Consider parameter interactions
            warn_independent_sampling=False  # Suppress warnings
        )
        
        # Create study
        study_name = f"{self.retrieval_method}_optimization_{int(time.time())}"
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=f"sqlite:///{self.optuna_db}",
            load_if_exists=not clean_start,
            sampler=sampler
        )
        
        # Progress callback
        def progress_callback(study, trial):
            print(f"\n{'='*60}")
            print(f"Completed trial {trial.number + 1}/{n_trials}")
            print(f"Best {self.metric}: {study.best_value:.4f}")
            print(f"Best parameters: {study.best_params}")
            print(f"{'='*60}")
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[progress_callback]
        )
        
        # Print final results
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE")
        print("="*60)
        print(f"Best {self.metric}: {study.best_value:.4f}")
        print("Best parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        # Save best parameters
        best_result = {
            "retrieval_method": self.retrieval_method,
            "best_metric": self.metric,
            "best_score": study.best_value,
            "best_params": study.best_params,
            "optimized_params": self.optimize_params,
            "fixed_params": self.fixed_params,
            "n_trials": len(study.trials),
            "timestamp": time.time()
        }
        
        with open(self.best_params_file, "w") as f:
            json.dump(best_result, f, indent=2)
        
        print(f"\nBest parameters saved to: {self.best_params_file}")
        print(f"Full results: {self.results_log}")
        print(f"Optuna study: {self.optuna_db}")
        print("="*60 + "\n")
        
        return study

def main():
    parser = argparse.ArgumentParser(
        description="Optimize retrieval parameters using Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize BM25 k1, b and method parameters
  python3 optimize_retrieval.py --retrieval_method bm25 --optimize bm25_k1 bm25_b bm25_method
  
  # Optimize BM25 with score-based strategy
  python3 optimize_retrieval.py --retrieval_method bm25 --optimize bm25_k1 top_p --retrieval_strategy top_p
  
  # Optimize vector retrieval with top_p strategy
  python3 optimize_retrieval.py --retrieval_method vector --optimize top_p --retrieval_strategy top_p
  
  # Optimize vector with fixed_k strategy
  python3 optimize_retrieval.py --retrieval_method vector --optimize top_k_retriever --retrieval_strategy fixed_k
  
  # Optimize with fixed device and thread settings
  python3 optimize_retrieval.py --retrieval_method bm25 --optimize bm25_k1 bm25_b --device xpu --threads 8
  
  # Optimize with custom metric and more trials
  python3 optimize_retrieval.py --retrieval_method vector --optimize top_p --metric precision@10 --trials 100

Note: Any parameter not specified in --optimize will be fixed at its command-line value (or default).
      Use --list-params to see all available parameters.
        """
    )
    
    # Add ALL parameters from centralized definitions to support full configuration
    # These parameters are shared with single_shot_retrieval.py but hidden from help
    # to keep the help output clean and focused on optimization-specific options
    from params import add_all_args
    
    # Create a custom HelpFormatter that hides imported parameters from help
    # but still accepts them from command line
    class OptimizationHelpFormatter(argparse.RawDescriptionHelpFormatter):
        """Custom formatter that shows only optimization-specific parameters in help"""
        pass
    
    # We'll add all parameters but mark the imported ones with help=argparse.SUPPRESS
    # to hide them from --help output while still accepting them
    original_add_all_args = add_all_args
    
    def add_all_args_suppressed(parser):
        """Add all args but suppress their help text"""
        from params import ALL_PARAMS
        for param in ALL_PARAMS:
            # Create a copy with suppressed help
            import copy
            param_copy = copy.copy(param)
            original_help = param_copy.help
            param_copy.help = argparse.SUPPRESS  # Hide from help output
            param_copy.add_to_parser(parser)
    
    add_all_args_suppressed(parser)
    
    # Override eval with optimization-specific handling (must be integer)
    for action in parser._actions:
        if '--eval' in action.option_strings:
            action.type = int
            action.nargs = None
            action.const = None
            action.default = 100
            action.help = argparse.SUPPRESS  # Keep it hidden
            break
    
    # Add optimization-specific parameters in a visible group
    opt_group = parser.add_argument_group(
        'Optimization Parameters',
        'Parameters specific to the optimization process (other parameters from single_shot_retrieval.py are also supported)'
    )
    
    opt_group.add_argument("--trials", type=int, default=50,
                          help="Number of optimization trials")
    opt_group.add_argument("--timeout", type=int, default=None,
                          help="Timeout in seconds (None = no timeout)")
    opt_group.add_argument("--output-dir", default="optimization_results",
                          help="Output directory for results")
    opt_group.add_argument("--metric", default="average_precision",
                          choices=["precision@N", "recall@N", "f1@N", "average_precision"],
                          help="Metric to optimize")
    opt_group.add_argument("--resume", action="store_true",
                          help="Resume previous optimization")
    opt_group.add_argument("--optimize", nargs="+", default=None,
                          help=f"Parameters to optimize. All other parameters will be fixed at specified or default values. Available: {', '.join(OPTIMIZABLE_PARAM_NAMES)}")
    opt_group.add_argument("--list-params", action="store_true",
                          help="List all available parameters for optimization and exit")
    opt_group.add_argument("--show-all-params", action="store_true",
                          help="Show all supported parameters (including single_shot_retrieval.py params) in help")
    
    # Check if user wants to see all parameters before parsing
    if '--show-all-params' in sys.argv:
        print("\n" + "="*80)
        print("ALL SUPPORTED PARAMETERS")
        print("="*80)
        print("\nThis script supports ALL parameters from single_shot_retrieval.py")
        print("plus the optimization-specific parameters shown above.\n")
        print("For a complete list of all retrieval parameters, run:")
        print("  python3 params.py list [bm25|vector]\n")
        print("Or see single_shot_retrieval.py --help\n")
        print("="*80 + "\n")
        sys.exit(0)
    
    args = parser.parse_args()
    
    # List parameters if requested
    if args.list_params:
        from params import print_param_info
        print_param_info(args.retrieval_method)
        return None
    
    # Collect all parameter values from command line args
    # These will be used as fixed values for parameters not being optimized
    from params import PARAM_BY_NAME
    fixed_params = {}
    
    # Get all optimizable parameters that were specified on command line
    optimize_set = set(args.optimize) if args.optimize else set()
    
    for param_name, param_def in PARAM_BY_NAME.items():
        # Skip if this parameter is being optimized
        if param_name in optimize_set:
            continue
            
        # Check if this parameter was specified on command line
        # Use the first CLI argument name (without --)
        arg_name = param_def.arg_names[0].lstrip('-').replace('-', '_')
        if hasattr(args, arg_name):
            value = getattr(args, arg_name)
            # Only include if non-default or explicitly set
            if value is not None and value != param_def.default:
                fixed_params[param_name] = value
    
    # Create optimizer
    optimizer = RetrievalOptimizer(
        retrieval_method=args.retrieval_method,
        ingest_path=args.ingest,
        eval_queries=args.eval,
        dataset=args.dataset,
        output_dir=args.output_dir,
        num_threads=args.threads,
        metric=args.metric,
        device=args.device,
        optimize_params=args.optimize,
        fixed_params=fixed_params
    )
    
    # Run optimization
    study = optimizer.run_optimization(
        n_trials=args.trials,
        timeout=args.timeout,
        clean_start=not args.resume
    )
    
    return study

if __name__ == "__main__":
    main()
