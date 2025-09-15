#!/usr/bin/env python3
"""
Evaluate backward transfer in continual learning
"""
import logging
import argparse
import yaml
import os
from pathlib import Path
from eval_continual_learning import TSDiffEvaluator, load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_backward_transfer(base_config, task_checkpoints, task_datasets, output_dir, num_samples=100):
    """Evaluate each checkpoint on all previous + current tasks"""
    results = {
        "setup": {
            "num_tasks": len(task_checkpoints),
            "datasets": task_datasets,
            "checkpoints": task_checkpoints,
            "num_samples": num_samples
        },
        "task_performance": []
    }
    
    for i, checkpoint_path in enumerate(task_checkpoints):
        logger.info(f"{'='*60}")
        logger.info(f"Evaluating Task {i+1} checkpoint on all previous datasets")
        logger.info(f"Checkpoint: {os.path.basename(checkpoint_path)}")
        logger.info(f"{'='*60}")
        
        # Verify checkpoint exists
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            continue
        
        # Datasets to evaluate (current + all previous)
        eval_datasets = task_datasets[:i+1] 
        logger.info(f"Evaluating on datasets: {eval_datasets}")
        
        try:
            evaluator = TSDiffEvaluator(base_config, checkpoint_path)
            task_results = evaluator.evaluate_multiple_datasets(eval_datasets, num_samples=num_samples)
            
            results["task_performance"].append({
                "task_id": i + 1,
                "checkpoint": checkpoint_path,
                "evaluated_datasets": eval_datasets,
                "results": task_results["results"]
            })
            
        except Exception as e:
            logger.error(f"Failed to evaluate Task {i+1}: {e}")
            results["task_performance"].append({
                "task_id": i + 1,
                "checkpoint": checkpoint_path,
                "error": str(e)
            })
    
    # Compute backward transfer matrix
    _compute_backward_transfer_metrics(results)
    
    # Save results
    results_file = Path(output_dir) / "backward_transfer_results.yaml"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    
    logger.info(f"Backward transfer evaluation completed!")
    logger.info(f"Results saved to: {results_file}")
    
    # Print summary
    _print_backward_transfer_summary(results)
    
    return results

def _compute_backward_transfer_metrics(results):
    """Compute forgetting metrics"""
    datasets = results["setup"]["datasets"]
    task_performance = results["task_performance"]
    
    # Create performance matrix
    performance_matrix = {}
    
    for task_perf in task_performance:
        task_id = task_perf["task_id"]
        if "results" in task_perf:
            for eval_result in task_perf["results"]:
                if eval_result.get("success", False):
                    dataset = eval_result["dataset"]
                    crps = eval_result.get("CRPS", None)
                    
                    if dataset not in performance_matrix:
                        performance_matrix[dataset] = {}
                    performance_matrix[dataset][task_id] = crps
    
    # Compute forgetting metrics
    forgetting_metrics = {}
    for dataset in datasets:
        if dataset in performance_matrix:
            perfs = performance_matrix[dataset]
            task_ids = sorted(perfs.keys())
            
            if len(task_ids) > 1:
                # Find when this dataset was first learned
                dataset_task_id = datasets.index(dataset) + 1
                
                if dataset_task_id in perfs:
                    initial_perf = perfs[dataset_task_id]
                    final_perf = perfs[max(task_ids)]
                    
                    if initial_perf is not None and final_perf is not None:
                        forgetting = ((final_perf - initial_perf) / initial_perf) * 100
                        forgetting_metrics[dataset] = {
                            "initial_performance": initial_perf,
                            "final_performance": final_perf,
                            "forgetting_percent": forgetting
                        }
    
    results["backward_transfer_analysis"] = {
        "performance_matrix": performance_matrix,
        "forgetting_metrics": forgetting_metrics
    }

def _print_backward_transfer_summary(results):
    """Print readable summary"""
    print("\n" + "="*70)
    print("BACKWARD TRANSFER SUMMARY")
    print("="*70)
    
    if "backward_transfer_analysis" in results:
        forgetting_metrics = results["backward_transfer_analysis"]["forgetting_metrics"]
        
        if forgetting_metrics:
            print("\nFORGETTING ANALYSIS:")
            print("-" * 50)
            for dataset, metrics in forgetting_metrics.items():
                initial = metrics["initial_performance"]
                final = metrics["final_performance"]
                forgetting = metrics["forgetting_percent"]
                
                status = "IMPROVED" if forgetting < 0 else "DEGRADED"
                print(f"{dataset}:")
                print(f"  Initial CRPS: {initial:.4f}")
                print(f"  Final CRPS:   {final:.4f}")
                print(f"  Change:       {forgetting:+.2f}% ({status})")
                print()
        else:
            print("No forgetting metrics computed - insufficient data")
    
    print("="*70)

def main():
    parser = argparse.ArgumentParser(description="Evaluate backward transfer")
    parser.add_argument("--base_config", type=str, required=True, 
                       help="Base evaluation config file")
    parser.add_argument("--task_checkpoints", nargs='+', required=True, 
                       help="Checkpoint paths in task order")
    parser.add_argument("--task_datasets", nargs='+', required=True, 
                       help="Dataset names in task order") 
    parser.add_argument("--out_dir", type=str, default="./backward_transfer_results", 
                       help="Output directory")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples for evaluation")
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.task_checkpoints) != len(args.task_datasets):
        logger.error("Number of checkpoints must match number of datasets")
        return
    
    if not os.path.exists(args.base_config):
        logger.error(f"Base config file not found: {args.base_config}")
        return
    
    # Load base config
    base_config = load_config(args.base_config, argparse.Namespace())
    
    logger.info("Starting backward transfer evaluation")
    logger.info(f"Tasks: {list(zip(args.task_datasets, [os.path.basename(c) for c in args.task_checkpoints]))}")
    logger.info(f"Number of samples: {args.num_samples}")
    
    results = evaluate_backward_transfer(
        base_config, 
        args.task_checkpoints, 
        args.task_datasets, 
        args.out_dir,
        args.num_samples
    )
    
    logger.info("Backward transfer evaluation completed!")

if __name__ == "__main__":
    main()
