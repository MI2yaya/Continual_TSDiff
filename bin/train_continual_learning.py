#!/usr/bin/env python3
"""
Continual Learning Pipeline for TSDiff with Score Function Regularization
"""
import logging
import argparse
import yaml
import os
from pathlib import Path
from train_model_no_lightning import TSDiffTrainer, load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_continual_sequence(task_configs, output_dir, lambda_reg=1.0):
    """Train tasks sequentially with score regularization"""
    previous_checkpoint = None
    all_results = []
    
    for i, task_config_path in enumerate(task_configs):
        logger.info(f"{'='*60}")
        logger.info(f"Training Task {i+1}/{len(task_configs)}: {os.path.basename(task_config_path)}")
        logger.info(f"{'='*60}")
        
        # Load config and add continual learning params
        config = load_config(task_config_path, argparse.Namespace())
        config["lambda_reg"] = lambda_reg
        
        # Create task-specific output directory
        dataset_name = config['dataset']
        task_output_dir = Path(output_dir) / f"task_{i+1}_{dataset_name}"
        
        # Create trainer
        trainer = TSDiffTrainer(config, str(task_output_dir))
        
        # Set previous model for regularization (skip first task)
        if i > 0 and previous_checkpoint and os.path.exists(previous_checkpoint):
            logger.info(f"Loading previous model for regularization: {previous_checkpoint}")
            trainer.set_previous_model(previous_checkpoint)
        else:
            logger.info("No previous model - training first task or checkpoint missing")
        
        # Train current task
        results = trainer.train()
        all_results.append({
            "task_id": i + 1,
            "dataset": dataset_name,
            "config_path": task_config_path,
            "results": results
        })

        previous_checkpoint = results.get("best_checkpoint")
        print(f"DEBUG: Task {i+1} completed")
        print(f"DEBUG: Best checkpoint: {previous_checkpoint}")
        print(f"DEBUG: Checkpoint exists: {os.path.exists(previous_checkpoint) if previous_checkpoint else 'None'}")
        logger.info(f"Task {i+1} completed. Best checkpoint: {previous_checkpoint}")

    
    # Save overall results
    final_results = {
        "continual_learning_setup": {
            "lambda_reg": lambda_reg,
            "num_tasks": len(task_configs),
            "task_sequence": [os.path.basename(config) for config in task_configs]
        },
        "task_results": all_results
    }
    
    results_file = Path(output_dir) / "continual_learning_summary.yaml"
    with open(results_file, "w") as f:
        yaml.dump(final_results, f, default_flow_style=False)
    
    logger.info(f"Continual learning completed! Summary saved to: {results_file}")
    return final_results

def main():
    parser = argparse.ArgumentParser(description="Train TSDiff continual learning")
    parser.add_argument("--task_configs", nargs='+', required=True, 
                       help="Task config files in training order")
    parser.add_argument("--out_dir", type=str, default="./continual_logs", 
                       help="Output directory")  
    parser.add_argument("--lambda_reg", type=float, default=1.0, 
                       help="Score regularization strength")
    
    args = parser.parse_args()
    
    # Validate config files exist
    for config_path in args.task_configs:
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            return
    
    logger.info("Starting continual learning pipeline")
    logger.info(f"Tasks: {[os.path.basename(c) for c in args.task_configs]}")
    logger.info(f"Lambda regularization: {args.lambda_reg}")
    logger.info(f"Output directory: {args.out_dir}")
    
    results = train_continual_sequence(args.task_configs, args.out_dir, args.lambda_reg)
    logger.info("Continual learning pipeline completed successfully!")
    return results

if __name__ == "__main__":
    main()
