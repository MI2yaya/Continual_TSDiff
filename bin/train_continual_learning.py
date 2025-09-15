#!/usr/bin/env python3

"""
Continual Learning Pipeline for TSDiff with Score Function Regularization Methods
"""

import logging
import argparse
import yaml
import os
from pathlib import Path
from train_model_no_lightning import TSDiffTrainer, load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_continual_sequence(task_configs, output_dir, method="score_l2", **method_kwargs):
    """Train tasks sequentially with specified regularization method"""
    previous_checkpoint = None
    all_results = []

    # Create method-specific output directory
    method_dir = Path(output_dir) / f"method_{method}"
    if method_kwargs:
        param_str = "_".join([f"{k}_{v}" for k, v in method_kwargs.items()])
        method_dir = method_dir / param_str
    method_dir.mkdir(parents=True, exist_ok=True)

    for i, task_config_path in enumerate(task_configs):
        logger.info(f"{'='*60}")
        logger.info(f"Training Task {i+1}/{len(task_configs)} with {method}: {os.path.basename(task_config_path)}")
        logger.info(f"Method kwargs: {method_kwargs}")
        logger.info(f"{'='*60}")

        # Load config and add method-specific params
        config = load_config(task_config_path, argparse.Namespace())

        # Apply regularization method settings
        if method == "naive":
            config["lambda_reg"] = 0.0
            config["score_loss_type"] = "l2"  # Default, but won't be used
            config["dropout_rate"] = 0.0

        elif method == "dropout":
            config["lambda_reg"] = 0.0
            config["score_loss_type"] = "l2"  # Default, but won't be used
            config["dropout_rate"] = method_kwargs.get("dropout_rate", 0.3)

        elif method == "score_l1":
            config["lambda_reg"] = method_kwargs.get("lambda_reg", 1.0)
            config["score_loss_type"] = "l1"
            config["dropout_rate"] = 0.0

        elif method == "score_l2":
            config["lambda_reg"] = method_kwargs.get("lambda_reg", 1.0)
            config["score_loss_type"] = "l2"
            config["dropout_rate"] = 0.0

        elif method == "combined":
            config["lambda_reg"] = method_kwargs.get("lambda_reg", 1.0)
            config["score_loss_type"] = method_kwargs.get("score_loss_type", "l2")
            config["dropout_rate"] = method_kwargs.get("dropout_rate", 0.3)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Create task-specific output directory
        dataset_name = config['dataset']
        task_output_dir = method_dir / f"task_{i+1}_{dataset_name}"

        # Create trainer
        trainer = TSDiffTrainer(config, str(task_output_dir))

        # Set previous model for methods that use score regularization
        if i > 0 and method in ["score_l1", "score_l2", "combined"] and previous_checkpoint and os.path.exists(previous_checkpoint):
            logger.info(f"Loading previous model for score regularization: {previous_checkpoint}")
            trainer.set_previous_model(previous_checkpoint)
        else:
            if i > 0 and method in ["score_l1", "score_l2", "combined"]:
                logger.warning("Previous checkpoint missing for score regularization method")
            else:
                logger.info("No previous model needed for this method")

        # Train current task
        results = trainer.train()
        
        all_results.append({
            "task_id": i + 1,
            "dataset": dataset_name,
            "config_path": task_config_path,
            "results": results
        })

        previous_checkpoint = results.get("best_checkpoint")
        logger.info(f"Task {i+1} completed. Best checkpoint: {previous_checkpoint}")

    # Save overall results
    final_results = {
        "continual_learning_setup": {
            "method": method,
            "method_params": method_kwargs,
            "num_tasks": len(task_configs),
            "task_sequence": [os.path.basename(config) for config in task_configs]
        },
        "task_results": all_results
    }

    results_file = method_dir / "continual_learning_summary.yaml"
    with open(results_file, "w") as f:
        yaml.dump(final_results, f, default_flow_style=False)

    logger.info(f"Continual learning completed! Summary saved to: {results_file}")
    return final_results

def main():
    parser = argparse.ArgumentParser(description="Train TSDiff continual learning")
    parser.add_argument("--task_configs", nargs='+', required=True,
                        help="Task config files in training order")
    parser.add_argument("--out_dir", type=str, default="./experiments",
                        help="Output directory")
    parser.add_argument("--method", type=str, default="score_l2",
                        choices=["naive", "dropout", "score_l1", "score_l2", "combined"],
                        help="Regularization method")

    # Method-specific parameters
    parser.add_argument("--lambda_reg", type=float, default=1.0,
                        help="Score regularization strength")
    parser.add_argument("--dropout_rate", type=float, default=0.3,
                        help="Dropout rate")
    parser.add_argument("--score_loss_type", type=str, default="l2",
                        choices=["l1", "l2"],
                        help="Distance metric for score regularization")

    args = parser.parse_args()

    # Prepare method kwargs
    method_kwargs = {}
    if args.method in ["score_l1", "score_l2", "combined"]:
        method_kwargs["lambda_reg"] = args.lambda_reg
    if args.method in ["dropout", "combined"]:
        method_kwargs["dropout_rate"] = args.dropout_rate
    if args.method == "combined":
        method_kwargs["score_loss_type"] = args.score_loss_type

    logger.info("Starting continual learning pipeline")
    logger.info(f"Tasks: {[os.path.basename(c) for c in args.task_configs]}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Method params: {method_kwargs}")

    results = train_continual_sequence(args.task_configs, args.out_dir, args.method, **method_kwargs)

    logger.info("Continual learning pipeline completed successfully!")
    return results

if __name__ == "__main__":
    main()
