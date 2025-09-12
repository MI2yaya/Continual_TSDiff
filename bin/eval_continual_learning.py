#!/usr/bin/env python3
"""
TSDiff Continual Learning Evaluation Script
Evaluates a trained checkpoint on any specified dataset
"""

import logging
import argparse
import yaml
import torch
from pathlib import Path
from tqdm.auto import tqdm

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.field_names import FieldName

import uncond_ts_diff.configs as diffusion_configs
from uncond_ts_diff.model import TSDiff
from uncond_ts_diff.sampler import DDPMGuidance, DDIMGuidance
from uncond_ts_diff.utils import (
    create_transforms,
    create_splitter,
    filter_metrics,
    MaskInput,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global constants
GUIDANCE_MAP = {"ddpm": DDPMGuidance, "ddim": DDIMGuidance}

class TSDiffEvaluator:
    """Evaluate TSDiff checkpoints on arbitrary datasets"""
    
    def __init__(self, config: dict, checkpoint_path: str):
        self.config = config
        self.checkpoint_path = checkpoint_path
        
        # Setup device
        self.device = torch.device(config.get("device", "cuda:0"))
        logger.info(f"Using device: {self.device}")
        
        # Load model and checkpoint
        self.model = self._load_model_from_checkpoint()
        
    def _load_model_from_checkpoint(self) -> TSDiff:
        """Load TSDiff model from checkpoint"""
        logger.info(f"Loading model from checkpoint: {self.checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            
            # Create model with config parameters
            model = TSDiff(
                **getattr(diffusion_configs, self.config["diffusion_config"]),
                freq=self.config["freq"],
                use_features=self.config["use_features"],
                use_lags=self.config["use_lags"],
                normalization=self.config["normalization"],
                context_length=self.config["context_length"],
                prediction_length=self.config["prediction_length"],
                lr=self.config["lr"],
                init_skip=self.config["init_skip"],
            )
            
            # Load model weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            logger.info("Model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def evaluate_dataset(self, dataset_name: str, num_samples: int = 100) -> dict:
        """Evaluate model on specified dataset"""
        logger.info(f"Evaluating on dataset: {dataset_name}")
        
        try:
            # Load dataset
            dataset = get_dataset(
                dataset_name,
                prediction_length=24,  # Standardized
                regenerate=True
            )
            
            # Auto-correct frequency if needed
            actual_freq = str(dataset.metadata.freq)
            eval_freq = self.config.get("freq", "H")
            if actual_freq != eval_freq:
                logger.warning(f"Frequency mismatch: config={eval_freq}, dataset={actual_freq}")
                logger.warning("Using dataset's native frequency for evaluation")
                self.config["freq"] = actual_freq
            
            logger.info(f"Dataset loaded: {dataset_name}")
            logger.info(f"  Frequency: {dataset.metadata.freq}")
            logger.info(f"  Prediction length: {dataset.metadata.prediction_length}")
            
            # Create transformation pipeline
            transformation = create_transforms(
                num_feat_dynamic_real=0,
                num_feat_static_cat=0,
                num_feat_static_real=0,
                time_features=self.model.time_features,
                prediction_length=self.config["prediction_length"],
            )
            
            # Setup guidance sampler
            Guidance = GUIDANCE_MAP[self.config["sampler"]]
            sampler_kwargs = self.config.get("sampler_params", {})
            
            sampler = Guidance(
                model=self.model,
                prediction_length=self.config["prediction_length"],
                num_samples=num_samples,
                missing_scenario="none",
                missing_values=0,
                **sampler_kwargs,
            )
            
            # Setup test data transformation
            transformed_testdata = transformation.apply(dataset.test, is_train=False)
            test_splitter = create_splitter(
                past_length=self.config["context_length"] + max(self.model.lags_seq),
                future_length=self.config["prediction_length"],
                mode="test",
            )
            
            masking_transform = MaskInput(
                FieldName.TARGET,
                FieldName.OBSERVED_VALUES,
                self.config["context_length"],
                "none",
                0,
            )
            test_transform = test_splitter + masking_transform
            
            # Create predictor
            predictor = sampler.get_predictor(
                test_transform,
                batch_size=max(1, 1280 // num_samples),
                device=str(self.device),
            )
            
            # Generate predictions
            logger.info(f"Generating {num_samples} samples for evaluation...")
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=transformed_testdata,
                predictor=predictor,
                num_samples=num_samples,
            )
            
            forecasts = list(tqdm(
                forecast_it, 
                total=len(transformed_testdata), 
                desc="Generating forecasts"
            ))
            tss = list(ts_it)
            
            # Compute metrics
            logger.info("Computing evaluation metrics...")
            evaluator = Evaluator()
            metrics, _ = evaluator(tss, forecasts)
            filtered_metrics = filter_metrics(metrics)
            
            # Add dataset info to results
            result = {
                "dataset": dataset_name,
                "checkpoint": self.checkpoint_path,
                "num_samples": num_samples,
                "frequency": actual_freq,
                "success": True,
                **filtered_metrics
            }
            
            logger.info("Evaluation completed successfully")
            logger.info(f"CRPS: {filtered_metrics.get('CRPS', 'N/A')}")
            logger.info(f"MASE: {filtered_metrics.get('MASE', 'N/A')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                "dataset": dataset_name,
                "checkpoint": self.checkpoint_path,
                "success": False,
                "error": str(e)
            }
    
    def evaluate_multiple_datasets(self, dataset_names: list, num_samples: int = 100) -> dict:
        """Evaluate model on multiple datasets"""
        logger.info(f"Evaluating on {len(dataset_names)} datasets")
        
        results = {
            "config": self.config,
            "checkpoint": self.checkpoint_path,
            "evaluation_settings": {
                "num_samples": num_samples,
                "device": str(self.device)
            },
            "results": []
        }
        
        for dataset_name in dataset_names:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating dataset: {dataset_name}")
            logger.info(f"{'='*60}")
            
            result = self.evaluate_dataset(dataset_name, num_samples)
            results["results"].append(result)
        
        # Summary
        successful_evals = [r for r in results["results"] if r.get("success", False)]
        logger.info(f"\nEvaluation Summary:")
        logger.info(f"  Total datasets: {len(dataset_names)}")
        logger.info(f"  Successful: {len(successful_evals)}")
        logger.info(f"  Failed: {len(dataset_names) - len(successful_evals)}")
        
        if successful_evals:
            logger.info("\nResults:")
            for result in successful_evals:
                crps = result.get('CRPS', 'N/A')
                mase = result.get('MASE', 'N/A')
                logger.info(f"  {result['dataset']}: CRPS={crps}, MASE={mase}")
        
        return results

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate TSDiff checkpoint on datasets")
    parser.add_argument("-c", "--config", type=str, required=True, 
                       help="Path to evaluation config YAML")
    parser.add_argument("--ckpt", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--eval_datasets", nargs='+', required=True,
                       help="Dataset names to evaluate on (solar_nips, pedestrian_counts)")
    parser.add_argument("--out_dir", type=str, default="./eval_results",
                       help="Output directory for results")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples for evaluation")
    return parser.parse_args()

def load_config(config_path: str, args: argparse.Namespace) -> dict:
    """Load and update configuration"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Override with command line arguments
    config["checkpoint_path"] = args.ckpt
    config["eval_datasets"] = args.eval_datasets
    config["num_samples"] = args.num_samples
    
    return config

def main():
    """Main evaluation function"""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load config
        config = load_config(args.config, args)
        
        logger.info("Starting TSDiff continual learning evaluation")
        logger.info(f"Checkpoint: {args.ckpt}")
        logger.info(f"Datasets: {args.eval_datasets}")
        logger.info(f"Output: {output_dir}")
        
        # Create evaluator
        evaluator = TSDiffEvaluator(config, args.ckpt)
        
        # Run evaluation
        results = evaluator.evaluate_multiple_datasets(
            args.eval_datasets, 
            args.num_samples
        )
        
        # Save results
        results_file = output_dir / "continual_evaluation_results.yaml"
        with open(results_file, "w") as f:
            yaml.dump(results, f, default_flow_style=False)
        
        logger.info(f"\nResults saved to: {results_file}")
        
        # Print summary
        successful_results = [r for r in results["results"] if r.get("success", False)]
        if successful_results:
            print("\n" + "="*60)
            print("EVALUATION SUMMARY")
            print("="*60)
            for result in successful_results:
                print(f"Dataset: {result['dataset']}")
                print(f"  CRPS: {result.get('CRPS', 'N/A')}")
                print(f"  MASE: {result.get('MASE', 'N/A')}")
                print(f"  sMAPE: {result.get('sMAPE', 'N/A')}")
                print()
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
