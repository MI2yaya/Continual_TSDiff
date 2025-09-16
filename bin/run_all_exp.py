import os
import sys
import subprocess
import itertools
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_runner.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# Experimental configuration
TASK_ORDERS = [
    # ["train_kdd_cup.yaml", "train_pedestrian_counts.yaml", "train_uber_tlc.yaml"],
    # ["train_kdd_cup.yaml", "train_uber_tlc.yaml", "train_pedestrian_counts.yaml"],
    ["train_pedestrian_counts.yaml", "train_kdd_cup.yaml", "train_uber_tlc.yaml"],
    ["train_pedestrian_counts.yaml", "train_uber_tlc.yaml", "train_kdd_cup.yaml"],
    ["train_uber_tlc.yaml", "train_kdd_cup.yaml", "train_pedestrian_counts.yaml"],
    ["train_uber_tlc.yaml", "train_pedestrian_counts.yaml", "train_kdd_cup.yaml"],
]

METHODS = {
    "naive": {},
    "dropout": {"dropout_rate": [0.1, 0.3, 0.5]},
    "score_l2": {"lambda_reg": [0.5, 1.0, 2.0]},
    "score_l2": {"lambda_reg": [0.5, 1.0, 2.0]},
}


# Directory configuration
BASE_OUTPUT_DIR = "./full_experiments_3"
CONFIG_DIR = "./configs/train_tsdiff"

# Execution parameters
TRAINING_TIMEOUT = 3600  # 2 hours
EVALUATION_TIMEOUT = 3600  # 1 hour


class ExperimentRunner:
    """Manages the execution of continual learning experiments."""
    
    def __init__(self):
        self.total_experiments = 0
        self.successful_experiments = 0
        self.successful_evaluations = 0
        self.failed_experiments = []
        self.failed_evaluations = []
        
    def generate_order_name(self, task_order: List[str], order_idx: int) -> str:
        """Generate a descriptive name for the task ordering."""
        task_names = [task.replace('train_', '').replace('.yaml', '') for task in task_order]
        return f"order_{order_idx+3}_{'_'.join(task_names)}"
    
    def run_training_experiment(self, method: str, params: Dict[str, Any], 
                               task_order: List[str], order_idx: int) -> bool:
        """Execute a single training experiment."""
        task_configs = [os.path.join(CONFIG_DIR, task) for task in task_order]
        order_name = self.generate_order_name(task_order, order_idx)
        
        # Construct command
        cmd = [
            "python", "bin/train_continual_learning.py",
            "--task_configs"] + task_configs + [
            "--out_dir", os.path.join(BASE_OUTPUT_DIR, order_name),
            "--method", method
        ]
        
        # Add method-specific parameters
        for param_name, param_value in params.items():
            cmd.extend([f"--{param_name}", str(param_value)])
        
        experiment_id = f"{method}({params}) on {order_name}"
        logger.info(f"Starting experiment: {experiment_id}")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=TRAINING_TIMEOUT,
                check=False
            )
            
            if result.returncode == 0:
                logger.info(f"Training completed successfully: {experiment_id}")
                return True
            else:
                logger.error(f"Training failed: {experiment_id}")
                logger.error(f"Error output: {result.stderr}")
                self.failed_experiments.append({
                    'experiment': experiment_id,
                    'error': result.stderr,
                    'return_code': result.returncode
                })
                return False
                
        except subprocess.TimeoutExpired:
            logger.warning(f"Training timeout exceeded: {experiment_id}")
            self.failed_experiments.append({
                'experiment': experiment_id,
                'error': 'Timeout exceeded',
                'return_code': None
            })
            return False
            
        except Exception as e:
            logger.error(f"Unexpected error in training: {experiment_id} - {str(e)}")
            self.failed_experiments.append({
                'experiment': experiment_id,
                'error': str(e),
                'return_code': None
            })
            return False
    
    def run_backward_transfer_evaluation(self, experiment_dir: Path) -> bool:
        """Execute backward transfer evaluation for a completed experiment."""
        try:
            # Locate task checkpoints and extract dataset names
            checkpoints = []
            task_datasets = []
            
            for task_dir in sorted(experiment_dir.glob("task_*")):
                checkpoint_files = list(task_dir.glob("*_checkpoint_best.pth"))
                if checkpoint_files:
                    checkpoints.append(str(checkpoint_files[0]))
                    # Extract dataset name from directory structure
                    dataset_name = task_dir.name.split('_', 2)[2]
                    task_datasets.append(dataset_name)
            
            if len(checkpoints) < 2:
                logger.warning(f"Insufficient checkpoints for evaluation: {experiment_dir}")
                logger.info(f"Found {len(checkpoints)} checkpoints, need at least 2")
                return False
            
            # Construct evaluation command
            cmd = [
                "python", "bin/eval_backward_transfer.py",
                "--base_config", "configs/eval_continual.yaml",
                "--task_checkpoints"] + checkpoints + [
                "--task_datasets"] + task_datasets + [
                "--out_dir", str(experiment_dir / "evaluation"),
                "--num_samples", "100"
            ]
            
            logger.info(f"Starting evaluation for: {experiment_dir.name}")
            logger.debug(f"Evaluation command: {' '.join(cmd[:10])}... (truncated)")
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=EVALUATION_TIMEOUT,
                check=False
            )
            
            if result.returncode == 0:
                logger.info(f"Evaluation completed successfully: {experiment_dir.name}")
                return True
            else:
                logger.error(f"Evaluation failed: {experiment_dir.name}")
                logger.error(f"Error output: {result.stderr}")
                self.failed_evaluations.append({
                    'experiment_dir': str(experiment_dir),
                    'error': result.stderr,
                    'return_code': result.returncode
                })
                return False
                
        except subprocess.TimeoutExpired:
            logger.warning(f"Evaluation timeout exceeded: {experiment_dir.name}")
            self.failed_evaluations.append({
                'experiment_dir': str(experiment_dir),
                'error': 'Timeout exceeded',
                'return_code': None
            })
            return False
            
        except Exception as e:
            logger.error(f"Unexpected error in evaluation: {experiment_dir.name} - {str(e)}")
            self.failed_evaluations.append({
                'experiment_dir': str(experiment_dir),
                'error': str(e),
                'return_code': None
            })
            return False
    
    def execute_experiment_set(self, method: str, param_ranges: Dict[str, List], 
                             task_order: List[str], order_idx: int) -> None:
        """Execute all parameter combinations for a given method and task order."""
        order_name = self.generate_order_name(task_order, order_idx)
        
        if not param_ranges:  # Method with no hyperparameters (e.g., naive)
            self.total_experiments += 1
            success = self.run_training_experiment(method, {}, task_order, order_idx)
            
            if success:
                self.successful_experiments += 1
                # Run evaluation
                exp_dir = Path(BASE_OUTPUT_DIR) / order_name / f"method_{method}"
                if self.run_backward_transfer_evaluation(exp_dir):
                    self.successful_evaluations += 1
        else:
            # Generate all parameter combinations
            param_names = list(param_ranges.keys())
            param_values = list(param_ranges.values())
            
            for param_combo in itertools.product(*param_values):
                params = dict(zip(param_names, param_combo))
                self.total_experiments += 1
                
                success = self.run_training_experiment(method, params, task_order, order_idx)
                
                if success:
                    self.successful_experiments += 1
                    # Run evaluation
                    param_str = "_".join([f"{k}_{v}" for k, v in params.items()])
                    exp_dir = Path(BASE_OUTPUT_DIR) / order_name / f"method_{method}" / param_str
                    if self.run_backward_transfer_evaluation(exp_dir):
                        self.successful_evaluations += 1
    
    def run_all_experiments(self) -> None:
        """Execute the complete experimental suite."""
        logger.info("="*80)
        logger.info("STARTING CONTINUAL LEARNING EXPERIMENTAL SUITE")
        logger.info("="*80)
        logger.info(f"Total task orders: {len(TASK_ORDERS)}")
        logger.info(f"Methods to evaluate: {list(METHODS.keys())}")
        
        # Calculate total expected experiments
        expected_total = 0
        for param_ranges in METHODS.values():
            if not param_ranges:
                expected_total += len(TASK_ORDERS)
            else:
                param_combos = 1
                for param_list in param_ranges.values():
                    param_combos *= len(param_list)
                expected_total += len(TASK_ORDERS) * param_combos
        
        logger.info(f"Expected total experiments: {expected_total}")
        
        # Ensure output directory exists
        os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
        
        # Execute all experiment combinations
        for order_idx, task_order in enumerate(TASK_ORDERS):
            logger.info(f"Processing task order {order_idx + 1}/{len(TASK_ORDERS)}: {task_order}")
            
            for method, param_ranges in METHODS.items():
                logger.info(f"Executing method: {method}")
                self.execute_experiment_set(method, param_ranges, task_order, order_idx)
        
        # Generate final summary
        self.generate_final_report()
    
    def generate_final_report(self) -> None:
        """Generate and display the final experimental summary."""
        logger.info("="*80)
        logger.info("EXPERIMENTAL SUITE COMPLETED")
        logger.info("="*80)
        
        # Basic statistics
        success_rate = (self.successful_experiments / self.total_experiments * 100 
                       if self.total_experiments > 0 else 0)
        eval_rate = (self.successful_evaluations / self.successful_experiments * 100 
                    if self.successful_experiments > 0 else 0)
        
        logger.info(f"Total experiments attempted: {self.total_experiments}")
        logger.info(f"Successful training runs: {self.successful_experiments}")
        logger.info(f"Successful evaluations: {self.successful_evaluations}")
        logger.info(f"Training success rate: {success_rate:.1f}%")
        logger.info(f"Evaluation success rate: {eval_rate:.1f}%")
        
        # Report failures
        if self.failed_experiments:
            logger.warning(f"Failed training experiments: {len(self.failed_experiments)}")
            for i, failure in enumerate(self.failed_experiments[:5], 1):  # Show first 5
                logger.warning(f"  {i}. {failure['experiment']}: {failure['error'][:100]}...")
            if len(self.failed_experiments) > 5:
                logger.warning(f"  ... and {len(self.failed_experiments) - 5} more")
        
        if self.failed_evaluations:
            logger.warning(f"Failed evaluations: {len(self.failed_evaluations)}")
            for i, failure in enumerate(self.failed_evaluations[:5], 1):  # Show first 5
                logger.warning(f"  {i}. {failure['experiment_dir']}: {failure['error'][:100]}...")
            if len(self.failed_evaluations) > 5:
                logger.warning(f"  ... and {len(self.failed_evaluations) - 5} more")
        
        # Output locations
        logger.info(f"Results saved to: {BASE_OUTPUT_DIR}")
        logger.info(f"Log file: experiment_runner.log")
        logger.info("="*80)


def validate_environment() -> bool:
    """Validate that the experimental environment is properly set up."""
    required_files = [
        "bin/train_continual_learning.py",
        "bin/eval_backward_transfer.py",
        "configs/eval_continual.yaml"
    ]
    
    required_dirs = [CONFIG_DIR]
    
    missing_items = []
    
    # Check required files
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_items.append(f"Missing file: {file_path}")
    
    # Check required directories
    for dir_path in required_dirs:
        if not os.path.isdir(dir_path):
            missing_items.append(f"Missing directory: {dir_path}")
        else:
            # Check for task configuration files
            for task_order in TASK_ORDERS:
                for task_file in task_order:
                    full_path = os.path.join(dir_path, task_file)
                    if not os.path.exists(full_path):
                        missing_items.append(f"Missing config file: {full_path}")
    
    if missing_items:
        logger.error("Environment validation failed:")
        for item in missing_items:
            logger.error(f"  - {item}")
        return False
    
    logger.info("Environment validation passed")
    return True


def main() -> int:
    """Main entry point for the experiment runner."""
    try:
        logger.info("Initializing Continual Learning Experiment Suite")
        
        # Validate environment
        if not validate_environment():
            logger.error("Environment validation failed. Please fix the issues and retry.")
            return 1
        
        # Initialize and run experiments
        runner = ExperimentRunner()
        runner.run_all_experiments()
        
        # Return appropriate exit code
        if runner.successful_experiments == 0:
            logger.error("No experiments completed successfully")
            return 1
        elif runner.failed_experiments:
            logger.warning("Some experiments failed, but others succeeded")
            return 2
        else:
            logger.info("All experiments completed successfully")
            return 0
            
    except KeyboardInterrupt:
        logger.warning("Experiment suite interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error in main execution: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
