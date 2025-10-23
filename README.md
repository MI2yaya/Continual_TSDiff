Expanded from TSDiff, check that repo out.
## Installation

StateDiff requires Python 3.10.11 or higher.

* Create a conda environment (optional, but recommended).
```sh
conda create --name tsdiff --yes python=3.11 && conda activate tsdiff
```
* Install this package.
```sh
pip install --editable "."
```

## Usage

### Training Models


```sh
# Train StateDiff on Sinusoidal
python bin/train_scoremodel.py -c configs/train_sfdiff/sin.yaml


### Evaluating Models
The unconditional models trained above can be used for the following tasks.

#### Predict using Observation Self-Guidance
Use the `plotscore.py` script to run the forecasting experiments.

Example commands:
```sh
# Run observation self-guidance on the trained dataset
python bin/plotscore.py -c configs/train_sfdiff/sin.yaml



