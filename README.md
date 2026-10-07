# Ra-NEM Experiments:
This is the repo for recreating the results of the paper:
"For Those Who Believe in Faithfulness: Optimizing the Area Under Insertion and
Deletion Curves for Ranking Relative Feature Importance"
# Installation:
1. Use anaconda to install the "environment.yml" file.
2. Activate the new environment called "ra_nem_experiments"
3. Download the ImageNet dataset.
4. In "exp_config.py" set "IMAGENET_DATAPATH" to the path to the downloaded ImageNet dataset. 
# Running experiments:
The "exp_config.py" file defines which combinations of dataset, explained model, XAI method and XAI metrics are run.
By default, it will run all models and methods and measure Faithfullness using the Full evaluation dataset. To run Robustness and Randomization, we highly suggest setting  "HEAVY_PIPELINE = True" in exp_config.py to calculate the metrics using a subset of the evaluation data, since these metrics are quite computationally demanding.

After defining the experimental setup in "exp_config.py" and activating the correct environment, run "main.py to start the experiments. Main.py will create and populate a new folder with experimental results. After "main.py" is done running, please use "show_res.py" to display the results.
