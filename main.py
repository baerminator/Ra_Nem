from exp_config import *
from pathlib import Path
import torch
if __name__ == '__main__':
    for dataset_name, dataset_constructor in CHOSEN_DATASETS.items():
        dataset = dataset_constructor()
        train_data, test_data = dataset.get_data()
        for model_name, model_constructor in CHOSEN_MODELS.items():
            model =( model_constructor()).cuda().eval()
            for method_name, method_constructor in CHOSEN_METHODS.items():
                exp_store = Path(EXP_STORAGE_PATH) / dataset_name / model_name / method_name
                method = method_constructor(model, train_data, use_predicted_labels = True)
                for experiment_name, experiment_constructor in CHOSEN_EXPERIMENTS.items():
                    print(f"Running experiment {experiment_name} on {dataset_name} with {model_name} and {method_name}")
                    experiment = experiment_constructor(method, model, train_data, test_data,exp_store)
                    

    