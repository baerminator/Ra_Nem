from attrs import *
from exp_utils.run_exp import *
from exp_utils.exp_data import ImageNetDataset
from nem_config import TRAIN_BATCH_SIZE
from torchvision import transforms
import timm

# STORAGE CONSTANTS
IMAGENET_DATAPATH = None
EXP_STORAGE_PATH = "./experiments"
REUSE_IMAGENET_BASE_DATA = True
TEST_PIPELINE = False
HEAVY_PIPELINE = False


TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
TEST_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

POSSIBLE_DATASETS = {
    "imagenet": lambda: ImageNetDataset(
        EXP_STORAGE_PATH=EXP_STORAGE_PATH,
        IMAGENET_DATAPATH=IMAGENET_DATAPATH,
        REUSE_BASE_DATA=REUSE_IMAGENET_BASE_DATA,
        BATCH_SIZE=TRAIN_BATCH_SIZE,
        TRAIN_TRANSFORM=TRAIN_TRANSFORM,
        TEST_TRANSFORM=TEST_TRANSFORM,
        TEST_PIPELINE=TEST_PIPELINE,
        HEAVY_PIPELINE=HEAVY_PIPELINE,
    ),
}

POSSIBLE_MODELS = {
    "resnet50": lambda: timm.create_model("resnet50", pretrained=True),
    "convnext_small": lambda: timm.create_model("convnext_small", pretrained=True),
    "vit_base_patch16_224": lambda: timm.create_model(
        "vit_base_patch16_224", pretrained=True
    ),
    "vgg16": lambda: timm.create_model("vgg16", pretrained=True),
}

POSSIBLE_METHODS = {
    "exp_ranem": ranem_atr,
    "exp_gradcam": gradcam_atr,
    "exp_gradshap": gradshap_atr,
    "exp_gradshap_forgrad": gradshap_forgrad_atr,
    "exp_iba": iba_atr,
    "exp_intgrad": itg_atr,
    "exp_intgrad_forgrad": itg_forgrad_atr,
    "exp_rise": rs_atr,
    "exp_saliency": saliency_atr,
    "exp_saliency_forgrad": saliency_forgrad_atr,
    "exp_smoothpixelmask": smoothpixelmask_atr,
    "exp_nemt": nemt_atr,
}

POSSIBLE_EXPERIMENTS = {
    "faithfullness": exp_faithfullness,
    "robustness": exp_robustness,  # <--- We Recommend to set HEAVY_PIPELINE to True before running this experiment
    "randomization": exp_randomization,  # <--- We Recommend to set  HEAVY_PIPELINE to True before running this experiment
}


# EXPERIMENT CONFIG
CHOSEN_DATASETS = POSSIBLE_DATASETS
CHOSEN_METHODS = POSSIBLE_METHODS
CHOSEN_MODELS = POSSIBLE_MODELS
CHOSEN_EXPERIMENTS = POSSIBLE_EXPERIMENTS
