import torch
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from copy import deepcopy
from .attribution_template import attribution_template_class


def reshaper(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def select_layer_and_trans(model):
    if model.__class__.__name__ == "ResNet":
        return [model.layer4[-1]], None
    elif model.__class__.__name__ == "VGG":
        return [model.features[-1]], None
    elif model.__class__.__name__ == "ConvNeXt":
        return [model.stages[-1]], None
    elif model.__class__.__name__ == "VisionTransformer":
        return [model.blocks[-1].norm1], reshaper
    else:
        raise ValueError(f"Unsupported model {model.__class__.__name__}")


class gradcam_atr(attribution_template_class):
    def __init__(
        self,
        model,
        train_data=None,
        use_predicted_labels=False,
    ):
        self.attr_name = "gradcam"
        self.model = deepcopy(
            model
        )  # <--- This is a hack to avoid leaky memory, but it works
        self.use_predicted_labels = use_predicted_labels
        self.layer, self.reshape_trans = select_layer_and_trans(self.model)
        self.method = GradCAMPlusPlus(
            self.model, self.layer, reshape_transform=self.reshape_trans
        )

    def gen_attr(self, X, y):
        attr = self.method(
            input_tensor=X.requires_grad_(), targets=[ClassifierOutputTarget(y)]
        ).squeeze()
        return attr

    def reinit(self, model, train_data=None):
        self.model = deepcopy(model)
        self.layer, self.reshape_trans = select_layer_and_trans(self.model)
        self.method = GradCAMPlusPlus(
            self.model, self.layer, reshape_transform=self.reshape_trans
        )


class gradcam_ranked_atr(gradcam_atr):
    def __init__(
        self,
        model,
        train_data=None,
        use_predicted_labels=False,
    ):
        super().__init__(model, train_data, use_predicted_labels)
        self.attr_name = "gradcam_ranked"

    def gen_attr(self, X, y):
        attr = self.method(
            input_tensor=X, targets=[ClassifierOutputTarget(y)]
        ).squeeze()
        attr = self.rank_attr(attr)
        return attr
