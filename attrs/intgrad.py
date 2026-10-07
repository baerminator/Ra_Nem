import torch
from captum.attr import IntegratedGradients
from .attribution_template import attribution_template_class
from .util_forgrad import fg


def select_sigma(model):
    if model.__class__.__name__ == "ResNet":
        return 12
    elif model.__class__.__name__ == "VGG":
        return 15
    elif model.__class__.__name__ == "ConvNeXt":
        return 55
    elif model.__class__.__name__ == "VisionTransformer":
        return 110
    else:
        raise ValueError(f"Unsupported model {model.__class__.__name__}")


class itg_atr(attribution_template_class):
    def __init__(
        self,
        model,
        train_data=None,
        use_predicted_labels=False,
    ):
        self.attr_name = "integrated_gradient"
        self.use_predicted_labels = use_predicted_labels
        self.model = (model).eval().cuda()
        self.method = IntegratedGradients(self.model)
        self.baseline = torch.zeros((1, 3, 224, 224)).cuda()
        self.sigma = select_sigma(model)

    def gen_attr(self, X, y):
        attr = (
            self.method.attribute(X, baselines=self.baseline, target=y)
            .squeeze()
            .detach()
            .mean(0)
            .clamp(min=0)
            .cpu()
            .numpy()
        )
        return attr

    def reinit(
        self,
        model,
        train_data=None,
    ):
        self.model = (model).eval().cuda()
        self.method = IntegratedGradients(self.model)


class itg_forgrad_atr(itg_atr):
    def __init__(
        self,
        model,
        train_data=None,
        use_predicted_labels=False,
    ):
        super().__init__(model, train_data, use_predicted_labels)
        self.attr_name = "integrated_gradient_forgrad"

    def gen_attr(self, X, y):
        attr = (
            self.method.attribute(X, baselines=self.baseline, target=y)
            .squeeze()
            .detach()
            .mean(0)
            .clamp(min=0)
            .unsqueeze(0)
        )
        attr = fg(attr, sigma=self.sigma).squeeze().cpu().numpy()
        return attr


class itg_ranked_atr(itg_atr):
    def __init__(
        self,
        model,
        train_data=None,
        use_predicted_labels=False,
    ):
        super().__init__(model, train_data, use_predicted_labels)
        self.attr_name = "integrated_gradient_ranked"

    def gen_attr(self, X, y):
        attr = (
            self.method.attribute(X, baselines=self.baseline, target=y)
            .squeeze()
            .detach()
            .mean(0)
            .clamp(min=0)
            .cpu()
            .numpy()
        )
        attr = self.rank_attr(attr)
        return attr


class itg_forgrad_ranked_atr(itg_forgrad_atr):
    def __init__(
        self,
        model,
        train_data=None,
        use_predicted_labels=False,
    ):
        super().__init__(model, train_data, use_predicted_labels)
        self.attr_name = "integrated_gradient_forgrad_ranked"

    def gen_attr(self, X, y):
        attr = (
            self.method.attribute(X, baselines=self.baseline, target=y)
            .squeeze()
            .detach()
            .mean(0)
            .clamp(min=0)
            .unsqueeze(0)
        )
        attr = fg(attr, sigma=self.sigma).squeeze().cpu().numpy()
        attr = self.rank_attr(attr)
        return attr
