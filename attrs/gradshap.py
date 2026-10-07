import torch
from captum.attr import GradientShap
from .attribution_template import attribution_template_class
from .util_forgrad import fg


def select_sigma(model):
    if model.__class__.__name__ == "ResNet":
        return 10
    elif model.__class__.__name__ == "VGG":
        return 10
    elif model.__class__.__name__ == "ConvNeXt":
        return 10
    elif model.__class__.__name__ == "VisionTransformer":
        return 10
    else:
        raise ValueError(f"Unsupported model {model.__class__.__name__}")


class gradshap_atr(attribution_template_class):
    def __init__(
        self,
        model,
        train_data=None,
        use_predicted_labels=False,
    ):
        self.attr_name = "gradient_shap"
        self.use_predicted_labels = use_predicted_labels
        self.model = model
        self.method = GradientShap(model)
        self.baseline = torch.zeros((1, 3, 224, 224)).to("cuda")
        self.sigma = select_sigma(model)

    def gen_attr(self, X, y):
        return (
            self.method.attribute(X, baselines=self.baseline, target=y, stdevs=0.1)
            .squeeze()
            .detach()
            .mean(0)
            .clamp(min=0)
            .cpu()
            .numpy()
        )

    def reinit(
        self,
        model,
        train_data=None,
    ):
        self.method = GradientShap(model)


class gradshap_forgrad_atr(gradshap_atr):
    def __init__(
        self,
        model,
        train_data=None,
        use_predicted_labels=False,
    ):
        super().__init__(model, train_data, use_predicted_labels)
        self.attr_name = "gradient_shap_forgrad"

    def gen_attr(self, X, y):
        attr = (
            self.method.attribute(X, baselines=self.baseline, target=y, stdevs=0.1)
            .squeeze()
            .detach()
            .mean(0)
            .clamp(min=0)
            .unsqueeze(0)
        )
        attr = fg(attr, sigma=self.sigma).squeeze().cpu().numpy()
        return attr


class gradshap_ranked_atr(gradshap_atr):
    def __init__(
        self,
        model,
        train_data=None,
        use_predicted_labels=False,
    ):
        super().__init__(model, train_data, use_predicted_labels)
        self.attr_name = "gradient_shap_ranked"

    def gen_attr(self, X, y):
        attr = (
            self.method.attribute(X, baselines=self.baseline, target=y, stdevs=0.1)
            .squeeze()
            .detach()
            .mean(0)
            .clamp(min=0)
            .cpu()
            .numpy()
        )
        attr = self.rank_attr(attr)
        return attr


class gradshap_forgrad_ranked_atr(gradshap_forgrad_atr):
    def __init__(
        self,
        model,
        train_data=None,
        use_predicted_labels=False,
    ):
        super().__init__(model, train_data, use_predicted_labels)
        self.attr_name = "gradient_shap_forgrad_ranked"

    def gen_attr(self, X, y):
        attr = (
            self.method.attribute(X, baselines=self.baseline, target=y, stdevs=0.1)
            .squeeze()
            .detach()
            .mean(0)
            .clamp(min=0)
            .unsqueeze(0)
        )
        attr = fg(attr, sigma=self.sigma).squeeze().cpu().numpy()
        attr = self.rank_attr(attr)
        return attr
