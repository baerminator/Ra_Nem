import torch
from captum.attr import Saliency
from .attribution_template import attribution_template_class
from .util_forgrad import fg

def select_sigma(model):
    if model.__class__.__name__ == "ResNet":
        return 2
    elif model.__class__.__name__ == "VGG":
        return 5
    elif model.__class__.__name__ == "ConvNeXt":
        return 3
    elif model.__class__.__name__ == "VisionTransformer":
        return 12
    else:
        raise ValueError(f"Unsupported model {model.__class__.__name__}")


class saliency_atr(attribution_template_class):
    def __init__(self, model, train_data = None, use_predicted_labels = False, ):
        self.attr_name = "saliency"
        self.method = Saliency(model) 
        self.use_predicted_labels = use_predicted_labels 
        self.model = model
        self.sigma = select_sigma(model)
    def gen_attr(self,  X, y):
        attr = self.method.attribute(X,target= y).squeeze().detach().mean(0).cpu().numpy()
        return attr
    def reinit(self,model, train_data = None):
        self.model = model
        self.method = Saliency(model) 
class saliency_forgrad_atr(saliency_atr):
    def __init__(self, model, train_data = None, use_predicted_labels = False, ):
        super().__init__(model, train_data, use_predicted_labels)
        self.attr_name = "saliency_forgrad"
    def gen_attr(self,  X, y):
        attr = self.method.attribute(X,target= y).squeeze().detach().mean(0).unsqueeze(0)
        attr = fg(attr, sigma=self.sigma).squeeze().cpu().numpy()
        return attr
    
class saliency_ranked_atr(saliency_atr):
    def __init__(self, model, train_data = None, use_predicted_labels = False, ):
        super().__init__(model, train_data, use_predicted_labels)
        self.attr_name = "saliency_ranked"
    def gen_attr(self,  X, y):
        attr = self.method.attribute(X,target= y).squeeze().detach().mean(0).cpu().numpy()
        attr = self.rank_attr(attr)
        return attr
class saliency_forgrad_ranked_atr(saliency_forgrad_atr):
    def __init__(self, model, train_data = None, use_predicted_labels = False, ):
        super().__init__(model, train_data, use_predicted_labels)
        self.attr_name = "saliency_forgrad_ranked"
    def gen_attr(self,  X, y):
        attr = self.method.attribute(X,target= y).squeeze().detach().mean(0).unsqueeze(0)
        attr = fg(attr, sigma=self.sigma).squeeze().cpu().numpy()
        attr = self.rank_attr(attr)
        return attr