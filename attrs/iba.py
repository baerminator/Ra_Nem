import torch
from .iba_utils.IBA import IBA
from .attribution_template import attribution_template_class
import torch.nn.utils as nn_utils
def select_layer(model):
    if model.__class__.__name__ == "ResNet":
        return model.layer4[-1]
    elif model.__class__.__name__ == "VGG":
        return model.features[-1]
    elif model.__class__.__name__ == "ConvNeXt":
        return model.stages[-1]
    elif model.__class__.__name__ == "VisionTransformer":
        return model.blocks[-1].norm1
    else:
        raise ValueError(f"Unsupported model {model.__class__.__name__}")

def construct_method(model, dataloader, samples = 10000):
    layer = select_layer(model)
    method = IBA(layer)
    method = method.cuda()
    method.estimate(model, dataloader,n_samples= samples,progbar=False)
    return method

class iba_atr(attribution_template_class):
    def __init__(self, model, train_data = None, use_predicted_labels = False, ):
        assert train_data is not None, "IBA requires train data"
        self.attr_name = "IBA"
        self.use_predicted_labels = use_predicted_labels 
        self.model = model
        if model.__class__.__name__ == "VisionTransformer":
            self.method = None
        else:
            self.method = construct_method(model, train_data.train_dataloader())
    def gen_attr(self,  X, y):
        model_loss_closure = lambda x: -torch.log_softmax(self.model(x), dim=-1)[:, y].mean()
        return self.method.analyze(X, model_loss_closure, beta=10)
    def reinit(self,model,train_data =None):
        self.model = model
        if model.__class__.__name__ == "VisionTransformer":
            self.method = None
        else:
            self.method = construct_method(model, train_data.train_dataloader())

class iba_ranked_atr(iba_atr):
    def gen_attr(self,  X, y):
        model_loss_closure = lambda x: -torch.log_softmax(self.model(x), dim=-1)[:, y].mean()
        attr = self.method.analyze(X, model_loss_closure, beta=10)
        attr = self.rank_attr(attr)
        return attr