import torch
from .attribution_template import attribution_template_class
from .nem_utils.NEM import load_nemt
class nemt_atr(attribution_template_class):
    def __init__(self, model, train_data = None, use_predicted_labels = False, ):
        assert train_data is not None, "nemt requires train data"
        self.attr_name = "Nemt"
        self.use_predicted_labels = use_predicted_labels 
        self.model = model
        self.method = load_nemt(model, train_data)
        self.method.eval().cuda()
    def gen_attr(self,  X, y):
        attr, _ ,_ = self.method(X)
        attr = 1 - attr # <-- given that NEMT returns inverted attributions
        return attr.cpu().squeeze().detach().numpy()
    def reinit(self,model,train_data = None,):
        self.model = model
        self.method.masking_network.encoder.core_model = model
        self.method.eval().cuda()
        
class nemt_ranked_atr(nemt_atr):
    def __init__(self, model, train_data = None, use_predicted_labels = False, ):
        super().__init__(model, train_data, use_predicted_labels)
        self.attr_name = "Nemt_ranked"
    def gen_attr(self,  X, y):
        attr, _ ,_ = self.method(X)
        attr = 1 - attr # <-- given that NEMT returns inverted attributions
        attr = attr.cpu().squeeze().detach().numpy()
        attr = self.rank_attr(attr)
        return attr