import torch
from rise import RISE
from .attribution_template import attribution_template_class
class rs_atr(attribution_template_class):
    def __init__(self, model, train_data = None, use_predicted_labels = False, ):
        self.attr_name = "rise"
        self.use_predicted_labels = use_predicted_labels 
        self.model = model
        self.method = RISE(model)
        self.baseline = torch.zeros((1, 3, 224, 224)).cuda()
        self.n_mask = 2**12
        self.initial_mask_shape = (7,7) 
    def gen_attr(self,  X, y):  
         return self.method.attribute(
              X,baselines=self.baseline,
              target= y,
              n_masks=self.n_mask, 
              initial_mask_shapes=(self.initial_mask_shape,)).squeeze().detach().cpu().numpy()
    def reinit(self,model, train_data = None):
        self.model = model
        self.method = RISE(model)
class rs_ranked_atr(rs_atr):
    def __init__(self, model, train_data = None, use_predicted_labels = False, ):
        super().__init__(model, train_data, use_predicted_labels)
        self.attr_name = "rise_ranked"
    def gen_attr(self,  X, y):  
         attr = self.method.attribute(
              X,baselines=self.baseline,
              target= y,
              n_masks=self.n_mask, 
              initial_mask_shapes=(self.initial_mask_shape,)).squeeze().detach().cpu().numpy()
         attr = self.rank_attr(attr)
         return attr