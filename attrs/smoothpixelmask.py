import torch
import warnings
from torchray.attribution.extremal_perturbation import extremal_perturbation, contrastive_reward
from .attribution_template import attribution_template_class

class SmoothMask:
        def __init__(self, area, model):
            self.area = area 
            self.model = model
        def __call__(self, x, pred):
            mask, _ = extremal_perturbation(
                self.model, x, pred,
                reward_func=contrastive_reward,
                debug=False,
                areas=[self.area]
            )
            return mask

class smoothpixelmask_atr(attribution_template_class):
    def __init__(self, model, train_data = None, use_predicted_labels = False, ):
        self.attr_name = "smooth_pixel_mask"
        self.use_predicted_labels = use_predicted_labels 
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        self.softmax = torch.nn.Softmax(dim=-1)
        warnings.filterwarnings("ignore", category=UserWarning)
        self.method = SmoothMask(model =model, area=0.10)
    def gen_attr(self,  X, y):
        attr = self.method(X, int(y)).squeeze().detach().cpu().numpy()
        return attr
    def reinit(self,model, train_data = None):
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        self.softmax = torch.nn.Softmax(dim=-1)
        warnings.filterwarnings("ignore", category=UserWarning)
        self.method = SmoothMask(model =model, area=0.10)

class smoothpixelmask_ranked_atr(smoothpixelmask_atr):
    def __init__(self, model, train_data = None, use_predicted_labels = False, ):
        super().__init__(model, train_data, use_predicted_labels)
        self.attr_name = "smooth_pixel_mask_ranked"
    def gen_attr(self,  X, y):
        attr = self.method(X, int(y)).squeeze().detach().cpu().numpy()
        attr = self.rank_attr(attr)
        return attr