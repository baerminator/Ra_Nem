import torch
from .nem_utils.NEM import load_ranem
import numpy as np
from .attribution_template import attribution_template_class
import os


class ranem_atr(attribution_template_class):
    def __init__(
        self,
        model,
        train_data=None,
        use_predicted_labels=False,
    ):
        assert train_data is not None, "Ra-NEM requires train data"
        self.attr_name = "Ra-NEM"
        self.use_predicted_labels = use_predicted_labels
        self.model = model
        self.method = load_ranem(model, train_data)
        self.method.eval().cuda()

    def gen_attr(self, X, y):
        attr = self.method.masking_network(X)
        attr = attr.cpu().squeeze().detach().numpy()
        if np.all(attr < 0):
            attr = attr - attr.min()
        return attr

    def reinit(
        self,
        model,
        train_data=None,
    ):
        self.model = model
        self.method.masking_network.encoder.core_model = model
        self.method.eval().cuda()


class ranem_ranked_atr(ranem_atr):
    def gen_attr(self, X, y):
        attr = self.method.masking_network(X)
        attr = self.rank_batch_2d_tensor(attr.squeeze(1))
        return attr.cpu().squeeze().detach().numpy()

    def rank_batch_2d_tensor(self, batch):
        batch_size, height, width = batch.shape
        flat_tensor = batch.view(batch_size, -1)
        sorted_indices = torch.argsort(flat_tensor, dim=1)
        ranks = torch.empty_like(sorted_indices)
        ranks.scatter_(
            1,
            sorted_indices,
            torch.arange(flat_tensor.size(1), device=batch.device).expand_as(
                flat_tensor
            ),
        )
        return ranks.view(batch.shape)
