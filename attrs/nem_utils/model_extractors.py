import torch
import timm
from torch.nn.functional import interpolate


class resnet50_img_extractor(torch.nn.Module):
    def __init__(
        self,
        model=None,
    ):
        super().__init__()
        if model is None:
            model = timm.create_model("resnet50", pretrained=True)
        self.core_model = model

        self.channels = (3, 64, 256, 512, 1024, 2048)
        self.shapes = (
            112,
            56,
            28,
            14,
            7,
        )
        self.channels = (64, 256, 512, 1024, 2048)

        self.output_shape = (-1, 2048, 7, 7)
        self.use_img_space = False

    def forward(self, x: torch.Tensor):
        embs = []
        x = self.core_model.conv1(x)
        x = self.core_model.bn1(x)
        x = self.core_model.act1(x)
        embs += [x]
        x = self.core_model.maxpool(x)

        x = self.core_model.layer1(x)
        embs += [x]
        x = self.core_model.layer2(x)
        embs += [x]
        x = self.core_model.layer3(x)
        embs += [x]
        x = self.core_model.layer4(x)
        embs += [x]
        return embs

    def get_embeddings(self, x: torch.Tensor):
        return self.core_model.forward_features(x)

    def get_output_from_embeddings(self, features: torch.Tensor):
        return self.core_model.forward_head(features)

    def get_output(self, x: torch.Tensor):
        return self.core_model(x)


class convnext_extractor(torch.nn.Module):
    def __init__(
        self,
        model=None,
    ):
        super().__init__()
        if model is None:
            model = timm.create_model("convnext_small", pretrained=True)
        self.core_model = model
        self.shapes = (
            56,
            56,
            28,
            14,
            7,
        )
        self.channels = (96, 96, 192, 384, 768)
        self.output_shape = (-1, 768, 7, 7)
        self.use_img_space = False

    def forward(self, x: torch.Tensor):
        embs = []
        x = self.core_model.stem(x)
        embs.append(x)
        for stage in self.core_model.stages[:-1]:
            x = stage(x)
            embs.append(x)
        x = self.core_model.stages[-1](x)
        x = self.core_model.norm_pre(x)
        embs.append(x)
        return embs

    def get_embeddings(self, x: torch.Tensor):
        return self.core_model.forward_features(x)

    def get_output_from_embeddings(self, features: torch.Tensor):
        return self.core_model.forward_head(features)

    def get_output(self, x: torch.Tensor):
        return self.core_model(x)


class vgg_img_extractor(torch.nn.Module):
    def __init__(
        self,
        model=None,
    ):
        super().__init__()
        if model is None:
            model = timm.create_model("vgg16", pretrained=True)
        self.core_model = model
        self.shapes = (
            112,
            56,
            28,
            14,
            7,
        )
        self.channels = (64, 128, 256, 512, 512)
        self.output_shape = (-1, 512, 7, 7)
        self.use_img_space = False

    def forward(self, x: torch.Tensor):
        embs = []
        for i, layer in enumerate(self.core_model.features):
            x = layer(x)
            if i in [4, 9, 16, 23, 30]:
                embs.append(x)
        return embs

    def get_embeddings(self, x: torch.Tensor):
        return self.core_model.forward_features(x)

    def get_output_from_embeddings(self, features: torch.Tensor):
        return self.core_model.forward_head(features)

    def get_output(self, x: torch.Tensor):
        return self.core_model(x)


class vit_feature_extractor(torch.nn.Module):
    def __init__(
        self,
        core_model=None,
        shapes=(
            112,
            56,
            28,
            14,
            7,
        ),
        # channels=(12, 48, 192, 768, 3072),
        channels=(768, 768, 768, 768, 768),
        output_shape=(-1, 768, 14, 14),
    ):
        super().__init__()
        if core_model is None:
            core_model = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.core_model = core_model
        self.shapes = shapes
        self.channels = channels
        self.output_shape = output_shape
        self.pretrained_cfg = self.core_model.pretrained_cfg
        self.selected_layers = [2, 4, 6, 8, 10, 12]
        self.use_img_space = False

    def forward(self, x: torch.Tensor):
        out = self.core_model.get_intermediate_layers(
            x, self.selected_layers, reshape=True
        )
        return [
            interpolate(rep, size=shap, mode="nearest")
            for rep, shap in zip(out, self.shapes)
        ]

    # return [
    #     rep.reshape((-1, channel, shap, shap))
    #     for rep, shap, channel in zip(out, self.shapes, self.channels)
    # ]

    def get_embeddings(self, x: torch.Tensor):
        return self.core_model.forward_features(x)

    def get_output_from_embeddings(self, features: torch.Tensor):
        return self.core_model.forward_head(features)

    def get_output(self, x: torch.Tensor):
        return self.core_model(x)
