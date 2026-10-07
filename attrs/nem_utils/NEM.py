from torch import nn
import pytorch_lightning as pl
import torch
from .unet import Unet
from prodigyopt import Prodigy
from kornia.filters import filter2d
import torch.nn.functional as F
from exp_utils.metric_util import gkern
import numpy as np
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .model_extractors import *
from nem_config import TRAIN_BATCH_SIZE, EPOCHS, MIXED_PRECISION_TRAINING
import os
import time


def subsetoperator(
    scores,
    k,
    tau=1,
    revert=False,
    training=False,
    cumsum_ranking=False,
    gumbel_inference=False,
    epoch=10,
    samples=6,
):
    score_shape = scores.shape
    scores = scores.view(scores.size(0), -1)
    scores = scores - scores.logsumexp(dim=1).unsqueeze(1)
    if training or gumbel_inference:
        m = torch.distributions.gumbel.Gumbel(
            torch.zeros_like(scores), torch.ones_like(scores)
        )
        g = m.sample()
        scores = scores + g

    val_sorted, val_indices = torch.sort(scores)

    mask_k = torch.zeros_like(val_sorted)
    if training:
        size = score_shape[-1] * score_shape[-2]
        bin_size = size // samples
        sizes = torch.tensor([int(size - bin_size * i) for i in range(1, samples + 1)])
        bin_add = torch.rand(len(sizes)) * bin_size
        k_values = (bin_add + sizes).type(torch.int)
        # k_values = torch.randint(low=1,  high=int(val_sorted.size(1))  , size=( mask_k.size(0),))
        # Vectorized, out-of-place version to avoid in-place ops
        batch = mask_k.size(0)
        j_idx = torch.arange(batch, device=k_values.device) // int(batch / samples)
        k_idx = k_values[j_idx]
        # Subtract the threshold from each row
        threshold = val_sorted[torch.arange(batch), -k_idx]
        val_sorted = val_sorted - threshold.unsqueeze(1)
        # Set mask_k for top-k elements
        arange_idx = (
            torch.arange(val_sorted.size(1), device=val_sorted.device)
            .unsqueeze(0)
            .expand(batch, -1)
        )
        mask_k = (
            arange_idx
            >= (val_sorted.size(1) - k_idx.to(arange_idx.device).unsqueeze(1))
        ).float()
    else:
        mask_k[:, -k:] = 1
    # Given we have resetted at the k value, using the sigmoid is approximating a thresholding function
    val_sorted = torch.sigmoid(val_sorted)
    mask = val_sorted - val_sorted.detach()

    hardmask = mask + mask_k
    indices = val_indices + (
        torch.arange(scores.size(0)).view(-1, 1) * scores.size(1)
    ).to(val_sorted.device)

    new_mask_soft = torch.zeros_like(val_sorted, dtype=val_sorted.dtype)
    new_mask_soft.view(-1)[indices.view(-1)] = val_sorted.view(-1)
    soft_mask = new_mask_soft.view(score_shape)

    new_mask_hard = torch.zeros_like(val_sorted)
    new_mask_hard.view(-1)[indices.view(-1)] = hardmask.view(-1)
    hard_mask = new_mask_hard.view(score_shape)

    return hard_mask, soft_mask


def cosine_mean(x1, x2):
    return (
        1 - torch.mean(F.cosine_similarity(x1, x2, dim=-1))
    ) / 2  # 1 is close 0 is far so say 1 - sim


class EQ_dist(torch.nn.Module):
    def __init__(self):
        super(EQ_dist, self).__init__()

    def forward(self, target, pred_mask, pred, sep=False):
        pred = F.softmax(pred, dim=-1)
        pred_mask = F.softmax(pred_mask, dim=-1)
        if sep:
            return (pred_mask[:, target] - pred[:, target] + 1e-8).pow(2)
        return (pred_mask[:, target] - pred[:, target] + 1e-8).pow(2).mean()


class MAX_dist(torch.nn.Module):
    def __init__(self):
        super(MAX_dist, self).__init__()

    def forward(self, target, pred_mask, pred, sep=False):
        pred_mask = F.softmax(pred_mask, dim=-1)
        if sep:
            return 1 - pred_mask[:, target]
        return (1 - pred_mask[:, target]).mean()


def max_target_dist(target, pred_mask, pred):
    return (1 - torch.softmax(pred_mask, dim=-1)[:, target]).mean()


def cross_entropy(target, pred_mask, pred):
    return F.cross_entropy(pred_mask, pred.softmax(dim=-1)).mean()


class Masking_Loss(nn.Module):
    def __init__(
        self, constrastive=False, supervised=False, inversed=False, use_ranking=False
    ):
        super(
            Masking_Loss,
            self,
        ).__init__()
        self.constrastive = constrastive
        self.supervised = supervised

        self.n_cor = 1e-6
        self.inverse = inversed
        self.supervised_ratio = 50
        self.unsupervised_ratio = 1
        self.use_ranking = use_ranking

        self.measure_emb = cosine_mean
        self.measure_output = MAX_dist()

    def unsupervised_loss(
        self, masks, true_vector_emb, masked_vector_emb, neg_masked_vector_emb
    ):
        masking_ratio = torch.mean((masks))
        pdist_emb = self.measure_emb(true_vector_emb, masked_vector_emb)
        if self.use_ranking:
            loss = pdist_emb
            if self.inverse:
                loss = 2 - loss
            return loss, pdist_emb, masking_ratio

        if self.constrastive:
            ndist_emb = self.measure_emb(masked_vector_emb, neg_masked_vector_emb)
            loss = masking_ratio + (ndist_emb - 1.5) * pdist_emb
        else:
            loss = 4 * pdist_emb + 1 / 2 * masking_ratio

        return loss, pdist_emb, masking_ratio

    def supervised_loss(
        self, masks, true_vector_output, masked_vector_output, target, sep=False
    ):
        if target is None:
            target = true_vector_output.argmax(dim=-1)

        masking_ratio = torch.mean((masks))
        pdist = self.measure_output(
            target, masked_vector_output, true_vector_output, sep=sep
        )
        if self.use_ranking:
            if sep:
                loss = pdist.mean()
            else:
                loss = pdist
            masking_ratio = torch.mean(torch.sigmoid(masks))
            if self.inverse:
                loss = 1 - loss
            return loss, pdist, masking_ratio

        loss = masking_ratio + self.supervised_ratio * pdist

        if self.inverse:
            loss = 2 - loss
        return loss, pdist, masking_ratio

    def forward(
        self,
        masks,
        true_vector_emb,
        true_vector_output,
        masked_vector_emb,
        masked_vector_output,
        neg_masked_vector_emb,
        neg_masked_vector_output,
        target=None,
        sep=False,
    ):
        if self.supervised:
            return self.supervised_loss(
                masks=masks,
                true_vector_output=true_vector_output,
                masked_vector_output=masked_vector_output,
                target=target,
                sep=sep,
            )
        return self.unsupervised_loss(
            masks, true_vector_emb, masked_vector_emb, neg_masked_vector_emb
        )


class masking_network(pl.LightningModule):
    def __init__(
        self,
        epochs,
        batch_size,
        lr=1,
        img_size=(224, 224),
        partition=1,
        noise_mask=False,
        blur_mask=False,
        blur=False,
        constrastive=False,
        variational=False,
        supervised=False,
        use_real_target=False,
        inverse=False,
        use_random_filter=False,
        use_reinmax=False,
        use_ranking=False,
        use_scale_space_partition=False,
        cumsum_ranking=False,
        complex_reduction=None,
    ):
        super().__init__()
        assert not (use_reinmax & use_ranking), (
            "Cannot use Reinmax and Ranking at the same time"
        )
        assert not (use_ranking & constrastive), (
            "Cannot use Ranking and Constrastive at the same time"
        )
        assert not (use_ranking & inverse), (
            "Cannot use Ranking and Inverse at the same time"
        )

        self.noise_mask = noise_mask
        self.blur_mask = blur_mask
        self.blur_kernel = torch.Tensor(gkern(11, 5))
        self.negative = constrastive
        self.patcher = partition is not None
        self.blur = blur
        self.partition = partition
        self.use_scale_space_partition = use_scale_space_partition

        self.samples = 6 if use_ranking else 1
        self.tau_start = 3 if use_ranking else 5
        self.tau_min = 0.1 if use_ranking else 1
        self.tau_reduction = 0.33 if use_ranking else 0.05
        sample_reduction = (
            1 if use_scale_space_partition or partition == None else partition
        )
        self.top_k_val = int(0.05 * 224 * 224 / sample_reduction) if use_ranking else 0
        self.topk_revert = False
        self.tau = self.tau_start
        self.rand_kernel_size = 21
        self.cumsum_ranking = cumsum_ranking
        self.complex_reduction = complex_reduction
        self.gumbel_inference = False

        self.supervised = supervised
        self.use_reinmax = use_reinmax
        self.use_ranking = use_ranking
        self.supervised_use_real_target = use_real_target & supervised

        self.use_random_filter = use_random_filter
        self.loss_func = Masking_Loss(
            constrastive=constrastive,
            supervised=supervised,
            inversed=inverse,
            use_ranking=use_ranking,
        )
        self.learning_rate = lr
        self.epochs = epochs

        self.img_size = img_size
        self.batch_size = batch_size
        if self.patcher:
            self.patching_layer = nn.Conv2d(
                1, 1, kernel_size=partition, stride=partition, padding=0
            )
        self.masking_network = None
        self.frozen_network = None
        self.global_pool = None

    def gen_mask(self, x, mask=None):
        # GENERATE MASK LOGITS
        if mask is None:
            mask = self.masking_network(x)
        if self.patcher and not self.use_scale_space_partition:
            mask = self.patching_layer(mask)

        ### TRANSFORM MASK LOGITS
        if self.supervised & self.use_random_filter:
            rand_kernel = torch.rand(
                size=(1, self.rand_kernel_size, self.rand_kernel_size)
            )
            mask = filter2d(mask, rand_kernel, normalized=True, border_type="circular")

        ### CREATE MULTIPLE SAMPLES FOR RANKING
        if self.training and self.use_ranking:
            mask = mask.repeat(self.samples, 1, 1, 1).reshape(-1, *mask.shape[1:])
            x = x.repeat(self.samples, 1, 1, 1).reshape(-1, *x.shape[1:])

        ### CREATE APPLIED MASK

        if self.use_ranking:
            applied_mask, soft_mask = subsetoperator(
                mask.permute(0, 2, 3, 1).squeeze(3),
                self.top_k_val,
                tau=self.tau,
                training=self.training,
                revert=self.topk_revert,
                gumbel_inference=self.gumbel_inference,
                cumsum_ranking=self.cumsum_ranking,
                epoch=self.current_epoch,
                samples=self.samples,
            )
            applied_mask = applied_mask.unsqueeze(3).permute(0, 3, 1, 2)
        else:
            mask = torch.sigmoid(mask)
            applied_mask = mask

        ### APPLY MASK
        if self.noise_mask:
            device = next(self.masking_network.parameters()).device
            noise = torch.normal(0, 1, size=x.shape).to(device)
            data_mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=device)[
                None, :, None, None
            ]
            data_std = torch.tensor(IMAGENET_DEFAULT_STD, device=device)[
                None, :, None, None
            ]
            perturbation = noise * data_std + data_mean
        elif self.blur:
            perturbation = torch.nn.functional.conv2d(
                x, self.blur_kernel.type(x.dtype).to(x.device), padding=11 // 2
            )
        else:
            perturbation = 0 * x

        x_masked = x * applied_mask + (1 - applied_mask) * perturbation
        if self.negative:
            self.negative_img = x * (1 - applied_mask) + (applied_mask) * perturbation
        return mask, x_masked, applied_mask

    def gen_representations(self, x, x_masked, pred_emb=None, pred=None):
        if pred_emb is None:
            pred_emb = self.frozen_network.get_embeddings(x)
        pred_masked_emb = self.frozen_network.get_embeddings(x_masked)
        pred_neg_emb = (
            self.frozen_network.get_embeddings(self.negative_img)
            if self.negative
            else None
        )
        if self.supervised:
            if pred is None:
                pred = self.frozen_network.get_output_from_embeddings(pred_emb)
            pred_neg = (
                self.frozen_network.get_output_from_embeddings(pred_neg_emb)
                if self.negative
                else None
            )
            pred_masked = self.frozen_network.get_output_from_embeddings(
                pred_masked_emb
            )
        else:
            pred = pred_emb
            pred_neg = pred_neg_emb
            pred_masked = pred_masked_emb

        if self.training and self.use_ranking:
            if self.supervised:
                pred = pred.repeat(self.samples, 1, 1).reshape(-1, *pred.shape[1:])
            else:
                pred_emb = pred
                pred_masked_emb = pred_masked
                pred_emb = pred_emb.repeat(self.samples, 1, 1).reshape(
                    -1, *pred_emb.shape[1:]
                )

        return pred_emb, pred, pred_masked_emb, pred_masked, pred_neg_emb, pred_neg

    def run_step(self, x, target=None):
        mask, x_masked, applied_mask = self.gen_mask(x)
        pred_emb, pred, pred_masked_emb, pred_masked, pred_neg_emb, pred_neg = (
            self.gen_representations(x=x, x_masked=x_masked)
        )
        loss, pdist, masking_ratio = self.loss_func(
            mask,
            pred_emb,
            pred,
            pred_masked_emb,
            pred_masked,
            pred_neg_emb,
            pred_neg,
            target=target,
        )
        return loss, pdist, masking_ratio, pred, pred_masked

    def training_step(self, batch, batch_idx):
        if self.use_ranking:
            masking_choices = ["remove", "blur", "noise"]
            masking = masking_choices[torch.randint(0, 3, (1,)).item()]
            self.noise_mask = masking == "noise"
            self.blur = masking == "blur"

        if len(batch) == 2:
            x, y = batch
        else:
            x = batch

        if self.masking_network.freeze_backbone:
            self.masking_network.encoder.eval()

        if self.use_ranking and self.training and self.supervised:
            y = y.repeat(self.samples, 1, 1, 1).reshape(-1, *y.shape[1:])

        target = y if self.supervised_use_real_target else None
        loss, pdist, masking_ratio, _, _ = self.run_step(x, target=target)
        self.log("train_loss", loss, batch_size=self.batch_size)
        self.log("train_mask_norm", masking_ratio, batch_size=self.batch_size)
        self.log("train_dist", pdist, batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 2:
            x, y = batch
        else:
            x = batch
        loss, pdist, masking_ratio, _, _ = self.run_step(
            x, target=y if self.supervised else None
        )
        self.log("val_loss", loss, batch_size=self.batch_size)
        self.log("val_mask_norm", masking_ratio, batch_size=self.batch_size)
        self.log("val_dist", pdist, batch_size=self.batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss, pdist, masking_ratio, _, _ = self.run_step(
            x, target=y if self.supervised else None
        )
        self.log("test_loss", loss, batch_size=self.batch_size)
        self.log("test_mask_norm", masking_ratio, batch_size=self.batch_size)
        self.log("test_dist", pdist, batch_size=self.batch_size)
        return loss

    def forward(self, x):
        return self.gen_mask(x)

    def configure_optimizers(self):
        optim_spec = lambda params: Prodigy(params, weight_decay=1e-4)

        # Only optimize decoder if backbone is frozen
        if self.masking_network.freeze_backbone:
            if self.patcher:
                optimizer = optim_spec(
                    list(self.masking_network.decoder.parameters())
                    + list(self.patching_layer.parameters())
                )
            else:
                optimizer = optim_spec(self.masking_network.decoder.parameters())
        else:
            optimizer = optim_spec(self.masking_network.parameters())
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                20,
                eta_min=1e-6,
            ),
            "interval": "epoch",  # Adjust learning rate every epoch
            "frequency": 1,  # How often to apply the scheduler
        }
        return [optimizer], [scheduler]


class resnet50_nem(masking_network):
    def __init__(
        self,
        explained_model,
        epochs,
        batch_size,
        lr=1.0,
        center=False,
        partition=None,
        noise_mask=False,
        constrastive=False,
        inverse=False,
        use_random_filter=False,
        supervised=True,
        use_real_target=False,
        use_reinmax=False,
        use_ranking=False,
        use_scale_space_partition=False,
        cumsum_ranking=False,
    ):
        super().__init__(
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            partition=partition,
            constrastive=constrastive,
            noise_mask=noise_mask,
            supervised=supervised,
            inverse=inverse,
            use_random_filter=use_random_filter,
            use_real_target=use_real_target,
            use_reinmax=use_reinmax,
            use_ranking=use_ranking,
            use_scale_space_partition=use_scale_space_partition,
            cumsum_ranking=cumsum_ranking,
        )

        backbone = resnet50_img_extractor(explained_model)
        self.reshaper = backbone.output_shape
        encoder_channels = backbone.channels
        decoder_channels = (256, 128, 64, 32, 16)
        self.masking_network = Unet(
            num_classes=1,
            backbone=backbone,
            freeze_backbone=True,
            center=center,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            use_img_space=backbone.use_img_space,
        )
        self.frozen_network = backbone
        # print(f"Running {supervised} resnet50")


class vgg16_nem(masking_network):
    def __init__(
        self,
        explained_model,
        epochs,
        batch_size,
        lr=1.0,
        center=False,
        partition=None,
        noise_mask=False,
        constrastive=False,
        inverse=False,
        use_random_filter=False,
        supervised=True,
        use_real_target=False,
        use_reinmax=False,
        use_ranking=False,
        use_scale_space_partition=False,
        cumsum_ranking=False,
    ):
        super().__init__(
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            partition=partition,
            constrastive=constrastive,
            noise_mask=noise_mask,
            supervised=supervised,
            inverse=inverse,
            use_random_filter=use_random_filter,
            use_real_target=use_real_target,
            use_reinmax=use_reinmax,
            use_ranking=use_ranking,
            use_scale_space_partition=use_scale_space_partition,
            cumsum_ranking=cumsum_ranking,
        )

        backbone = vgg_img_extractor(explained_model)
        self.reshaper = backbone.output_shape
        encoder_channels = backbone.channels
        decoder_channels = (256, 128, 64, 32, 16)
        self.masking_network = Unet(
            num_classes=1,
            backbone=backbone,
            freeze_backbone=True,
            center=center,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            use_img_space=backbone.use_img_space,
        )
        self.frozen_network = backbone
        # print(f"Running {supervised} vgg16")


class convnext_nem(masking_network):
    def __init__(
        self,
        explained_model,
        epochs,
        batch_size,
        lr=1.0,
        center=False,
        partition=None,
        noise_mask=False,
        constrastive=False,
        inverse=False,
        use_random_filter=False,
        supervised=True,
        use_real_target=False,
        use_reinmax=False,
        use_ranking=False,
        use_scale_space_partition=False,
        cumsum_ranking=False,
    ):
        super().__init__(
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            partition=partition,
            constrastive=constrastive,
            noise_mask=noise_mask,
            supervised=supervised,
            inverse=inverse,
            use_random_filter=use_random_filter,
            use_real_target=use_real_target,
            use_reinmax=use_reinmax,
            use_ranking=use_ranking,
            use_scale_space_partition=use_scale_space_partition,
            cumsum_ranking=cumsum_ranking,
        )

        backbone = convnext_extractor(explained_model)
        self.reshaper = backbone.output_shape
        encoder_channels = backbone.channels
        decoder_channels = (256, 128, 64, 32, 16)
        self.masking_network = Unet(
            num_classes=1,
            backbone=backbone,
            freeze_backbone=True,
            center=center,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            use_img_space=backbone.use_img_space,
        )
        self.frozen_network = backbone
        # print(f"Running {supervised} convnext")


class vit_nem(masking_network):
    def __init__(
        self,
        explained_model,
        epochs,
        batch_size,
        lr=1.0,
        center=False,
        partition=None,
        noise_mask=False,
        constrastive=False,
        inverse=False,
        use_random_filter=False,
        supervised=True,
        use_real_target=False,
        use_reinmax=False,
        use_ranking=False,
        use_scale_space_partition=False,
        cumsum_ranking=False,
    ):
        super().__init__(
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            partition=partition,
            constrastive=constrastive,
            noise_mask=noise_mask,
            supervised=supervised,
            inverse=inverse,
            use_random_filter=use_random_filter,
            use_real_target=use_real_target,
            use_reinmax=use_reinmax,
            use_ranking=use_ranking,
            use_scale_space_partition=use_scale_space_partition,
            cumsum_ranking=cumsum_ranking,
        )

        backbone = vit_feature_extractor(explained_model)
        self.reshaper = backbone.output_shape
        encoder_channels = backbone.channels
        decoder_channels = (256, 128, 64, 32, 16)
        self.masking_network = Unet(
            num_classes=1,
            backbone=backbone,
            freeze_backbone=True,
            center=center,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            use_img_space=backbone.use_img_space,
        )
        self.frozen_network = backbone
        # print(f"Running {supervised} vit")


def select_nem(model):
    if model.__class__.__name__ == "ResNet":
        return resnet50_nem
    elif model.__class__.__name__ == "VGG":
        return vgg16_nem
    elif model.__class__.__name__ == "ConvNeXt":
        return convnext_nem
    elif model.__class__.__name__ == "VisionTransformer":
        return vit_nem
    else:
        raise ValueError(f"Unsupported model {model.__class__.__name__}")


def train_nem(nem, data, log_path=None):
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        devices="auto",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=16 if MIXED_PRECISION_TRAINING else 32,
        default_root_dir=log_path,
    )
    trainer.fit(nem, data)


def load_ranem(model, train_data):
    nem_constructor = select_nem(model)
    log_path = f"attrs/nem_utils/logs/ranem/{model.__class__.__name__}"
    if os.path.exists(log_path):
        checkpoint_path = f"{log_path}/lightning_logs/version_0/checkpoints/"
        checkpoint_path = os.path.join(checkpoint_path, os.listdir(checkpoint_path)[0])
        ranem = nem_constructor.load_from_checkpoint(
            checkpoint_path,
            epochs=EPOCHS,
            batch_size=TRAIN_BATCH_SIZE,
            explained_model=model,
            use_ranking=True,
            strict=False,
        )
    else:
        ranem = nem_constructor(
            model, EPOCHS, batch_size=TRAIN_BATCH_SIZE, use_ranking=True
        )
        start_time = time.time()
        print("Training")
        print(f"Storing data at {log_path}")
        train_nem(ranem, train_data, log_path)
        total_time = time.time() - start_time
        print(f"Training time {total_time}")
        np.save(f"{log_path}/training_time.npy", total_time)
    return ranem


def load_nemt(model, train_data):
    nem_constructor = select_nem(model)
    log_path = f"attrs/nem_utils/logs/nemt/{model.__class__.__name__}"
    if os.path.exists(log_path):
        checkpoint_path = f"{log_path}/lightning_logs/version_0/checkpoints/"
        checkpoint_path = os.path.join(checkpoint_path, os.listdir(checkpoint_path)[0])
        nemt = nem_constructor.load_from_checkpoint(
            checkpoint_path,
            epochs=EPOCHS,
            batch_size=TRAIN_BATCH_SIZE,
            explained_model=model,
            inverse=True,
            use_random_filter=True,
            strict=False,
        )
    else:
        nemt = nem_constructor(
            model,
            EPOCHS,
            batch_size=TRAIN_BATCH_SIZE,
            inverse=True,
            use_random_filter=True,
        )
        start_time = time.time()
        print("Training")
        print(f"Storing data at {log_path}")
        train_nem(nemt, train_data, log_path)
        total_time = time.time() - start_time
        print(f"Training time {total_time}")
        np.save(f"{log_path}/training_time.npy", total_time)
    return nemt
