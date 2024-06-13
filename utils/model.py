import pytorch_lightning as pl
import torch
from torch.nn.functional import mse_loss
import segmentation_models_pytorch as smp
from torch import nn
import torchvision.utils
import os
import numpy as np
from torchvision import models
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
from utils.losses import ArcFace
from pyeer.eer_info import get_eer_stats
from torch.nn.utils import clip_grad_norm_
import random
from numpy.linalg import norm

class CycleTattooTransformerNetwork(nn.Module):
    def __init__(self, model_name = 'resnet34', num_features = 512):
        super(CycleTattooTransformerNetwork, self).__init__()

        self.template_model = nn.Sequential(
            smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", classes=3, in_channels=3),
            nn.Tanh()
            )
        
        self.transformation_model = nn.Sequential(
            smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", classes=3, in_channels=3),
            nn.Tanh()
            )

        self.raw_model = models.get_model(model_name, weights="DEFAULT") #models.resnet34(pretrained=pretrained)
        if('mobilenet_v3_large' in model_name or 'efficientnet' in model_name):
            self.raw_model.classifier[-1] = nn.Linear(in_features=1280, out_features=num_features)
        elif('resnet' in model_name):
            self.raw_model.fc = nn.Linear(in_features=2048, out_features=num_features)
        elif('densenet' in model_name):
            self.raw_model.classifier = nn.Linear(in_features=1024, out_features=num_features)
        elif('swin' in model_name):
            self.raw_model.head = nn.Linear(in_features=768, out_features=num_features)

    def forward(self, x):
        temp = self.template_model(x), 
        img = self.transformation_model(temp[0])
        return temp[0], img, self.raw_model(temp[0])


class RawImageEmbedding(nn.Module):
    def __init__(self, model_name = 'resnet34', num_features = 512):
        super(RawImageEmbedding, self).__init__()

        self.raw_model = models.get_model(model_name, weights="DEFAULT") #models.resnet34(pretrained=pretrained)
        if('mobilenet_v3_large' in model_name or 'efficientnet' in model_name):
            self.raw_model.classifier[-1] = nn.Linear(in_features=1280, out_features=num_features)
        elif('resnet' in model_name):
            self.raw_model.fc = nn.Linear(in_features=2048, out_features=num_features)
        elif('densenet' in model_name):
            self.raw_model.classifier = nn.Linear(in_features=1024, out_features=num_features)
        elif('swin' in model_name):
            self.raw_model.head = nn.Linear(in_features=768, out_features=num_features)

    # combined
    def forward(self, x):
        raw_x = self.raw_model(x)
        return raw_x
        

class TTN_wrapper(pl.LightningModule):
    def __init__(self, model = 'resnet34', num_features = 512, num_identities = 100, s = 64, m = 0.5, L=4):
        super().__init__()
        self.backbone_trans = CycleTattooTransformerNetwork(model_name=model, num_features=num_features)
        self.backbone_raw = RawImageEmbedding(model_name=model, num_features=num_features)
        self.header = ArcFace(in_features=num_features, out_features=num_identities, s= s, m= m)
        self.criterion = CrossEntropyLoss()
        self.mse_criterion = MSELoss()
        # Important: This property activates manual optimization.
        # self.automatic_optimization = False
        self.L = L

    def forward(self, x):
        template, img, emb_template = self.backbone_trans(x)
        emb_image = self.backbone_raw(x)
        return template, img, emb_template, emb_image


    def training_step(self, batch, batch_idx):
        x, template, target = batch
        # opt_backbone_raw, opt_backbone_rec, opt_header = self.optimizers()
        # scheduler_backbone_raw, scheduler_backbone_rec, scheduler_header = self.lr_schedulers()

        recons_t, recons_i, emb_template = self.backbone_trans(x)
        emb_image = self.backbone_raw(x)

        thetas = self.header(F.normalize(emb_image), target)
        loss_feat_img = self.criterion(thetas, target)

        loss_t = self.mse_criterion(recons_t, template)
        loss_i = self.mse_criterion(recons_i, x)
        loss_rec = loss_t + loss_i

        thetas = self.header(F.normalize(emb_template), target)
        loss_feat_temp = self.criterion(thetas, target)

        loss = (loss_feat_img + loss_feat_temp + self.L*loss_rec)/3 #We can try with different lambda values

        #header and backbone optimisation
        # opt_backbone_rec.zero_grad()
        # opt_backbone_raw.zero_grad()
        # opt_header.zero_grad()
        # self.manual_backward(loss)

        clip_grad_norm_(self.backbone_raw.parameters(), max_norm=5, norm_type=2)
        # self.clip_gradients(opt_backbone_raw, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        # self.clip_gradients(opt_backbone_rec, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

        # opt_backbone_raw.step()
        # opt_backbone_rec.step()
        # opt_header.step()

        # step every N epochs
        # if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % 4 == 0:
        #     scheduler_backbone_raw.step()
        #     scheduler_backbone_rec.step()
        #     scheduler_header.step()
            
        self.log_dict({"train_loss": loss, "raw_features_loss": loss_feat_img, "template_features_loss": loss_feat_temp, "template_loss": self.L*loss_rec},
                          prog_bar=True)

        return loss
    

    def validation_step(self, batch, batch_idx):
        pass


    def configure_optimizers(self):
        params = list(self.backbone_raw.parameters())
        params += list(self.backbone_trans.parameters())
        params += list(self.header.parameters())

        return torch.optim.Adam(params, lr=1e-5) 
    #     opt_backbone_raw = torch.optim.Adam(self.backbone_raw.parameters(), lr=1e-5) 
    #     opt_backbone_rec = torch.optim.Adam(self.backbone_trans.parameters(), lr=1e-5) 
    #     opt_header = torch.optim.Adam(self.header.parameters(), lr=1e-5) 

    #     scheduler_backbone_raw = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_backbone_raw, gamma = 0.95)
    #     scheduler_backbone_rec = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_backbone_rec, gamma = 0.95)
    #     scheduler_header = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_header, gamma = 0.95)

    #     return (
    #     {
    #         "optimizer": opt_backbone_raw,
    #         "lr_scheduler": {
    #             "scheduler": scheduler_backbone_raw,
    #             "interval": "epoch",
    #             "frequency": 1
    #         },
    #     },
    #     {
    #         "optimizer": opt_backbone_rec,
    #         "lr_scheduler": {
    #             "scheduler": scheduler_backbone_rec,
    #             "interval": "epoch",
    #             "frequency": 1
    #         },
    #     },
    #     {
    #         "optimizer": opt_header, 
    #         "lr_scheduler": scheduler_header,
    #         "interval": "epoch",
    #         "frequency": 1
    #      }
    # )


    def test_step(self, batch, batch_idx):
        pass

