import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class PriorExtractionModule(nn.Module) : 

    def __init__(self, num_priors, num_features) :
        super().__init__()
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(num_features * 3, num_features, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(num_features)
        )
        self.conv_logit = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(num_features, num_priors, kernel_size = 1, stride = 1, padding = 0),
        )

    def forward(self, pixel_embeddings, mask_features) :
        B, C, H, W = mask_features.size() 
        pixel_embeddings_resized = [F.interpolate(pixel_embedding, size = (H, W), mode = "bilinear", align_corners = True) 
                                    for pixel_embedding in pixel_embeddings]
        pixel_embeddings = torch.cat(pixel_embeddings_resized, dim = 1)
        fusion = self.conv_fusion(pixel_embeddings)
        logit = self.conv_logit(fusion)
        attn = F.softmax(rearrange(logit, "b c h w -> b c (h w)"), dim = -1)
        prior = torch.bmm(attn, rearrange(mask_features, "b c h w -> b (h w) c"))

        return prior



