import torch
import torch.nn as nn
from .model.v2 import MoGeModel

class Moge(nn.Module):
    def __init__(self, pretrain, renormalize_image=False, depthmap_output=False, load_weights=True, ignore_key=None, **kwargs) -> None:
        super().__init__()
        self.model = MoGeModel.from_pretrained(
            pretrain, 
            load_checkpoint=load_weights,
            ignore_key=ignore_key,
            model_kwargs=kwargs
        )
        self.depthmap_output = depthmap_output
        self.renormalize_image = renormalize_image

    def forward(
        self, 
        image: torch.Tensor, 
        num_tokens: int = None,
        resolution_level: int = 9,
        return_dict=True,
    ):
        if num_tokens is None:
            min_tokens, max_tokens = self.model.num_tokens_range
            num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))
        
        if self.renormalize_image:
            image = image * 0.5 + 0.5

        output = self.model(image, num_tokens, return_features=True)
        pointmap = output.pop('points', None)
        if self.depthmap_output:
            pointmap = pointmap[..., [-1]]
        output['pointmap'] = pointmap.permute(0, 3, 1, 2)
        
        if return_dict:
            return output
        return output['pointmap']
    
    def infer(self, image, return_dict=False):
        with torch.no_grad():
            output = self.model.infer(image, use_fp16=True)
        
        if return_dict:
            return output
        return output['depth']

