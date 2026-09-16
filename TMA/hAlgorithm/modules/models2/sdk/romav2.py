from hAlgorithm.modules.models2.head.romav2_match_head import RoMaV2MatchHead
from hAlgorithm.modules.models2.external.romav2.models.romav2.features import Descriptor
import torch
import torch.nn as nn
from dataclasses import is_dataclass

class RomaV2(nn.Module):
    def __init__(self, rgb_config=None, matcher_config=None):
        super().__init__()
        
        rgb_cfg = Descriptor.Cfg()
        # update cfg
        def update_nested(obj, updates):
            for k, v in updates.items():
                if not hasattr(obj, k):
                    pass
                current = getattr(obj, k)
                if is_dataclass(current) and isinstance(v, dict):
                    # 递归更新嵌套 dataclass
                    update_nested(current, v)
                else:
                    # 直接赋值
                    setattr(obj, k, v)
        if rgb_config:
            update_nested(rgb_cfg, rgb_config)

        self.rgb_encoder = Descriptor(rgb_cfg)
        matcher_config = matcher_config if matcher_config is not None else {}
        self.matcher = RoMaV2MatchHead(**matcher_config)
        self.freeze_modules = ["rgb_encoder"]
        self.freeze()
    
    def freeze(self):
        """Freeze specified modules."""
        if self.freeze_modules is None:
            return
        for module_name in self.freeze_modules:
            if hasattr(self, module_name):
                module = getattr(self, module_name)
                if module is not None:
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False

    def forward(self, image: torch.Tensor, pairs=None, **kwargs):
        assert image.ndim == 5, "RomaV2 Matcher Requires at least two image"
        b, n, c, h, w = image.shape
        
        # batched view dinov3
        image = image.reshape(b*n, c, h, w)
        rgb_feat = self.rgb_encoder(image)
        rgb_feat = [feat.reshape(b, n, *feat.shape[1:]) for feat in rgb_feat]
        image = image.reshape(b, n, c, h, w)
        
        matches = self.matcher(image, rgb_feat, bidirectional=False)
        out = {
            "match": matches
        }
        return out
    
