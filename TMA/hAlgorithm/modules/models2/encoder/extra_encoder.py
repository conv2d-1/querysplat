import torch
import torch.nn as nn

from hAlgorithm.utils import instantiate_from_config
from hAlgorithm.modules.models.vggt.utils.rotation import mat_to_quat

class ExtraEncoder(nn.Module):
    def __init__(self, scale_encoder=None, rot_encoder=None, trans_encoder=None, with_log_scale=False, **kwargs):
        super(ExtraEncoder, self).__init__()
        self.scale_encoder = instantiate_from_config(scale_encoder)
        self.rot_encoder = instantiate_from_config(rot_encoder)
        self.trans_encoder = instantiate_from_config(trans_encoder)

        self.with_log_scale = with_log_scale
        if self.with_log_scale:
            self.trans_scale_encoder = instantiate_from_config(trans_encoder)

    def forward(self, w2c=None, scale=None, meta_data=None, **kwargs):
        features = []

        if self.scale_encoder is not None and scale is not None:
            scale_features = self.scale_encoder(scale)
            features.append(scale_features)
        
        if w2c is not None and (self.rot_encoder is not None or self.trans_encoder is not None):
            if self.rot_encoder is not None:
                quat = mat_to_quat(w2c[:, :3, :3])
                rot_features = self.rot_encoder(quat)
                features.append(rot_features)

            if self.trans_encoder is not None:
                trans_features = self.trans_encoder(w2c[:, :3, 3])
                features.append(trans_features)

                if self.with_log_scale:
                    trans_scale_features = self.trans_scale_encoder(torch.log(w2c[:, :3, 3] + 1e-8))
                    features.append(trans_scale_features)

        assert len(features) > 0, "No extra encoder is used"

        return features


