"""
Encoder Factory for UniCeption
"""

import os

from hAlgorithm.modules.models2.external.uniception.models.encoders.base import (
    EncoderGlobalRepInput,
    EncoderInput,
    UniCeptionEncoderBase,
    UniCeptionViTEncoderBase,
    ViTEncoderInput,
    ViTEncoderNonImageInput,
    ViTEncoderOutput,
)

from hAlgorithm.modules.models2.external.uniception.models.encoders.base import (
    UniCeptionEncoderBase,
)
from hAlgorithm.modules.models2.external.uniception.models.encoders.dinov2 import DINOv2Encoder, DINOv2IntermediateFeatureReturner
from hAlgorithm.modules.models2.external.uniception.models.encoders.dense_rep_encoder import DenseRepresentationEncoder
from hAlgorithm.modules.models2.external.uniception.models.encoders.global_rep_encoder import GlobalRepresentationEncoder

# Define encoder configurations
ENCODER_CONFIGS = {
    "dinov2": {
        "class": DINOv2Encoder,
        "intermediate_feature_returner_class": DINOv2IntermediateFeatureReturner,
        "supported_models": ["DINOv2", "DINOv2-Registers", "DINOv2-Depth-Anythingv2"],
    },
    "dense_rep_encoder": {
        "class": DenseRepresentationEncoder,
        "supported_models": ["Dense-Representation-Encoder"],
    },
    "global_rep_encoder": {
        "class": GlobalRepresentationEncoder,
        "supported_models": ["Global-Representation-Encoder"],
    },
    # Add other encoders here
}


def encoder_factory(encoder_str: str, **kwargs) -> UniCeptionEncoderBase:
    """
    Encoder factory for UniCeption.
    Please use python3 -m uniception.models.encoders.list to see available encoders.

    Args:
        encoder_str (str): Name of the encoder to create.
        **kwargs: Additional keyword arguments to pass to the encoder constructor.

    Returns:
        UniCeptionEncoderBase: An instance of the specified encoder.
    """
    if encoder_str not in ENCODER_CONFIGS:
        raise ValueError(
            f"Unknown encoder: {encoder_str}. For valid encoder_str options, please use python3 -m uniception.models.encoders.list"
        )

    encoder_config = ENCODER_CONFIGS[encoder_str]
    encoder_class = encoder_config["class"]

    return encoder_class(**kwargs)

