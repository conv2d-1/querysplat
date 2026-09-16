from functools import partial
from typing import Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from hAlgorithm.modules.models.encoder.aux_encoder_fixact import ResidualConvUnitV2


class MLP(nn.Module):
    """
    Adapted from Uniception class GlobalRepresentationEncoder
    """

    def __init__(
        self,
        in_chans: int = 3,
        enc_embed_dim: int = 1024,
        intermediate_dims: List[int] = [128, 256, 512],
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Union[Type[nn.Module], Callable[..., nn.Module]] = partial(
            nn.LayerNorm, eps=1e-6
        ),
        pretrained_checkpoint_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Global Representation Encoder for projecting a global representation to a desired latent dimension.

        Args:
            name (str): Name of the Encoder.
            in_chans (int): Number of input channels.
            enc_embed_dim (int): Embedding dimension of the encoder.
            intermediate_dims (List[int]): List of intermediate dimensions of the encoder.
            act_layer (Type[nn.Module]): Activation layer to use in the encoder.
            norm_layer (Union[Type[nn.Module], Callable[..., nn.Module]]): Final normalization layer to use in the encoder.
            pretrained_checkpoint_path (Optional[str]): Path to pretrained checkpoint. (default: None)
        """
        super().__init__(*args, **kwargs)

        # Initialize the attributes
        self.in_chans = in_chans
        self.enc_embed_dim = enc_embed_dim
        self.intermediate_dims = intermediate_dims
        self.pretrained_checkpoint_path = pretrained_checkpoint_path

        # Init the activation layer
        self.act_layer = act_layer()

        # Initialize the encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.in_chans, self.intermediate_dims[0]),
            self.act_layer,
        )
        for intermediate_idx in range(1, len(self.intermediate_dims)):
            self.encoder = nn.Sequential(
                self.encoder,
                nn.Linear(
                    self.intermediate_dims[intermediate_idx - 1],
                    self.intermediate_dims[intermediate_idx],
                ),
                self.act_layer,
            )
        self.encoder = nn.Sequential(
            self.encoder,
            nn.Linear(self.intermediate_dims[-1], self.enc_embed_dim),
        )

        # Init weights of the final norm layer
        self.norm_layer = norm_layer(enc_embed_dim) if norm_layer else nn.Identity()
        if isinstance(self.norm_layer, nn.LayerNorm):
            nn.init.constant_(self.norm_layer.bias, 0)
            nn.init.constant_(self.norm_layer.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Global Representation Encoder Forward Pass

        Args:
            encoder_input (EncoderGlobalRepInput): Input data for the encoder.
                The provided data must contain a tensor of size (B, C).

        Returns:
            EncoderGlobalRepOutput: Output features from the encoder.
        """
        # Get the input data and verify the shape of the input
        input_data = x
        assert input_data.ndim == 2, "Input data must have shape (B, C)"
        assert (
            input_data.shape[1] == self.in_chans
        ), f"Input data must have {self.in_chans} channels"

        # Encode the global representation
        features = self.encoder(input_data)

        # Normalize the output
        features = self.norm_layer(features)

        return features


class AuxResnet(nn.Module):
    def __init__(
        self, layer_num, use_dims, use_bn, output_channel=None, kernel_size=3, input_channel=None
    ):
        super().__init__()

        self.layer_num = layer_num
        self.output_channel = output_channel
        self.use_bn = use_bn
        self.kernel_size = kernel_size
        self.use_dims = use_dims
        self.input_channel = len(self.use_dims) if self.use_dims is not None else input_channel

        layers = []
        if self.output_channel is None:
            for idx in range(self.layer_num):
                layers.append(
                    ResidualConvUnitV2(
                        features=self.input_channel, activation=nn.ReLU(), bn=self.use_bn
                    )
                )
        else:
            if isinstance(self.output_channel, int):
                layers.append(
                    nn.Conv2d(
                        self.input_channel,
                        self.output_channel,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                        bias=True,
                        groups=1,
                    )
                )
                layers.append(nn.ReLU())

                for idx in range(self.layer_num):
                    layers.append(
                        ResidualConvUnitV2(
                            features=self.output_channel, activation=nn.ReLU(), bn=self.use_bn
                        )
                    )
            else:
                input_channel = self.input_channel
                for output_channel in self.output_channel:
                    layers.append(
                        nn.Conv2d(
                            input_channel,
                            output_channel,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=kernel_size // 2,
                            bias=True,
                            groups=1,
                        )
                    )
                    layers.append(nn.ReLU())

                    for idx in range(self.layer_num):
                        layers.append(
                            ResidualConvUnitV2(
                                features=output_channel, activation=nn.ReLU(), bn=self.use_bn
                            )
                        )
                    input_channel = output_channel

        assert len(layers) > 0
        self.convs = nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        if self.use_dims is not None:
            return self.convs(x[:, self.use_dims])
        else:
            return self.convs(x)
