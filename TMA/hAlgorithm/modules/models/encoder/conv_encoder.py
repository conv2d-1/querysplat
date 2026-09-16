import torch
import torch.nn as nn


class PixelLayerNorm(nn.Module):
    def __init__(self, channel_dims):
        super().__init__()
        self.channel_dims = channel_dims
        self.layer_norm = nn.LayerNorm([self.channel_dims])

    def forward(self, x):
        # channel already in last dim
        if x.shape[-1] == self.channel_dims:
            return self.layer_norm(x)
        elif x.shape[1] == self.channel_dims:
            x = x.movedim(1, -1)
            x = self.layer_norm(x)
            return x.movedim(-1, 1)
        else:
            raise NotImplementedError("PixelLayerNorm: tensor channel must in dim 1 or dim -1.")


class ConvModule(nn.Module):
    def __init__(
        self,
        layer_num,
        use_dims,
        output_channel=None,
        kernel_size=3,
        input_channel=None,
        with_layernorm=False,
    ):
        super().__init__()

        self.layer_num = layer_num
        self.with_layernorm = with_layernorm
        self.output_channel = (
            output_channel if isinstance(output_channel, (list, tuple)) else [output_channel]
        )
        self.kernel_size = kernel_size
        self.use_dims = use_dims
        self.input_channel = len(self.use_dims) if self.use_dims is not None else input_channel

        layers = []
        if self.output_channel is None:
            for idx in range(self.layer_num):
                layers.append(
                    nn.Conv2d(
                        self.input_channel,
                        self.input_channel,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                        bias=True,
                        groups=1,
                    )
                )
            if self.with_layernorm:
                layers.append(PixelLayerNorm(self.input_channel))
            layers.append(nn.ReLU())
        else:
            input_channel = self.input_channel
            for output_channel in self.output_channel:
                for idx in range(self.layer_num):
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
                    input_channel = output_channel
                if self.with_layernorm:
                    layers.append(PixelLayerNorm(output_channel))
                layers.append(nn.ReLU())
        assert len(layers) > 0
        self.convs = nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        if self.use_dims is not None:
            return self.convs(x[:, self.use_dims])
        else:
            return self.convs(x)
