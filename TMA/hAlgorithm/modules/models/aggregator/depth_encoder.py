from .base import AuxResnet


class AuxResnetPromptEncoder(AuxResnet):
    def __init__(self, layer_num, use_dims, use_bn, **kwargs):
        super().__init__(layer_num=layer_num, use_dims=use_dims, use_bn=use_bn, **kwargs)

    def forward(self, prompt_depth, **kwargs):
        return super().forward(prompt_depth)
