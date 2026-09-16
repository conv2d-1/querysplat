import torch.nn as nn

from hAlgorithm.utils import instantiate_from_config


class CombinedModel(nn.Module):
    def __init__(
        self, rgb_encoder=None, prompt_encoder=None, decoder=None, head=None, freeze_modules=[]
    ):
        super().__init__()
        self.rgb_encoder_cfg = rgb_encoder
        self.prompt_encoder_cfg = prompt_encoder
        self.decoder_cfg = decoder
        self.head_cfg = head

        self.build_modules()

        self.freeze_modules = freeze_modules
        self.fp8_enabled = False

    def fp8_enabled(self):
        import transformer_engine.common.recipe as te_recipe
        from transformer_engine.common.recipe import DelayedScaling

        # import transformer_engine.pytorch as te

        self.fp8_enabled = True
        FP8_RECIPE_KWARGS = {
            "fp8_format": te_recipe.Format.HYBRID,
            "amax_history_len": 32,
            "amax_compute_algo": "max",
        }
        self.fp8_recipe = DelayedScaling(**FP8_RECIPE_KWARGS)

    def build_modules(self):
        # Initialize a list to store the names of the modules that are successfully built.
        self.module_names = []

        # Instantiate the RGB encoder module using the provided configuration.
        # If the instantiation is successful, add its name to the module_names list.
        self.rgb_encoder = instantiate_from_config(self.rgb_encoder_cfg)
        if self.rgb_encoder is not None:
            self.module_names.append("rgb_encoder")

        # Instantiate the prompt encoder module using the provided configuration.
        # If the instantiation is successful, add its name to the module_names list.
        self.prompt_encoder = instantiate_from_config(self.prompt_encoder_cfg)
        if self.prompt_encoder is not None:
            self.module_names.append("prompt_encoder")

        # Configure and instantiate the decoder module if its configuration is provided.
        # if self.decoder_cfg is not None:
        #     # Set default values for missing or undefined parameters in the decoder configuration.
        #     if "in_channels" not in self.decoder_cfg or self.decoder_cfg["in_channels"] is None:
        #         self.decoder_cfg["in_channels"] = self.rgb_encoder.get_out_channels()
        #     if "features" not in self.decoder_cfg or self.decoder_cfg["features"] is None:
        #         self.decoder_cfg["features"] = self.rgb_encoder.model_configs["features"]
        #     if (
        #         "prompt_inchannel" not in self.decoder_cfg
        #         or self.decoder_cfg["prompt_inchannel"] is None
        #     ):
        #         self.decoder_cfg["prompt_inchannel"] = (
        #             self.prompt_encoder.output_channel[-1]
        #             if isinstance(self.prompt_encoder.output_channel, (list, tuple))
        #             else self.prompt_encoder.output_channel
        #         )

        # Instantiate the decoder module using the updated configuration.
        # If the instantiation is successful, add its name to the module_names list.
        self.decoder = instantiate_from_config(self.decoder_cfg)
        if self.decoder is not None:
            self.module_names.append("decoder")

        # Configure and instantiate the head module if its configuration is provided.
        # if self.head_cfg is not None:
        #     # Set default values for missing or undefined parameters in the head configuration.
        #     if "features" not in self.head_cfg or self.head_cfg["features"] is None:
        #         self.head_cfg["features"] = self.rgb_encoder.model_configs["features"]

        # Instantiate the head module using the updated configuration.
        # If the instantiation is successful, add its name to the module_names list.
        self.head = instantiate_from_config(self.head_cfg)
        if self.head is not None:
            self.module_names.append("head")

    def freeze(self):
        for module_name in self.freeze_modules:
            if module_name in self.module_names:
                module = getattr(self, module_name)
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def forward(self, x, prompt_depth, with_freeze=False, meta_data=None, **kwargs):
        """
        Processes a single frame of input data.

        Parameters:
        - x (Tensor): The input image tensor.
        - prompt_depth (Tensor or None): Optional depth information to guide the prediction.

        Returns:
        - results (Dict): A dictionary containing the output predictions.
        """
        if with_freeze:
            self.freeze()

        h, w = x.shape[-2:]
        patch_h, patch_w = h // self.rgb_encoder.patch_size, w // self.rgb_encoder.patch_size

        if not self.fp8_enabled:
            rgb_features = self.rgb_encoder(x, meta_data=meta_data)
            if self.prompt_encoder is not None and prompt_depth is not None:
                prompt_features = self.prompt_encoder(prompt_depth, meta_data=meta_data)
            else:
                prompt_features = None
            if self.decoder is not None:
                features = self.decoder(rgb_features, prompt_features, meta_data=meta_data)
                results = self.head(
                    features, patch_h, patch_w, return_dict=True, meta_data=meta_data
                )
            else:
                results = self.head(
                    rgb_features,
                    prompt_features=prompt_features,
                    return_dict=True,
                    meta_data=meta_data,
                )
        return results
