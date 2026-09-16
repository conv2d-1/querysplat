from .base import CombinedModel as Base

class PromptPi3Model(Base):
    def __init__(self, encoder_use_prompt=False, decoder_use_prompt=True,**kwargs):
        super().__init__(**kwargs)
        self.encoder_use_prompt = encoder_use_prompt
        self.decoder_use_prompt = decoder_use_prompt

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
        meta_data["image_h"] = h
        meta_data["image_w"] = w
        if not self.fp8_enabled:
            if self.prompt_encoder is not None:
                prompt_features = self.prompt_encoder(prompt_depth, meta_data=meta_data)
            else:
                prompt_features = None

            if self.encoder_use_prompt:
                rgb_features = self.rgb_encoder(x, meta_data=meta_data, condition=prompt_features)
            else:
                rgb_features = self.rgb_encoder(x, meta_data=meta_data)

            if not self.decoder_use_prompt:
                prompt_features = None

            if self.decoder is not None:
                features = self.decoder(rgb_features, prompt_features=prompt_features, meta_data=meta_data)
                results = self.head(
                    features, (h, w), return_dict=True, meta_data=meta_data
                )
            else:
                results = self.head(
                    rgb_features,
                    prompt_features=prompt_features,
                    return_dict=True,
                    meta_data=meta_data,
                )
        return results