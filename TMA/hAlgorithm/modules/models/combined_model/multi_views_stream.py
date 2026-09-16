from collections import defaultdict

import torch

from .multi_views_v4 import MVCombinedModelV2


class MVCombinedModelV4(MVCombinedModelV2):
    """
    Version 4 of the Multi-View Combined Model.

    This version introduces the following changes:
    1. support stream model and use kv cache.
    """

    def __init__(self, use_cache=False, **kwargs):
        super(MVCombinedModelV4, self).__init__(**kwargs)
        self.use_cache = use_cache

    def forward_mv(
        self,
        rgb,
        prompt_depth=None,
        prompt_scale=None,
        meta_data=None,
        intrinsics=None,
        query_points=None,
    ):

        if self.training or (not self.use_cache):
            return super().forward_mv(
                rgb=rgb,
                prompt_depth=prompt_depth,
                prompt_scale=prompt_scale,
                meta_data=meta_data,
                intrinsics=intrinsics,
                query_points=query_points,
            )

        # NOTE: use_cache
        if self.mv_decoder is not None:
            past_key_values = [None] * self.mv_decoder.depth
        if self.camera_head is not None:
            past_key_values_camera = [None] * self.camera_head.trunk_depth

        # Extract frame number and view number from metadata
        frame_num = meta_data["frames"][0].item()
        view_num = meta_data["views"][0].item()

        meta_data["frames"][...] = 1
        meta_data["views"][...] = 1

        # Unpack input dimensions for batch size (b), number of frames (n), channels (c), height (h), width (w)
        b, n, c, h, w = rgb.shape
        assert n == frame_num * view_num, "Number of frames must match frame_num * view_num"

        total_results = defaultdict(list)
        total_features_dict = defaultdict(list)

        for i in range(n):
            single_rgb = rgb[:, i]

            if prompt_depth is not None:
                single_prompt_depth = prompt_depth[:, i]

            if prompt_scale is not None:
                single_prompt_scale = prompt_scale[:, i]

            prompt_features = rgb_features = camera_tokens = None
            local_feature_maps = glb_feature_maps = None
            local_refine_features = glb_refine_features = None
            results = dict()
            features_dict = dict()

            # If a prompt encoder exists and a depth prompt is provided, encode the depth prompt
            if self.prompt_encoder is not None and prompt_depth is not None:
                prompt_features = self.prompt_encoder(single_prompt_depth, meta_data=meta_data)

            # Encode RGB images using the RGB encoder
            if self.rgb_encoder is not None:
                rgb_features = self.rgb_encoder(
                    single_rgb, condition=prompt_features, meta_data=meta_data
                )

            # Process multi-view depth results
            if self.head is not None:
                sf_results = self.head(
                    rgb_features,
                    prompt_features=prompt_features,
                    return_dict=True,
                    meta_data=meta_data,
                    intrinsics=intrinsics,
                )
                results["pointmap"] = sf_results.pop("pointmap")
                results["confidence"] = sf_results.pop("confidence")
                results["pointmap"] = results["pointmap"].view(
                    b, 1, *results["pointmap"].shape[-3:]
                )
                results["confidence"] = results["confidence"].view(
                    b, 1, *results["confidence"].shape[-3:]
                )

            # Further process features using the multi-view decoder
            if self.mv_decoder is not None:
                patch_features, camera_tokens = self.mv_decoder(
                    rgb_features,
                    prompt_features=prompt_features,
                    prompt_scale=single_prompt_scale,
                    meta_data=meta_data,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                patch_features = rgb_features

            # Process multi-view depth results
            if self.mv_depth_head is not None:
                mv_depth_results = self.mv_depth_head(
                    patch_features,
                    prompt_features=prompt_features,
                    return_dict=True,
                    meta_data=meta_data,
                    intrinsics=intrinsics,
                )
                results["mv_depth"] = mv_depth_results.pop("pointmap")
                results["mv_depth_confidence"] = mv_depth_results.pop("confidence")
                results["mv_depth"] = results["mv_depth"].view(
                    b, 1, *results["mv_depth"].shape[-3:]
                )
                results["mv_depth_confidence"] = results["mv_depth_confidence"].view(
                    b, 1, *results["mv_depth_confidence"].shape[-3:]
                )
                local_feature_maps = mv_depth_results.get("features", None)
                local_refine_features = mv_depth_results.get("refine_features", None)

            # Process multi-view point cloud results
            if self.mv_point_head is not None:
                mv_point_results = self.mv_point_head(
                    patch_features,
                    prompt_features=prompt_features,
                    return_dict=True,
                    meta_data=meta_data,
                    intrinsics=intrinsics,
                )
                results["mv_pointmap"] = mv_point_results.pop("pointmap")
                results["mv_confidence"] = mv_point_results.pop("confidence")
                results["mv_pointmap"] = results["mv_pointmap"].view(
                    b, 1, *results["mv_pointmap"].shape[-3:]
                )
                results["mv_confidence"] = results["mv_confidence"].view(
                    b, 1, *results["mv_confidence"].shape[-3:]
                )
                glb_feature_maps = mv_point_results.get("features", None)
                glb_refine_features = mv_point_results.get("refine_features", None)

            # Process camera pose encoding
            if self.camera_head is not None:
                pose_enc = self.camera_head(
                    camera_tokens,
                    meta_data=meta_data,
                    past_key_values_camera=past_key_values_camera,
                    use_cache=True,
                )
                results["pose_enc"] = pose_enc

            # Process tracking results
            if self.track_head is not None and query_points is not None:
                track, track_vis, track_confidence = self.track_head(
                    patch_features,
                    query_points=query_points,
                    prompt_features=prompt_features,
                    meta_data=meta_data,
                    local_feature_maps=local_feature_maps,
                    glb_feature_maps=glb_feature_maps,
                )
                results["track"] = track
                results["track_vis"] = track_vis
                results["track_confidence"] = track_confidence

            if self.disp_head is not None:
                init_disp, seq_disp = self.disp_head(
                    local_feature_maps,
                    glb_feature_maps,
                    prompt_features=prompt_features,
                    meta_data=meta_data,
                )
                results["seq_disp"] = seq_disp
                results["disp"] = init_disp

            for key, val in results.items():
                total_results[key].append(val)

            if prompt_features is not None:
                total_features_dict["prompt_features"].append(prompt_features)
            if rgb_features is not None:
                total_features_dict["rgb_features"].append(rgb_features)
            if patch_features is not None:
                total_features_dict["patch_features"].append(patch_features)

            if local_feature_maps is not None:
                total_features_dict["local_feature_maps"].append(local_feature_maps)
            if glb_feature_maps is not None:
                total_features_dict["glb_feature_maps"].append(glb_feature_maps)

            if local_refine_features is not None:
                total_features_dict["local_refine_features"].append(local_refine_features)
            if glb_refine_features is not None:
                total_features_dict["glb_refine_features"].append(glb_refine_features)

        for key in total_results.keys():
            if key == "pose_enc":
                merge_list = []
                for i in range(len(total_results[key][0])):
                    merge_list.append(torch.cat([data[i] for data in total_results[key]], dim=1))
                total_results[key] = merge_list
                # print(key, [data.shape for data in merge_list])
            else:
                total_results[key] = torch.cat(total_results[key], dim=1)
                # print(key, total_results[key].shape)

        for key in total_features_dict.keys():
            if isinstance(total_features_dict[key][0], (list, tuple)):
                merge_list = []
                for i in range(len(total_features_dict[key][0])):
                    if key == "rgb_features":
                        merge_list.append(
                            torch.cat([data[i] for data in total_features_dict[key]], dim=0)
                        )
                    elif key == "patch_features":
                        merge_list.append(
                            torch.cat([data[i] for data in total_features_dict[key]], dim=1)
                        )
                    else:
                        raise ValueError(f"{key}")
                total_features_dict[key] = merge_list
                # print(key, [data.shape for data in merge_list])
            else:
                if total_features_dict[key][0].ndim == 4:
                    total_features_dict[key] = torch.cat(total_features_dict[key], dim=0)
                else:
                    total_features_dict[key] = torch.cat(total_features_dict[key], dim=1)
                # print(key, total_features_dict[key].shape)

        meta_data["frames"][...] = frame_num
        meta_data["views"][...] = view_num

        # NOTE: accelerate fp16 不支持 defaultdict
        total_results = dict(**total_results)
        total_features_dict = dict(**total_features_dict)

        return total_features_dict, total_results
