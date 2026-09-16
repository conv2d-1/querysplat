# Adapted from prior-depth: https://github.com/SpatialVision/Prior-Depth-Anything/blob/main/prior_depth_anything/depth_completion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_cluster

from hAlgorithm.utils import instantiate_from_config


# Adapted from Marigold, available at https://github.com/prs-eth/Marigold
def depth2disparity(depth, return_mask=False):
    disparity = torch.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = (1.0 / depth[non_negtive_mask]).to(depth.dtype)
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity


def disparity2depth(disparity, **kwargs):
    return depth2disparity(disparity, **kwargs)


def debug_rel_depth(depth):
    import cv2
    import numpy as np

    print(f"min:{depth.min()}, max:{depth.max()}")
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.cpu().numpy().astype(np.uint8)
    depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    cv2.imwrite("./debug/rel_depth_debug.png", depth)


class MDEPromptModel(nn.Module):
    def __init__(
        self,
        mde_config,
        align_mode="both",
        with_uncertainty=False,
        pretrain=None,
        K=5,
        disp_space=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.mde_model = instantiate_from_config(mde_config)
        self.K = K
        assert align_mode in ["knn", "global", "both", "none"]
        self.global_align = align_mode in ["global", "both"]
        self.knn_align = align_mode in ["knn", "both"]
        self.with_uncertainty = with_uncertainty and (align_mode == "both")
        self.disp_space = disp_space
        self.disp_min = 1.0
        self.disp_max_ratio = 100.0

        if self.mde_model is not None and pretrain is not None:
            self.mde_model.load_state_dict(torch.load(pretrain))

    def prepare_image(self, image):
        image = (image + 1) * 0.5
        return image

    @torch.no_grad()
    def preprocess(self, images, sparse_depths):
        """
        1. Unify the format of all the inputs.
        2. Obtain the model-predicted affine-invariant depth map.
        3. Convert the ground-truth depth to disparity.
        """
        images = self.prepare_image(images)
        sparse_depths = sparse_depths[:, -1]
        h, w = sparse_depths.shape[-2:]
        sparse_mask = sparse_depths > 0.01

        # Preprocess pred_disparities.
        pred_disparities = self.mde_model(images)
        pred_disparities = F.interpolate(
            pred_disparities, (h, w), mode="bilinear", align_corners=True
        )
        pred_disparities = pred_disparities.squeeze(1).to(sparse_depths.dtype)

        # Preprocess sparse_depths and prior depths.
        sparse_disparities = depth2disparity(sparse_depths)

        return pred_disparities, sparse_disparities, sparse_mask

    def forward(self, images, sparse_depths, **kwargs):
        """
        Processe input images and sparse depth information to produce completed depth maps.
        We use global alignment and KNN alignment to refine the depth predictions.

        Args:
            images (torch.Tensor): The input images.
            sparse_depths (torch.Tensor): The sparse depth information.
            sparse_masks (torch.Tensor): Indicating which points in the sparse depth are valid.
            cover_masks (torch.Tensor, optional): Indicating areas to be covered by prior depth.
            prior_depths (torch.Tensor, optional): Prior depth information for covering large areas.
            pattern (optional): Pattern for sampling sparse depth points.

        Returns:
            Dict[str, torch.Tensor]: Containing the processed data, including:
                - 'uncertainties': A tensor representing the uncertainty of the depth predictions.
                - 'scaled_preds': A tensor representing the scaled depth predictions.
                - 'global_preds': A tensor representing the globally aligned depth predictions.
        """
        pred_disparities, sparse_disparities, sparse_masks = self.preprocess(images, sparse_depths)

        prompt_output = []

        # ================================== KNN Alignments.
        # The masks denote the areas to be completed. Exclude the sparse points to accelerate.
        if self.knn_align:
            complete_masks = torch.ones_like(sparse_masks).to(torch.bool)
            complete_masks[sparse_masks] = False

            # Scale the pred_disparities with KNN alignment.
            scaled_preds = self.kss_completer(
                sparse_disparities=sparse_disparities,
                pred_disparities=pred_disparities,
                sparse_masks=sparse_masks,
                K=self.K,
                complete_masks=complete_masks,
            )
            if not self.disp_space:
                scaled_depths = disparity2depth(scaled_preds)
                prompt_output.append(scaled_depths)
            else:
                prompt_output.append(scaled_preds)
        else:
            if not self.disp_space:
                prompt_output.append(sparse_depths[:, -1])
            else:
                prompt_output.append(sparse_disparities)

        # ================================== Global Alignment.
        if self.global_align:
            global_preds = self.ss_completer(
                sparse_disparities=sparse_disparities,
                pred_disparities=pred_disparities,
                sparse_masks=sparse_masks,
            )
            if not self.disp_space:
                global_depths = disparity2depth(global_preds)
                prompt_output.append(global_depths)
            else:
                prompt_output.append(global_preds)
        else:
            if not self.disp_space:
                pred_depths = disparity2depth(pred_disparities)
                prompt_output.append(pred_depths)
            else:
                prompt_output.append(pred_disparities)

        # ================================== Uncertainty map.
        if self.with_uncertainty:
            cal_mask = global_preds > 0.0
            masked_scaled, scaled_global = scaled_preds[cal_mask], global_preds[cal_mask]
            uctn = torch.abs(masked_scaled - scaled_global) / scaled_global
            uncertainties = torch.zeros_like(scaled_preds, dtype=sparse_depths.dtype)
            uncertainties[cal_mask] = uctn

            uncertainties = (uncertainties - uncertainties.min()) / (
                uncertainties.max() - uncertainties.min()
            )
            prompt_output.append(uncertainties)

        return torch.stack(prompt_output, dim=1)

    def calc_scale_shift(self, k_sparse_targets, k_pred_targets, currk_dists=None, knn=False):
        k_pred_targets += torch.rand(*k_pred_targets.shape, device=k_sparse_targets.device) * 1e-5
        X = torch.stack(
            [k_pred_targets, torch.ones_like(k_pred_targets, device=k_sparse_targets.device)], dim=2
        )

        # To perform weights to the knn points.
        if knn > 0:
            k_sparse_targets, X = self.perform_weighted(k_sparse_targets, X, currk_dists)
        elif k_pred_targets.shape[0] > 1:
            k_sparse_targets = k_sparse_targets.unsqueeze(-1)

        solution = torch.linalg.lstsq(X.float(), k_sparse_targets.float())
        scale, shift = solution[0][:, 0].squeeze().to(k_sparse_targets.dtype), solution[0][
            :, 1
        ].squeeze().to(k_sparse_targets.dtype)

        return scale, shift

    def perform_weighted(self, sparse_ori, pred_ori, dists):
        """
        Perform weighted operations on input tensors using distance-based weights. A diagonal
        matrix is created from the normalized weights and used to weight the inputs.

        Notes:
            - Weights are calculated as the inverse of the distances.
            - Weights are normalized to ensure they sum to 1.

        Args:
            sparse_ori (torch.Tensor): Sparse original map.
            pred_ori (torch.Tensor): Predicted map.
            dists (torch.Tensor): Distances used for weight calculation.

        Returns:
            Tuple: Containing two tensors:
                - sparse_weighted: The weighted version of the sparse original map.
                - pred_weighted: The weighted version of the predicted map.
        """

        weights = 1 / dists
        wsum = weights.sum(dim=1, keepdim=True)
        weights = weights / wsum
        W = torch.diag_embed(weights)

        pred_weighted = W @ pred_ori
        sparse_weighted = W @ sparse_ori.unsqueeze(-1)
        return sparse_weighted, pred_weighted

    def knn_aligns(self, sparse_disparities, pred_disparities, sparse_masks, complete_masks, K):
        """
        Perform K-Nearest Neighbors (KNN) alignment on sparse and predicted disparities.

        Args:
            sparse_disparities (torch.Tensor): Disparities for sparse map points.
            pred_disparities (torch.Tensor): Predicted disparities for sparse map points.
            sparse_masks (torch.Tensor): Indicating which points in the sparse map are valid.
            complete_masks (torch.Tensor): Indicating which points in the map to be completed.
            K (int): The number of nearest neighbors to find for each map point.

        Returns:
            Tuple: Containing three tensors:
                - dists: The Euclidean distances from each sparse point to its K nearest neighbors.
                - k_sparse_targets: Disparities of the K nearest neighbors from the sparse data.
                - k_pred_targets: Disparities of the K nearest neighbors from the predicted data.
        """
        # Coordinates are processed to ensure compatibility with the KNN function.
        batch_sparse = torch.nonzero(sparse_masks, as_tuple=False)[
            ..., [0, 2, 1]
        ].float()  # [N, 3] (b, x, y)
        batch_complete = torch.nonzero(complete_masks, as_tuple=False)[
            ..., [0, 2, 1]
        ].float()  # [M, 3] (b, x, y)

        batch_x, batch_y = batch_sparse[:, 0].contiguous(), batch_complete[:, 0].contiguous()
        x, y = batch_sparse[:, -2:].contiguous(), batch_complete[:, -2:].contiguous()

        # Use `torch_cluster.knn` to find K nearest neighbors.
        knn_map = torch_cluster.knn(x=x, y=y, k=K, batch_x=batch_x, batch_y=batch_y)  # [2, M * K]
        knn_indices = knn_map[1, :].view(-1, K)

        k_sparse_targets = sparse_disparities[sparse_masks][knn_indices]
        k_pred_targets = pred_disparities[sparse_masks][knn_indices]

        knn_coords = x[knn_indices]
        expanded_complete_points = y.unsqueeze(dim=1).repeat(1, K, 1)
        dists = torch.norm(expanded_complete_points - knn_coords, dim=2)

        return dists, k_sparse_targets, k_pred_targets

    def kss_completer(
        self, sparse_disparities, pred_disparities, complete_masks, sparse_masks, K=5
    ) -> torch.Tensor:
        """
        Perform K-Nearest Neighbors (KNN) interpolation to complete sparse disparities.Use a batch-oriented
        implementation of KNN interpolation to complete the sparse disparities. We leverages "torch_cluster.knn"
        for acceleration and GPU memory efficiency.

        Args:
            sparse_disparities (torch.Tensor): Disparities for sparse map.
            pred_disparities (torch.Tensor): Dredicted disparities for sparse map points.
            complete_masks (torch.Tensor): Indicating which points in the complete map are valid.
            sparse_masks (torch.Tensor): Indicating which points in the sparse map are valid.
            K (int): The number of nearest neighbors to use for interpolation. Defaults to 5.

        Returns:
            The completed disparities, interpolated from the nearest neighbors.
        """

        # Use `knn_aligns` to find the K nearest neighbors and calculate distances.
        bottomk_dists, k_sparse_targets, k_pred_targets = self.knn_aligns(
            sparse_disparities=sparse_disparities,
            pred_disparities=pred_disparities,
            sparse_masks=sparse_masks,
            K=K,
            complete_masks=complete_masks,
        )

        scaled_preds = torch.zeros_like(
            sparse_disparities, device=sparse_disparities.device, dtype=sparse_disparities.dtype
        )
        scale, shift = self.calc_scale_shift(
            k_sparse_targets=k_sparse_targets,
            k_pred_targets=k_pred_targets,
            currk_dists=bottomk_dists,
            knn=True,
        )

        # Apply scaling and shifting to the predicted disparities based on the nearest neighbors.
        scaled_preds[complete_masks] = pred_disparities[complete_masks] * scale + shift
        # The completed disparities are computed by combining the scaled predictions and the original sparse disparities.
        scaled_preds[sparse_masks] = sparse_disparities[sparse_masks]

        # # min max clamp
        # _max, _ = sparse_disparities.flatten(1).max(dim=1)
        # _max = _max * 1.5
        # for i, _scaled_pred in enumerate(scaled_preds):
        #     scaled_preds[i][_scaled_pred > _max[i]] = _max[i]

        # _min = 1e0
        # scaled_preds[scaled_preds < _min] = _min

        scaled_preds = self._min_max_clamp(scaled_preds, sparse_disparities)
        return scaled_preds

    def global_aligns(self, sparse_disparities, pred_disparities, sparse_masks):
        """
        Perform global alignment on sparse and predicted disparities. Extract the valid disparities from
        both sparse and predicted map based on the sparse masks.

        Args:
            sparse_disparities (torch.Tensor): Disparities for sparse map points.
            pred_disparities (torch.Tensor): Predicted disparities for sparse map points.
            sparse_masks (torch.Tensor): Indicating which points in the sparse map are valid.

        Returns:
            Tuple[torch.Tensor]: Containing two tensors:
                - k_sparse_targets: The valid disparities from the sparse map.
                - k_pred_targets: The valid disparities from the predicted map.
        """

        # The valid disparities are extracted and unsqueezed to maintain consistent dimensions.
        k_sparse_targets = sparse_disparities[sparse_masks].unsqueeze(dim=0)
        k_pred_targets = pred_disparities[sparse_masks].unsqueeze(dim=0)

        return k_sparse_targets, k_pred_targets

    def ss_completer(self, sparse_disparities, pred_disparities, sparse_masks) -> torch.Tensor:
        """
        Complete sparse disparities using a simple scaling and shifting approach. Perform a global
        alignment of the sparse and predicted disparities, then applies a scaling and shifting
        transformation to complete the sparse disparities.

        Args:
            sparse_disparities (torch.Tensor): Disparities for sparse map points.
            pred_disparities (torch.Tensor): Predicted disparities for sparse map points.
            sparse_masks (torch.Tensor): Indicating which points in the sparse map are valid.

        Returns:
            The completed disparities, computed by scaling and shifting the predicted disparities.
        """

        # Use `global_aligns` to extract valid disparities.
        k_sparse_targets, k_pred_targets = self.global_aligns(
            sparse_disparities=sparse_disparities,
            pred_disparities=pred_disparities,
            sparse_masks=sparse_masks,
        )

        scale, shift = self.calc_scale_shift(
            k_sparse_targets=k_sparse_targets, k_pred_targets=k_pred_targets
        )

        # Apply scaling and shifting to the predicted disparities based on the nearest neighbors.
        scaled_preds = pred_disparities * scale + shift
        scaled_preds = self._min_max_clamp(scaled_preds, sparse_disparities)
        return scaled_preds

    def _min_max_clamp(self, align_pred, prompt):
        # min max clamp
        _max, _ = prompt.flatten(1).max(dim=1)
        _max = _max * self.disp_max_ratio
        for i, _pred in enumerate(align_pred):
            align_pred[i][_pred > _max[i]] = _max[i]

        _min = self.disp_min
        align_pred[align_pred < _min] = 0

        return align_pred
