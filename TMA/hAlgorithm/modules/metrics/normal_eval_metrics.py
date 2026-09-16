import torch


class NormalEvalMetrics:
    def __init__(self, metrics, valid_mask_name):
        self.metrics = metrics
        self.valid_mask_name = valid_mask_name

    def eval_single_data(self, inputs, output, eval_idx):
        if output.pointmap_gt is None and output.normal_gt is None:
            return dict()
        if output.normal is None:
            return dict()

        if eval_idx is not None:
            valid_mask = inputs[self.valid_mask_name][:, eval_idx, ...].squeeze().clone()
        else:
            valid_mask = inputs[self.valid_mask_name].squeeze().clone()

        if output.normal_gt is None:
            target = torch.from_numpy(output.pointmap_gt).reshape([output.pointmap_h, output.pointmap_w, 3])
            target, targets_normals_masks = pointmap2normal(target, valid_mask)
            # [1, 3, h, w] -> [h, w, 3]
            target = target.squeeze(0).permute(1, 2, 0).contiguous()
            valid_mask = valid_mask & targets_normals_masks.squeeze(0)
        else:
            target = output.normal_gt
            target = torch.from_numpy(target)

        output = torch.from_numpy(output.normal)

        results_dict = dict()
        for metric in self.metrics:
            results = eval(metric)(output, target, valid_mask)
            results_dict[metric] = results
        return results_dict

    def eval_mf_data(self, inputs, output):
        results_dict = dict()
        valid_result = 0
        for i, out in enumerate(output):
            if out is None:
                continue
            result = self.eval_single_data(inputs, out, eval_idx=i)
            for k, v in result.items():
                results_dict[k] = results_dict.get(k, 0) + v
            valid_result += 1
        if valid_result == 0:
            raise ValueError("Valid Result is zero!")
        results_dict = {k: v / valid_result for k, v in results_dict.items()}
        return results_dict

    def __call__(self, inputs, output):
        if isinstance(output, list):
            return self.eval_mf_data(inputs, output)
        else:
            return self.eval_single_data(inputs, output, eval_idx=None)


def normal_cos(prediction, target, mask):
    masks_normals = mask
    h, w, c = target.size()

    prediction = prediction.view(-1, c)
    target = target.view(-1, c)
    masks_normals = masks_normals.view(-1)

    # angle between target and pred normal
    cos_angle = torch.einsum("nc,nc->n", target[masks_normals], prediction[masks_normals])
    if len(cos_angle) == 0:
        return 0 * prediction.sum()

    loss = (1 - cos_angle**2).mean()
    return loss


def get_surface_normalv2(xyz, mask_valid, patch_size=5):
    """
    xyz: xyz coordinates, in [b, h, w, c]
    patch: [p1, p2, p3,
            p4, p5, p6,
            p7, p8, p9]
    surface_normal = [(p9-p1) x (p3-p7)] + [(p6-p4) - (p8-p2)]
    return: normal [h, w, 3, b]
    """
    if xyz.ndim == 3:
        xyz = xyz[None]
        mask_valid = mask_valid[None]

    b, h, w, c = xyz.shape
    half_patch = patch_size // 2

    mask_pad = torch.zeros(
        (b, h + patch_size - 1, w + patch_size - 1), device=mask_valid.device
    ).bool()
    mask_pad[:, half_patch:-half_patch, half_patch:-half_patch] = mask_valid

    xyz_pad = torch.zeros(
        (b, h + patch_size - 1, w + patch_size - 1, c), dtype=xyz.dtype, device=xyz.device
    )
    xyz_pad[:, half_patch:-half_patch, half_patch:-half_patch, :] = xyz

    xyz_left = xyz_pad[:, half_patch : half_patch + h, :w, :]  # p4
    xyz_right = xyz_pad[:, half_patch : half_patch + h, -w:, :]  # p6
    xyz_top = xyz_pad[:, :h, half_patch : half_patch + w, :]  # p2
    xyz_bottom = xyz_pad[:, -h:, half_patch : half_patch + w, :]  # p8
    xyz_horizon = xyz_left - xyz_right  # p4p6
    xyz_vertical = xyz_top - xyz_bottom  # p2p8

    xyz_left_in = xyz_pad[:, half_patch : half_patch + h, 1 : w + 1, :]  # p4
    xyz_right_in = xyz_pad[
        :, half_patch : half_patch + h, patch_size - 1 : patch_size - 1 + w, :
    ]  # p6
    xyz_top_in = xyz_pad[:, 1 : h + 1, half_patch : half_patch + w, :]  # p2
    xyz_bottom_in = xyz_pad[
        :, patch_size - 1 : patch_size - 1 + h, half_patch : half_patch + w, :
    ]  # p8
    xyz_horizon_in = xyz_left_in - xyz_right_in  # p4p6
    xyz_vertical_in = xyz_top_in - xyz_bottom_in  # p2p8

    n_img_1 = torch.cross(xyz_horizon_in, xyz_vertical_in, dim=3)
    n_img_2 = torch.cross(xyz_horizon, xyz_vertical, dim=3)

    # re-orient normals consistently
    orient_mask = torch.sum(n_img_1 * xyz, dim=3) > 0
    n_img_1[orient_mask] *= -1
    orient_mask = torch.sum(n_img_2 * xyz, dim=3) > 0
    n_img_2[orient_mask] *= -1

    n_img1_L2 = torch.sqrt(torch.sum(n_img_1**2, dim=3, keepdim=True) + 1e-4)
    n_img1_norm = n_img_1 / (n_img1_L2 + 1e-8)

    n_img2_L2 = torch.sqrt(torch.sum(n_img_2**2, dim=3, keepdim=True) + 1e-4)
    n_img2_norm = n_img_2 / (n_img2_L2 + 1e-8)

    # average 2 norms
    n_img_aver = n_img1_norm + n_img2_norm
    n_img_aver_L2 = torch.sqrt(torch.sum(n_img_aver**2, dim=3, keepdim=True) + 1e-4)
    n_img_aver_norm = n_img_aver / (n_img_aver_L2 + 1e-8)
    # re-orient normals consistently
    orient_mask = torch.sum(n_img_aver_norm * xyz, dim=3) > 0
    n_img_aver_norm[orient_mask] *= -1
    # n_img_aver_norm_out = n_img_aver_norm.permute((1, 2, 3, 0))  # [h, w, c, b]

    # get mask for normals
    mask_p4p6 = (
        mask_pad[:, half_patch : half_patch + h, :w] & mask_pad[:, half_patch : half_patch + h, -w:]
    )
    mask_p2p8 = (
        mask_pad[:, :h, half_patch : half_patch + w] & mask_pad[:, -h:, half_patch : half_patch + w]
    )
    mask_normal = mask_p2p8 & mask_p4p6
    n_img_aver_norm[~mask_normal] = 0

    # a = torch.sum(n_img1_norm_out*n_img2_norm_out, dim=2).cpu().numpy().squeeze()
    # plt.imshow(np.abs(a), cmap='rainbow')
    # plt.show()
    n_img_aver_norm[mask_normal] = n_img_aver_norm[mask_normal] / (
        torch.norm(n_img_aver_norm[mask_normal], dim=-1, keepdim=True) + 1e-9
    )
    return n_img_aver_norm.permute(0, 3, 1, 2).contiguous(), mask_normal  # [b, h, w, 3], [b, h, w]


def pointmap2normal(pointmap, masks):
    """
    Args:
        pointmap (B,H,W,3): point map
    Returns:
        normal (B,H,W,3): normalized surface normal
        normal_masks (B,H,W,1): valid mask for surface normal
    """
    normals, normal_masks = get_surface_normalv2(pointmap, mask_valid=masks.squeeze())
    normal_masks = normal_masks & masks
    return normals, normal_masks


def normal_thresh_percentage(prediction, target, mask, thresh):
    masks_normals = mask
    h, w, c = target.size()

    prediction = prediction.view(-1, c)
    target = target.view(-1, c)
    masks_normals = masks_normals.view(-1)

    # angle between target and pred normal
    cos_angle = torch.einsum("nc,nc->n", target[masks_normals], prediction[masks_normals])

    cos_angle = torch.clamp(cos_angle, min=-1.0, max=1.0)
    angle_error = torch.acos(cos_angle) * 180.0 / torch.pi
    angle_error[angle_error>90] = 180 - angle_error[angle_error>90]
    valid_pics = torch.sum(mask, dtype=torch.float32) + 1e-6
    
    error = 100.0 * (torch.sum(angle_error < thresh) / valid_pics)
    return error


def normal_angle(prediction, target, mask):
    masks_normals = mask
    h, w, c = target.size()

    prediction = prediction.view(-1, c)
    target = target.view(-1, c)
    masks_normals = masks_normals.view(-1)

    # angle between target and pred normal
    cos_angle = torch.einsum("nc,nc->n", target[masks_normals], prediction[masks_normals])

    cos_angle = torch.clamp(cos_angle, min=-1.0, max=1.0)
    angle_error = torch.acos(cos_angle) * 180.0 / torch.pi
    angle_error[angle_error > 90] = 180 - angle_error[angle_error > 90]
    
    valid_pics = torch.sum(mask, dtype=torch.float32) + 1e-6
    error = angle_error.sum() / valid_pics
    return error


def normal_delta_5(prediction, target, mask):
    return normal_thresh_percentage(prediction, target, mask, 5)


def normal_delta_30(prediction, target, mask):
    return normal_thresh_percentage(prediction, target, mask, 30)
