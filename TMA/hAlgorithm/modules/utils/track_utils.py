import torch

def sample_query_points_uvt(track_query_points : torch.Tensor, track_vis : torch.Tensor, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    B, T, N, _ = track_query_points.shape
    vis_flat = track_vis.bool().transpose(1, 2).reshape(B * N, T)
    is_all_false = ~(vis_flat.any(dim=1))
    prob = vis_flat.float()
    uniform_prob = torch.ones_like(prob) / T
    prob = torch.where(
        is_all_false.unsqueeze(-1).expand(-1, T),
        uniform_prob,
        prob
    )
    sampled_indices = torch.multinomial(prob, num_samples=1, replacement=True)
    t_indices_flat = sampled_indices.squeeze(-1).long()
    query_frame = t_indices_flat.reshape(B, N)
    index_expanded = query_frame.unsqueeze(1).unsqueeze(-1).expand(B, 1, N, 2)
    gathered = torch.gather(track_query_points, dim=1, index=index_expanded)
    query_points = gathered.squeeze(1) # [B, N, 2]
    query_points = torch.concat([query_points, query_frame.unsqueeze(-1)], dim=-1)
    return query_points