import torch

def gripper_metadata(support):
    """Map RoboTwin's four surfaces into the encoder's anatomical slots."""
    batch = support.shape[0] // 4
    native = torch.tensor([0, 1, 16, 17], device=support.device).repeat(batch)
    hand = torch.tensor([1, 1, 2, 2], device=support.device).repeat(batch)
    finger = torch.tensor([1, 2, 1, 2], device=support.device).repeat(batch)
    segment = torch.full_like(finger, 2)
    grid = support[:, 0].bool()
    valid = grid.flatten(1).any(-1)
    # UV coordinates span the valid surface bounding box.
    rows = torch.arange(10, device=grid.device)[None, :, None].expand_as(grid)
    cols = torch.arange(14, device=grid.device)[None, None, :].expand_as(grid)
    lo_r = rows.masked_fill(~grid, 10).flatten(1).amin(1)
    hi_r = rows.masked_fill(~grid, 0).flatten(1).amax(1)
    lo_c = cols.masked_fill(~grid, 14).flatten(1).amin(1)
    hi_c = cols.masked_fill(~grid, 0).flatten(1).amax(1)
    u = (cols - lo_c[:, None, None]).float() / (hi_c - lo_c).clamp_min(1)[:, None, None]
    v = (rows - lo_r[:, None, None]).float() / (hi_r - lo_r).clamp_min(1)[:, None, None]
    metadata = dict(region_mask=valid, grid_mask=grid,
                    uv_coordinates=torch.stack((u, v), 1) * grid[:, None],
                    hand_side_id=hand * valid, finger_id=finger * valid,
                    segment_id=segment * valid)
    return native, metadata
