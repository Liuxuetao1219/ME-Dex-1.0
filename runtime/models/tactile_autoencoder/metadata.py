import torch

def packed_metadata(support, slots):
    """Translate loader transport slots into explicit V3 physical identities.

    Four declared slots describe ONE RoboTwin gripper surface, not four regions.
    One declared slot describes ONE DexJoCo pad. Loader's 34-slot numbering is
    only a transport convention and is never used as the V3 latent layout.
    """
    first = slots[:, 0]
    local = first.remainder(17)
    gripper = slots[:, 1].ge(0)
    if not ((first >= 0) & (first < 34)).all():
        raise ValueError('unknown surface slot')
    if not ((~gripper & local.ne(3)) | (gripper & ((local == 0) | (local == 4)))).all():
        raise ValueError('invalid gripper or native pad mapping')
    hand = first.div(17, rounding_mode='floor') + 1
    finger = torch.where(local == 16, 5, local.div(4, rounding_mode='floor') + 1)
    segment = 4 - local.remainder(4)
    segment = torch.where(local < 3, 3 - local, segment)
    segment = torch.where(local == 16, 4, segment)
    segment = torch.where(gripper, 2, segment)
    # Dense input region IDs, independent of the final 12 anatomy IDs.
    native = torch.where(local == 16, 0, torch.where(local < 3, local + 1, local))
    native = torch.where(gripper, local.div(4, rounding_mode='floor'), native)
    native = (hand - 1) * 16 + native
    grid = support[:, 0].bool()
    valid = grid.flatten(1).any(-1)
    # Same UV convention as V3 native_grid_metadata: [0,1] on valid bounding box.
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
