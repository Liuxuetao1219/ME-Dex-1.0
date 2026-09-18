from __future__ import annotations

import torch


SURFACE_TYPE_NAMES = (
    "padding",
    "thumb_tip",
    "thumb_pad",
    "thumb_base",
    "index_tip",
    "index_pad",
    "index_base",
    "middle_tip",
    "middle_pad",
    "middle_base",
    "ring_tip",
    "ring_pad",
    "ring_base",
    "little_tip",
    "little_pad",
    "little_base",
    "palm",
)


def surface_type_ids(finger_id: torch.Tensor, segment_id: torch.Tensor, *, include_proximal: bool = False) -> torch.Tensor:
    """Map the frozen finger/segment vocabularies to the V6 surface vocabulary."""
    table = finger_id.new_full((7, 5), -1)
    table[0, 0] = 0
    for finger, base in ((1, 1), (2, 4), (3, 7), (4, 10), (6, 13)):
        table[finger, 1] = base
        table[finger, 2] = base + 1
        table[finger, 3] = base + 2
    table[5, 4] = 16
    # Opt-in extension for the new four-pad fingers; legacy IDs stay unchanged.
    if include_proximal:
        for index, finger in enumerate((1, 2, 3, 4, 6)):
            table[finger, 4] = 17 + index
    result = table[finger_id, segment_id]
    if (result < 0).any():
        invalid = torch.stack((finger_id[result < 0], segment_id[result < 0]), dim=-1)
        raise ValueError(f"Invalid V6 finger/segment combination: {invalid[0].tolist()}")
    return result
