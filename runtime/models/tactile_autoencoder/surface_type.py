import torch


def surface_type_ids(finger_id: torch.Tensor, segment_id: torch.Tensor) -> torch.Tensor:
    """Map finger and segment identities to surface embedding indices."""
    table = finger_id.new_full((7, 5), -1)
    table[0, 0] = 0
    for finger, base in ((1, 1), (2, 4), (3, 7), (4, 10), (6, 13)):
        table[finger, 1] = base
        table[finger, 2] = base + 1
        table[finger, 3] = base + 2
    table[5, 4] = 16
    for index, finger in enumerate((1, 2, 3, 4, 6)):
        table[finger, 4] = 17 + index
    return table[finger_id, segment_id]
