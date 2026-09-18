from __future__ import annotations

import torch


ANATOMY_TOKEN_NAMES = (
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_little",
    "left_palm",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_little",
    "right_palm",
)

# Anatomical identities:
# hand: padding=0, left=1, right=2
# finger: padding=0, thumb=1, index=2, middle=3, ring=4, palm=5, little=6
_FINGER_TO_LOCAL_TOKEN = (0, 0, 1, 2, 3, 5, 4)


def anatomy_token_ids(
    hand_side_id: torch.Tensor,
    finger_id: torch.Tensor,
    region_mask: torch.Tensor,
) -> torch.Tensor:
    """Map every physical region to one of 12 fixed anatomy token identities.

    Invalid/padded regions return -1.
    """

    lookup = finger_id.new_tensor(_FINGER_TO_LOCAL_TOKEN)
    local = lookup[finger_id]
    token = (hand_side_id - 1) * 6 + local
    valid_identity = region_mask & hand_side_id.ne(0) & finger_id.ne(0)
    return torch.where(valid_identity, token, token.new_full((), -1))


def anatomy_token_presence(region_token_id: torch.Tensor) -> torch.Tensor:
    token_ids = torch.arange(12, device=region_token_id.device)
    return (region_token_id[..., None] == token_ids).any(dim=1)
