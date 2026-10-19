"""Augmentation for the joint model that keeps every target consistent with the image.

Random affine (rotation, scale, translation) is applied with ONE sampled transform to the image, the 3-channel mask,
the boundary heat-maps and the signed-distance maps; masks are re-binarised afterwards (FAZ/vessel disjoint, background
recomputed). Intensity jitter and noise change the image only. No patch erasing: erasing part of the image while keeping
the full ground truth would teach the network to hallucinate vessels and would break the connectivity it is trained for.
"""
import math
import torch
import torch.nn.functional as F


def joint_augment(x, masks, heat, dist, p_affine=0.7, degrees=15.0, scale=(0.9, 1.1), translate=0.05, contrast=0.2, brightness=0.2, noise_std=0.05):
    """x (B,1,H,W) standardised image; masks (B,3,H,W) [FAZ, vessel, background]; heat, dist (B,2,H,W)."""
    B, dev = x.shape[0], x.device
    cat = torch.cat([x, masks, heat, dist], dim=1)
    ang = (torch.rand(B, device=dev) * 2 - 1) * degrees * math.pi / 180
    sc = torch.empty(B, device=dev).uniform_(*scale)
    tx, ty = [(torch.rand(B, device=dev) * 2 - 1) * translate * 2 for _ in range(2)]
    on = (torch.rand(B, device=dev) < p_affine).float()
    ang, tx, ty, sc = ang * on, tx * on, ty * on, sc * on + (1 - on)
    theta = torch.stack([torch.stack([torch.cos(ang) / sc, -torch.sin(ang) / sc, tx], 1), torch.stack([torch.sin(ang) / sc, torch.cos(ang) / sc, ty], 1)], 1)
    grid = F.affine_grid(theta, list(cat.shape), align_corners=False)
    out = F.grid_sample(cat, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    x2 = out[:, 0:1]
    faz = (out[:, 1:2] > 0.5).float()
    ves = (out[:, 2:3] > 0.5).float() * (1 - faz)
    masks2 = torch.cat([faz, ves, 1 - faz - ves], dim=1)
    heat2, dist2 = out[:, 4:6], out[:, 6:8]
    u = lambda: (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1)
    x2 = x2 * (1 + contrast * u()) + brightness * u()
    x2 = x2 + noise_std * torch.randn_like(x2) * (torch.rand(B, 1, 1, 1, device=dev) < 0.5)
    return x2, masks2, heat2, dist2
