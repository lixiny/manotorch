import os

import numpy as np
import torch
from torch.nn import Module

from .utils.anchorutils import anchor_load, recover_anchor, recover_anchor_batch


class AnchorLayer(Module):
    def __init__(self, anchor_root="assets/anchor"):
        super().__init__()

        face_vert_idx, anchor_weight, merged_vertex_assignment, anchor_mapping = anchor_load(anchor_root)
        self.register_buffer("face_vert_idx", torch.from_numpy(face_vert_idx).long().unsqueeze(0))
        self.register_buffer("anchor_weight", torch.from_numpy(anchor_weight).float().unsqueeze(0))
        self.register_buffer("merged_vertex_assignment", torch.from_numpy(merged_vertex_assignment).long())
        self.anchor_mapping = anchor_mapping

    def forward(self, vertices):
        """
        vertices: TENSOR[N_BATCH, 778, 3]
        """
        anchor_pos = recover_anchor_batch(vertices, self.face_vert_idx, self.anchor_weight)
        # anchor_pos2 = recover_anchor(vertices[vertices.shape[0] - 1], self.face_vert_idx[0], self.anchor_weight[0])
        return anchor_pos
