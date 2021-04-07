import os

import numpy as np
import torch
from torch.nn import Module


class AxisLayer(Module):
    def __init__(self):
        super(AxisLayer, self).__init__()
        self.joints_mapping = [5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3]
        up_axis_base = np.vstack((np.array([[0, 1, 0]]).repeat(12, axis=0), np.array([[1, 1, 1]]).repeat(3, axis=0)))
        self.register_buffer("up_axis_base", torch.from_numpy(up_axis_base).float().unsqueeze(0))

    def forward(self, hand_joints, transf):
        """
        input: hand_joints[B, 21, 3], transf[B, 16, 4, 4]
        output: b_axis[B, 15, 3], u_axis[B, 15, 3], l_axis[B, 15, 3]
        b: back; u: up; l: left
        """
        bs = transf.shape[0]

        b_axis = hand_joints[:, self.joints_mapping] - hand_joints[:, [i + 1 for i in self.joints_mapping]]
        b_axis = (transf[:, 1:, :3, :3].transpose(2, 3) @ b_axis.unsqueeze(-1)).squeeze(-1)

        l_axis = torch.cross(b_axis, self.up_axis_base.expand(bs, 15, 3))

        u_axis = torch.cross(l_axis, b_axis)

        return (
            b_axis / torch.norm(b_axis, dim=2, keepdim=True),
            u_axis / torch.norm(u_axis, dim=2, keepdim=True),
            l_axis / torch.norm(l_axis, dim=2, keepdim=True),
        )
