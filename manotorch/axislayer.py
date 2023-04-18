import os
import warnings

import numpy as np
import torch
from torch.nn import Module

from deprecation import deprecated
from manotorch.manolayer import ManoLayer
from manotorch.utils.geometry import matrix_to_euler_angles, euler_angles_to_matrix, rotation_to_axis_angle


class AxisAdaptiveLayer(torch.nn.Module):

    def __init__(self):
        super(AxisAdaptiveLayer, self).__init__()
        self.joints_mapping = [5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3]
        self.parent_joints_mappings = [0, 5, 6, 0, 9, 10, 0, 17, 18, 0, 13, 14, 0, 1, 2]
        up_axis_base = np.vstack((np.array([[0, 1, 0]]).repeat(13, axis=0), np.array([[1, 1, 1]]).repeat(3, axis=0)))
        self.register_buffer("up_axis_base", torch.from_numpy(up_axis_base).float().unsqueeze(0))

    def forward(self, hand_joints, transf):
        """ Compute the back (twist), up (spread), and left (bend) axes direction of the hand
        Args:
            hand_joints (torch.Tensor): (B, 16, 3)
            transf (torch.Tensor): (B, 16, 4, 4)
        Returns:
            b_axis (torch.Tensor): (B, 16, 3)
            u_axis (torch.Tensor): (B, 16, 3)
            l_axis (torch.Tensor): (B, 16, 3)
        """
        bs = transf.shape[0]

        # b_axis = hand_joints[:, self.joints_mapping] - hand_joints[:, [i + 1 for i in self.joints_mapping]]
        b_axis = hand_joints[:, self.parent_joints_mappings] - hand_joints[:, self.joints_mapping]
        b_axis = (transf[:, 1:, :3, :3].transpose(2, 3) @ b_axis.unsqueeze(-1)).squeeze(-1)
        b_axis_init = torch.tensor([1, 0, 0]).float().unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1).to(b_axis.device)
        b_axis = torch.cat((b_axis_init, b_axis), dim=1)  # (B, 16, 3)

        l_axis = torch.cross(b_axis, self.up_axis_base.expand(bs, 16, 3))

        u_axis = torch.cross(l_axis, b_axis)

        return (
            b_axis / torch.norm(b_axis, dim=2, keepdim=True),
            u_axis / torch.norm(u_axis, dim=2, keepdim=True),
            l_axis / torch.norm(l_axis, dim=2, keepdim=True),
        )


class AxisLayerFK(Module):

    def __init__(self, side: str = "right", mano_assets_root: str = "assets/mano"):
        super(AxisLayerFK, self).__init__()
        self.transf_parent_mapping = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]

        tmpl_pose = torch.zeros(1, 48)
        tmpl_shape = torch.zeros(1, 10)
        tmpl_mano = ManoLayer(side=side, mano_assets_root=mano_assets_root)(tmpl_pose, tmpl_shape)
        tmpl_joints = tmpl_mano.joints
        tmpl_transf_abs = tmpl_mano.transforms_abs  # tmpl_T_g_p

        tmpl_b_axis, tmpl_u_axis, tmpl_l_axis = AxisAdaptiveLayer()(tmpl_joints, tmpl_transf_abs)  # (1, 16, 3)
        tmpl_R_p_a = torch.cat((tmpl_b_axis.unsqueeze(-1), tmpl_u_axis.unsqueeze(-1), tmpl_l_axis.unsqueeze(-1)), dim=3)
        zero_tsl = torch.zeros(1, 16, 3, 1)
        zero_pad = torch.tensor([[[[0, 0, 0, 1]]]]).repeat(*zero_tsl.shape[0:2], 1, 1)
        _tmpl_T_p_a = torch.cat((tmpl_R_p_a, zero_tsl), dim=3)  # (1, 16, 3, 4)
        tmpl_T_p_a = torch.cat((_tmpl_T_p_a, zero_pad), dim=2)  # (1, 16, 4, 4)
        tmpl_T_g_a = torch.matmul(tmpl_transf_abs, tmpl_T_p_a)  # (1, 16, 4, 4)
        self.register_buffer("TMPL_T_p_a", tmpl_T_p_a.float())
        self.register_buffer("TMPL_R_p_a", tmpl_R_p_a.float())
        self.register_buffer("TMPL_T_g_a", tmpl_T_g_a.float())

    def forward(self, transf):
        """extract the anatomy aligned euler angles from the MANO global transformation
        #  transform order of right hand
        #         15-14-13-\
        #                   \
        #    3-- 2 -- 1 -----0
        #   6 -- 5 -- 4 ----/
        #   12 - 11 - 10 --/
        #    9-- 8 -- 7 --/ 

        Args:
            transf (torch.Tensor): [B, 16, 4, 4] MANO'joints global transformation 
                defined in MANO origial frame

        Returns:
            T_g_a: torch.Tensor: [B, 16, 4, 4] MANO's joints global transformation defined in  anatomy aligned frame;
            Ra_tmplchd_chd: torch.Tensor: [B, 16, 3, 3], anatomy aligned rotation matrix;
            ee_a_tmplchd_chd: torch.Tensor: [B, 16, 3] anatomy aligned euler angle;
        """

        T_g_p = transf
        R_g_p = T_g_p[:, :, :3, :3]
        R_g_a = torch.matmul(R_g_p, self.TMPL_R_p_a)  # (B, 16, 3, 3)
        T_g_a = torch.cat((R_g_a, T_g_p[:, :, :3, 3:]), dim=3)  # (B, 16, 3, 4)
        zero_pad = torch.tensor([[[[0, 0, 0, 1]]]]).repeat(*T_g_a.shape[0:2], 1, 1).to(T_g_a.device)
        T_g_a = torch.cat((T_g_a, zero_pad), dim=2)  # (B, 16, 4, 4)

        Ta_par_chd = torch.matmul(T_g_a[:, self.transf_parent_mapping, ...].transpose(2, 3), T_g_a)  # (B, 16, 4, 4)
        Ra_par_chd = Ta_par_chd[:, :, :3, :3]  # (B, 16, 3, 3)

        Ra_par_tmplchd = torch.matmul(self.TMPL_R_p_a[:, self.transf_parent_mapping, ...].transpose(2, 3),
                                      self.TMPL_R_p_a)
        Ra_chd_tmplchd = torch.matmul(Ra_par_chd.transpose(2, 3), Ra_par_tmplchd)
        Ra_tmplchd_chd = Ra_chd_tmplchd.transpose(2, 3)

        ee_a_tmplchd_chd = matrix_to_euler_angles(Ra_tmplchd_chd, convention="XYZ")  # (B, 16, 3)
        return T_g_a, Ra_tmplchd_chd, ee_a_tmplchd_chd

    def compose(self, angles):
        """Compose the MANO pose (\theta) from the anatomy aligned euler angles.
        #  transform order of right hand
        #         15-14-13-\
        #                   \
        #    3-- 2 -- 1 -----0
        #   6 -- 5 -- 4 ----/
        #   12 - 11 - 10 --/
        #    9-- 8 -- 7 --/ 

        Args:
            angles (torch.Tensor): [B, 16, 3] anatomy aligned euler angles

        Returns:
            torch.Tensor: mano pose (\theta) in the MANO original frame
        """
        ee_tmplchd_chd = angles  # (B, 16, 3)
        Ra_tmplchd_chd = euler_angles_to_matrix(ee_tmplchd_chd, convention="XYZ")  # (B, 16, 3, 3)

        Ra_par_tmplchd = torch.matmul(self.TMPL_R_p_a[:, self.transf_parent_mapping, ...].transpose(2, 3),
                                      self.TMPL_R_p_a)
        Ra_par_chd = torch.matmul(Ra_par_tmplchd, Ra_tmplchd_chd)  # (B, 16, 3, 3)

        # chains
        lev1_idxs = [1, 4, 7, 10, 13]
        lev2_idxs = [2, 5, 8, 11, 14]
        lev3_idxs = [3, 6, 9, 12, 15]

        all_rot_chains = [Ra_par_chd[:, 0:1]]
        lev1_rots = Ra_par_chd[:, [idx for idx in lev1_idxs]]  # (B, 5, 3, 3)
        lev2_rots = Ra_par_chd[:, [idx for idx in lev2_idxs]]  # (B, 5, 3, 3)
        lev3_rots = Ra_par_chd[:, [idx for idx in lev3_idxs]]  # (B, 5, 3, 3)

        lev1_rot_chains = torch.matmul(Ra_par_chd[:, 0:1].repeat(1, 5, 1, 1), lev1_rots)  # (B, 5, 3, 3)
        all_rot_chains.append(lev1_rot_chains)
        lev2_rot_chains = torch.matmul(lev1_rot_chains, lev2_rots)  # (B, 5, 3, 3)
        all_rot_chains.append(lev2_rot_chains)
        lev3_rot_chains = torch.matmul(lev2_rot_chains, lev3_rots)  # (B, 5, 3, 3)
        all_rot_chains.append(lev3_rot_chains)
        reorder_idxs = [0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15]
        R_g_a = torch.cat(all_rot_chains, 1)[:, reorder_idxs]  # (B, 16, 3, 3)
        R_g_p = torch.matmul(R_g_a, self.TMPL_R_p_a.transpose(2, 3))  # (B, 16, 3, 3)

        Rp_par_chd = torch.matmul(R_g_p[:, self.transf_parent_mapping].transpose(2, 3), R_g_p)  # (B, 16, 3, 3)
        aa_p_par_chd = rotation_to_axis_angle(Rp_par_chd)  # (B, 16, 3)
        return aa_p_par_chd


@deprecated(deprecated_in="0.0.2",
            removed_in="0.0.3",
            details="This class is deprecated. Please use the new class 'AxisLayerFK' instead.")
class AxisLayer(Module):

    def __init__(self):
        super().__init__()
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