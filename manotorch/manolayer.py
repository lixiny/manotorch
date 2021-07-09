import os
from collections import namedtuple
from typing import Optional
import warnings

import numpy as np
import torch

# import lietorch
from .utils.rodrigues import rodrigues
from .utils.quatutils import quaternion_to_rotation_matrix, quaternion_to_angle_axis

from mano.webuser.smpl_handpca_wrapper_HAND_only import ready_arguments

MANOOutput = namedtuple(
    "MANOOutput",
    [
        "verts",
        "joints",
        "center_idx",
        "center_joint",
        "full_poses",
        "betas",
        "transforms_abs",
    ],
)
MANOOutput.__new__.__defaults__ = (None,) * len(MANOOutput._fields)


def th_with_zeros(tensor):
    batch_size = tensor.shape[0]
    padding = tensor.new([0.0, 0.0, 0.0, 1.0])
    padding.requires_grad = False
    concat_list = [tensor, padding.view(1, 1, 4).repeat(batch_size, 1, 1)]
    cat_res = torch.cat(concat_list, 1)
    return cat_res


class ManoLayer(torch.nn.Module):
    def __init__(
        self,
        rot_mode: str = "axisang",
        side: str = "right",
        center_idx: Optional[int] = None,
        mano_assets_root: str = "assets/mano",
        use_pca: bool = False,
        flat_hand_mean: bool = True,  # Only used in pca mode
        ncomps: int = 15,  # Only used in pca mode
        **kargs,
    ):
        super().__init__()
        self.center_idx = center_idx
        self.rot_mode = rot_mode
        self.side = side
        self.use_pca = use_pca
        self.mano_assets_root = mano_assets_root
        self.flat_hand_mean = flat_hand_mean
        self.ncomps = ncomps if use_pca else -1


        if rot_mode == "axisang":
            self.rot_dim = 3
        elif rot_mode == "quat":
            self.rot_dim = 4
            if use_pca == True or flat_hand_mean == False:
                warnings.warn("Quat mode doesn't support PCA pose or non flat_hand_mean !")
        else:
            raise NotImplementedError(f"Unrecognized rotation mode, expect [pca|axisang|quat], got {rot_mode}")

        # load model according to side flag
        mano_assets_path = os.path.join(mano_assets_root, "models", f"MANO_{side.upper()}.pkl")  # eg.  MANO_RIGHT.pkl
        assert os.path.isfile(mano_assets_path), "Can not find MANO assets, please follow steps in README.md"

        # parse and register stuff
        smpl_data = ready_arguments(mano_assets_path)
        self.register_buffer("th_betas", torch.Tensor(np.array(smpl_data["betas"].r)).unsqueeze(0))
        self.register_buffer("th_shapedirs", torch.Tensor(np.array(smpl_data["shapedirs"].r)))
        self.register_buffer("th_posedirs", torch.Tensor(np.array(smpl_data["posedirs"].r)))
        self.register_buffer("th_v_template", torch.Tensor(np.array(smpl_data["v_template"].r)).unsqueeze(0))
        self.register_buffer("th_J_regressor", torch.Tensor(np.array(smpl_data["J_regressor"].toarray())))
        self.register_buffer("th_weights", torch.Tensor(np.array(smpl_data["weights"].r)))
        self.register_buffer("th_faces", torch.Tensor(np.array(smpl_data["f"]).astype(np.int32)).long())

        kintree_table = smpl_data["kintree_table"]
        self.kintree_parents = list(kintree_table[0].tolist())
        hands_components = smpl_data["hands_components"]

        if rot_mode == "axisang":
            hands_mean = np.zeros(hands_components.shape[1]) if flat_hand_mean else smpl_data["hands_mean"]
            hands_mean = hands_mean.copy()
            hands_mean = torch.Tensor(hands_mean).unsqueeze(0)
            self.register_buffer("th_hands_mean", hands_mean)

        if rot_mode == "axisang" and use_pca == True:
            selected_components = hands_components[:ncomps]
            selected_components = torch.Tensor(selected_components)
            self.register_buffer("th_selected_comps", selected_components)

        # End

    def rotation_by_axisang(self, pose_coeffs):
        batch_size = pose_coeffs.shape[0]
        hand_pose_coeffs = pose_coeffs[:, self.rot_dim :]
        root_pose_coeffs = pose_coeffs[:, : self.rot_dim]
        if self.use_pca:
            full_hand_pose = hand_pose_coeffs.mm(self.th_selected_comps)
        else:
            full_hand_pose = hand_pose_coeffs

        # Concatenate back global rot
        full_poses = torch.cat([root_pose_coeffs, self.th_hands_mean + full_hand_pose], 1)

        pose_vec_reshaped = full_poses.contiguous().view(-1, 3)  # (B x N, 3)
        rot_mats = rodrigues(pose_vec_reshaped)  # (B x N, 3, 3)
        # rot_mats = lietorch.SO3.exp(pose_vec_reshaped).matrix()[..., :3, :3]  # (B x N, 3, 3)
        full_rots = rot_mats.view(batch_size, 16, 3, 3)
        rotation_blob = {"full_rots": full_rots, "full_poses": full_poses}
        return rotation_blob

    def rotation_by_quaternion(self, pose_coeffs):
        batch_size = pose_coeffs.shape[0]
        full_quat_poses = pose_coeffs.view((batch_size, 16, 4))  # [B. 16, 4]
        full_rots = quaternion_to_rotation_matrix(full_quat_poses)  # [B, 16, 3, 3]
        full_poses = quaternion_to_angle_axis(full_quat_poses).reshape(batch_size, -1)  # [B, 16 x 3]

        rotation_blob = {"full_rots": full_rots, "full_poses": full_poses}
        return rotation_blob

    def skinning_layer(self, full_rots: torch.Tensor, betas: Optional[torch.Tensor]):
        batch_size = full_rots.shape[0]
        n_rot = int(full_rots.shape[1])  # 16

        root_rot = full_rots[:, 0, :, :]  # (B, 3, 3)
        hand_rot = full_rots[:, 1:, :, :]  # (B, 15, 3, 3)
        # Full axis angle representation with root joint

        # ============== Shape Blend Shape >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # $ B_S = \sum_{n=1}^{|\arrow{\beta}|} \beta_n \mathbf{S}_n $  #Eq.4 in MANO
        _betas = self.th_betas if betas is None else betas
        B_S = torch.matmul(self.th_shapedirs, _betas.transpose(1, 0)).permute(2, 0, 1)  # (?, 778, 3), ? = 1, or B

        # $ \mathcal{J}(\bar{\mathbf{T}} + B_S)$ # Eq.10 in SMPL
        J = torch.matmul(self.th_J_regressor, (self.th_v_template + B_S))  # (?, 16, 3)
        if betas is None:
            J = J.repeat(batch_size, 1, 1)  # (B, 16, 3)

        # ============== Pose Blender Shape >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        flat_rot = torch.eye(3, dtype=full_rots.dtype, device=full_rots.device)  # (3, 3)
        flat_rot = flat_rot.view(1, 1, 3, 3).repeat(batch_size, hand_rot.shape[1], 1, 1)  # (B, 15, 3, 3)

        # $ R_n (\arrow{\theta}) -  R_n (\arrow{\theta}^{*}) $
        rot_minus_mean_flat = (hand_rot - flat_rot).reshape(batch_size, hand_rot.shape[1] * 9)  # (B, 15 x 9)

        # $ B_P = \sum_{n=1}^{9K} (R_n (\arrow{\theta}) -  R_n (\arrow{\theta}^{*})) * \mathbf{P}_n $  #Eq.3 in MANO
        B_P = torch.matmul(self.th_posedirs, rot_minus_mean_flat.transpose(0, 1)).permute(2, 0, 1)  # (B, 778, 3)

        # $ T_P =\bar{\mathbf{T}} + B_S + B_P $ # Eq.2 in MANO
        T_P = self.th_v_template + B_S + B_P

        # ============== Constructing $ G_{k} $ >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # Global rigid transformation
        root_j = J[:, 0, :].contiguous().view(batch_size, 3, 1)
        root_transf = th_with_zeros(torch.cat([root_rot, root_j], 2))

        lev1_idxs = [1, 4, 7, 10, 13]
        lev2_idxs = [2, 5, 8, 11, 14]
        lev3_idxs = [3, 6, 9, 12, 15]
        lev1_rots = hand_rot[:, [idx - 1 for idx in lev1_idxs]]
        lev2_rots = hand_rot[:, [idx - 1 for idx in lev2_idxs]]
        lev3_rots = hand_rot[:, [idx - 1 for idx in lev3_idxs]]
        lev1_j = J[:, lev1_idxs]
        lev2_j = J[:, lev2_idxs]
        lev3_j = J[:, lev3_idxs]

        # From base to tips
        # Get lev1 results
        all_transforms = [root_transf.unsqueeze(1)]
        lev1_j_rel = lev1_j - root_j.transpose(1, 2)
        lev1_rel_transform_flt = th_with_zeros(torch.cat([lev1_rots, lev1_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        root_trans_flt = root_transf.unsqueeze(1).repeat(1, 5, 1, 1).view(root_transf.shape[0] * 5, 4, 4)
        lev1_flt = torch.matmul(root_trans_flt, lev1_rel_transform_flt)
        all_transforms.append(lev1_flt.view(hand_rot.shape[0], 5, 4, 4))

        # Get lev2 results
        lev2_j_rel = lev2_j - lev1_j
        lev2_rel_transform_flt = th_with_zeros(torch.cat([lev2_rots, lev2_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        lev2_flt = torch.matmul(lev1_flt, lev2_rel_transform_flt)
        all_transforms.append(lev2_flt.view(hand_rot.shape[0], 5, 4, 4))

        # Get lev3 results
        lev3_j_rel = lev3_j - lev2_j
        lev3_rel_transform_flt = th_with_zeros(torch.cat([lev3_rots, lev3_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        lev3_flt = torch.matmul(lev2_flt, lev3_rel_transform_flt)
        all_transforms.append(lev3_flt.view(hand_rot.shape[0], 5, 4, 4))

        reorder_idxs = [0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15]

        # Eq. 4 in SMPL
        G_k = torch.cat(all_transforms, 1)[:, reorder_idxs]
        th_transf_global = G_k

        # ============== Constructing $ G^{\prime}_{k} $ >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        joint_js = torch.cat([J, J.new_zeros(batch_size, 16, 1)], 2)
        tmp2 = torch.matmul(G_k, joint_js.unsqueeze(3))
        G_prime_k = (G_k - torch.cat([tmp2.new_zeros(*tmp2.shape[:2], 4, 3), tmp2], 3)).permute(0, 2, 3, 1)

        # ============== Finally, blender skinning >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # we define $ T = w_{k, i} * G^{\prime}_k $
        T = torch.matmul(G_prime_k, self.th_weights.transpose(0, 1))  # (B, 4, 4, 778)

        T_P_homo = torch.cat(
            [T_P.transpose(2, 1), torch.ones((batch_size, 1, B_P.shape[1]), dtype=T.dtype, device=T.device)], dim=1
        )
        T_P_homo = T_P_homo.unsqueeze(1)  # (B, 1, 4, 778)

        # Eq. 7 in SMPL
        # Theorem: A \cdot B = (A * B^{T}).sum(1) # A is a matrix, B is a vector
        verts = (T * T_P_homo).sum(2).transpose(2, 1)  # (B, 778, 4)
        joints = th_transf_global[:, :, :3, 3]  # (B, 16, 3)
        verts = verts[:, :, :3]  # (B, 778, 3)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # In addition to MANO reference joints we sample vertices on each finger
        # to serve as finger tips
        if self.side == "right":
            tips = verts[:, [745, 317, 444, 556, 673]]
        else:
            tips = verts[:, [745, 317, 445, 556, 673]]

        joints = torch.cat([joints, tips], 1)

        # ** original MANO joint order (right hand)
        #                16-15-14-13-\
        #                             \
        #          17 --3 --2 --1------0
        #        18 --6 --5 --4-------/
        #        19 -12 -11 --10-----/
        #          20 --9 --8 --7---/

        # Reorder joints to match SNAP definition
        joints = joints[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]

        if self.center_idx is not None:
            center_joint = joints[:, self.center_idx].unsqueeze(1)
        else:  # dummy center joint (B, 1, 3)
            center_joint = torch.zeros_like(joints[:, 0].unsqueeze(1))

        # apply center shift on verts and joints
        joints = joints - center_joint
        verts = verts - center_joint

        # apply center shift on global
        global_rot = th_transf_global[:, :, :3, :3]  # (B, 16, 3, 3)
        global_tsl = th_transf_global[:, :, :3, 3:]  # (B, 16, 3, 1)
        global_tsl = global_tsl - center_joint.unsqueeze(-1)  # (B, [16], 3, 1)
        global_transf = torch.cat([global_rot, global_tsl], dim=3)  # (B, 16, 3, 4)
        global_transf = th_with_zeros(global_transf.view(-1, 3, 4))
        global_transf = global_transf.view(batch_size, 16, 4, 4)

        skinning_blob = {
            "verts": verts,
            "joints": joints,
            "center_joint": center_joint,
            "transforms_abs": global_transf,
            "betas": _betas,
        }
        return skinning_blob

    def forward(self, pose_coeffs: torch.Tensor, betas: Optional[torch.Tensor] = None, **kwargs):
        if self.rot_mode == "axisang":
            rot_blob = self.rotation_by_axisang(pose_coeffs)
        elif self.rot_mode == "quat":
            rot_blob = self.rotation_by_quaternion(pose_coeffs)

        full_rots = rot_blob["full_rots"]  # TENSOR
        skinning_blob = self.skinning_layer(full_rots, betas)
        output = MANOOutput(
            verts=skinning_blob["verts"],
            joints=skinning_blob["joints"],
            center_idx=self.center_idx,
            center_joint=skinning_blob["center_joint"],
            full_poses=rot_blob["full_poses"],
            betas=skinning_blob["betas"],
            transforms_abs=skinning_blob["transforms_abs"],
        )
        return output

    def get_rotation_center(self, betas: Optional[torch.Tensor] = None):
        """

        V = MANO(theta, beta)

        Then we apply a rotation R on the vertices V
        V_1 = R @ V

        or, we can apply a rotation R on the global components of theta: first 3 elements of the theta
        theta' = CONCAT( SO3.log(R @ SO3.exp(theta[:3])), theta[3:] )

        V_2 = MANO(theta', beta)

        No doubt that, V_1 != V_2
        we found V_1 = V_2 + t, the t is an unknown translation offset

        Directly apply R on V would rotate V w.r.t the rotation center at V's [0,0,0] coordinate.
        However, apply R on the theta[:3] would cause the vertices rotate w.r.t to a rotation center at
        a non-zero, soley beta-sepcified center, C

        In other word, apply any disturb on the theta[:3] would not change the C's coordinates.
        the following code describe how we acquire the rotation center C

        This function will be called at artiboost/utils/refineunit.py in our upcoming work ArtiBoost
        """

        if betas is None:
            betas = self.th_betas

        batch_size = betas.shape[0]
        if self.center_idx is not None:
            return torch.zeros((batch_size, 3), device=betas.device)

        # $ B_S = \sum_{n=1}^{|\arrow{\beta}|} \beta_n \mathbf{S}_n $  #Eq.4 in MANO
        B_S = torch.matmul(self.th_shapedirs, betas.transpose(1, 0)).permute(2, 0, 1)

        # $ \mathcal{J}(\bar{\mathbf{T}} + B_S)$ # Eq.10 in SMPL
        J = torch.matmul(self.th_J_regressor, (self.th_v_template + B_S))  # (B, 16, 3)

        root_rotation_center = J[:, 0, :].contiguous().view(-1, 3)
        return root_rotation_center

    def get_mano_closed_faces(self):
        """
        The default MANO mesh is "open" at the wrist. By adding additional faces, the hand mesh is closed,
        which looks much better.
        https://github.com/hassony2/handobjectconsist/blob/master/meshreg/models/manoutils.py
        """
        close_faces = torch.Tensor(
            [
                [92, 38, 122],
                [234, 92, 122],
                [239, 234, 122],
                [279, 239, 122],
                [215, 279, 122],
                [215, 122, 118],
                [215, 118, 117],
                [215, 117, 119],
                [215, 119, 120],
                [215, 120, 108],
                [215, 108, 79],
                [215, 79, 78],
                [215, 78, 121],
                [214, 215, 121],
            ]
        )
        th_closed_faces = torch.cat([self.th_faces, close_faces.long()])
        # Indices of faces added during closing --> should be ignored as they match the wrist
        # part of the hand, which is not an external surface of the human

        # Valid because added closed faces are at the end
        hand_ignore_faces = [1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551]
        return th_closed_faces.detach().cpu() #, hand_ignore_faces