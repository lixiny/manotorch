from math import pi

import numpy as np
import torch
from manotorch.manolayer import ManoLayer, MANOOutput
from manotorch.axislayer import AxisAdaptiveLayer, AxisLayerFK
from manotorch.utils.visutils import VizContext, create_coord_system_can
import open3d as o3d


def main():
    viz_ctx = VizContext(non_block=True)
    viz_ctx.init()
    geometry_to_viz = dict(
        hand_mesh_init=None,
        hand_mesh_curr=None,
        axis=None,
        coord_system_list=None,
    )

    mano_layer = ManoLayer(rot_mode="axisang",
                           center_idx=9,
                           mano_assets_root="assets/mano",
                           use_pca=False,
                           flat_hand_mean=True)
    hand_faces = mano_layer.th_faces  # (NF, 3)

    axisIK = AxisLayerFK(mano_assets_root="assets/mano")

    composed_ee = torch.zeros((1, 16, 3))
    composed_ee[:, 1] = torch.tensor([0, pi / 2, pi / 2]).unsqueeze(0)
    composed_aa = axisIK.compose(composed_ee)  # (B, 16, 3)
    zero_shape = torch.zeros((1, 10))
    composed_mano = mano_layer(composed_aa.reshape(1, -1), zero_shape)

    # new_aa_p1_p2.requires_grad_(True)

    # param = []
    # param.append({"params": [new_aa_p1_p2]})
    # optimizer = torch.optim.Adam(param, lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    for i, _ in enumerate(range(5000)):
        # zero_pose = torch.cat((zero_pose3, opti_pose, zero_pose2), dim=1)
        # zero_pose = torch.cat([zero_pose[:, :6], new_aa_p1_p2, zero_pose[:, 9:]], dim=1)
        curr_pose = composed_aa.reshape(1, -1)
        mano_output: MANOOutput = mano_layer(curr_pose, zero_shape)
        hand_verts_curr = mano_output.verts

        T_g_p = mano_output.transforms_abs  # (B, 16, 4, 4)
        ikres = axisIK(T_g_p)
        T_g_a = ikres[0]
        ee_tmpla2_a2 = ikres[2][:, 2]

        # b_loss = torch.abs(ee_tmpla2_a2[:, 0]).mean()
        # u_loss = torch.abs(ee_tmpla2_a2[:, 1]).mean()
        # l_loss = torch.abs((ee_tmpla2_a2[:, 2])).mean()

        # loss = b_loss + u_loss + l_loss
        # proc_bar.set_description(f"b: {b_loss.item():.5f} | "
        #                          f"u: {u_loss.item():.5f} | "
        #                          f"l: {l_loss.item():.5f} | ")
        # loss.backward()
        # optimizer.step()
        # scheduler.step()

        # ===== draw hand curr >>>>>
        if i % 10 == 0:
            if geometry_to_viz["coord_system_list"] is not None:
                viz_ctx.remove_geometry_list(geometry_to_viz["coord_system_list"])

            coord_system_list = []
            for i in range(T_g_a.shape[1]):
                coord_system_list += create_coord_system_can(scale=1, transf=T_g_a[0, i].detach().cpu().numpy())
            viz_ctx.add_geometry_list(coord_system_list)
            geometry_to_viz["coord_system_list"] = coord_system_list

        if geometry_to_viz.get('hand_mesh_curr', None) is None:
            o3d_hand_mesh_curr = o3d.geometry.TriangleMesh()
            o3d_hand_mesh_curr.triangles = o3d.utility.Vector3iVector(hand_faces.detach().cpu().numpy())
            o3d_hand_mesh_curr.vertices = o3d.utility.Vector3dVector(hand_verts_curr[0].detach().cpu().numpy())
            o3d_hand_mesh_curr.compute_vertex_normals()
            o3d_hand_mesh_curr.compute_triangle_normals()
            o3d_hand_mesh_curr.paint_uniform_color([0.9, 0.0, 0.0])
            viz_ctx.add_geometry(o3d_hand_mesh_curr)
            geometry_to_viz["hand_mesh_curr"] = o3d_hand_mesh_curr
        else:
            o3d_hand_mesh_curr = geometry_to_viz["hand_mesh_curr"]
            o3d_hand_mesh_curr.vertices = o3d.utility.Vector3dVector(hand_verts_curr[0].detach().cpu().numpy())
            o3d_hand_mesh_curr.compute_vertex_normals()
            o3d_hand_mesh_curr.compute_triangle_normals()
            viz_ctx.update_geometry(o3d_hand_mesh_curr)
        # <<<<<

        viz_ctx.step()


if __name__ == "__main__":
    main()
