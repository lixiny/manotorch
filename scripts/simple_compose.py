from math import pi

import open3d as o3d
import torch

from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer, MANOOutput
from manotorch.utils.visutils import VizContext, create_coord_system_can


def main():
    mano_layer = ManoLayer(rot_mode="axisang",
                           center_idx=9,
                           mano_assets_root="assets/mano",
                           use_pca=False,
                           flat_hand_mean=True)
    hand_faces = mano_layer.th_faces  # (NF, 3)

    axisFK = AxisLayerFK(mano_assets_root="assets/mano")

    composed_ee = torch.zeros((1, 16, 3))

    #  transform order of right hand
    #         15-14-13-\
    #                   \
    #    3-- 2 -- 1 -----0   < NOTE: demo on this finger
    #   6 -- 5 -- 4 ----/
    #   12 - 11 - 10 --/
    #    9-- 8 -- 7 --/

    # NOTE: the ID: 1 joints have been rotated by pi/6 around spread-axis, and pi/2 around bend-axis
    composed_ee[:, 1] = torch.tensor([0, pi / 6, pi / 2]).unsqueeze(0)

    # NOTE: now, the ID: 2, 3 joints have been rotated by pi/2 around bend-axis
    composed_ee[:, 2] = torch.tensor([0, 0, pi / 2]).unsqueeze(0)
    composed_ee[:, 3] = torch.tensor([0, 0, pi / 2]).unsqueeze(0)

    composed_aa = axisFK.compose(composed_ee).clone()  # (B, 16, 3)
    composed_aa = composed_aa.reshape(1, -1)  # (1, 16x3)
    zero_shape = torch.zeros((1, 10))

    mano_output: MANOOutput = mano_layer(composed_aa, zero_shape)

    T_g_p = mano_output.transforms_abs  # (B, 16, 4, 4)
    T_g_a, R, ee = axisFK(T_g_p)
    T_g_a = T_g_a.squeeze(0)
    hand_verts = mano_output.verts.squeeze(0)  # (B, NV, 3)
    hand_faces = mano_layer.th_faces  # (NF, 3)

    viz_ctx = VizContext(non_block=True)
    viz_ctx.init()
    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = o3d.utility.Vector3iVector(hand_faces.detach().cpu().numpy())
    mesh.vertices = o3d.utility.Vector3dVector(hand_verts.detach().cpu().numpy())
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    mesh.paint_uniform_color([0.9, 0.0, 0.0])
    viz_ctx.add_geometry(mesh)

    coord_system_list = []
    for i in range(T_g_a.shape[0]):
        coord_system_list += create_coord_system_can(scale=0.7, transf=T_g_a[i].detach().cpu().numpy())
    viz_ctx.add_geometry_list(coord_system_list)

    while True:
        viz_ctx.step()


if __name__ == "__main__":
    main()
