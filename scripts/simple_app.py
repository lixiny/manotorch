import argparse

import numpy as np
import pyvista as pv
import torch
from trimesh import Trimesh

from manotorch.anchorlayer import AnchorLayer
from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer, MANOOutput


def main(args):
    mano_layer = ManoLayer(
        rot_mode="axisang",
        use_pca=False,
        side="right",
        center_idx=None,
        mano_assets_root="assets/mano",
        flat_hand_mean=False,
    )
    axis_layer = AxisLayerFK(mano_assets_root="assets/mano")
    anchor_layer = AnchorLayer(anchor_root="assets/anchor")

    BS = 1
    random_shape = torch.rand(BS, 10)
    root_pose = torch.tensor([[0, np.pi / 2, 0]]).repeat(BS, 1)
    finger_pose = torch.zeros(BS, 45)
    hand_pose = torch.cat([root_pose, finger_pose], dim=1)

    mano_results: MANOOutput = mano_layer(hand_pose, random_shape)
    verts = mano_results.verts
    faces = mano_layer.th_faces
    V = verts[0].numpy()
    F = faces.numpy()
    tmesh = Trimesh(V, F)
    mesh = pv.wrap(tmesh)

    T_g_p = mano_results.transforms_abs  # (B, 16, 4, 4)
    T_g_a, R, ee = axis_layer(T_g_p)  # ee (B, 16, 3)

    bul_axes_loc = torch.eye(3).reshape(1, 1, 3, 3).repeat(BS, 16, 1, 1).to(verts.device)
    bul_axes_glb = torch.matmul(T_g_a[:, :, :3, :3], bul_axes_loc)  # (B, 16, 3, 3)

    b_axes_dir = bul_axes_glb[:, :, :, 0].numpy()  # bend direction (B, 16, 3)
    u_axes_dir = bul_axes_glb[:, :, :, 1].numpy()  # up direction (B, 16, 3)
    l_axes_dir = bul_axes_glb[:, :, :, 2].numpy()  # left direction (B, 16, 3)

    axes_cen = T_g_a[:, :, :3, 3].numpy()  # center (B, 16, 3)

    pl = pv.Plotter(off_screen=False)
    pl.add_mesh(mesh, opacity=0.4, name="mesh", smooth_shading=True)

    if args.mode == "axis":
        pl.add_arrows(axes_cen, b_axes_dir, color="red", mag=0.02)
        pl.add_arrows(axes_cen, u_axes_dir, color="yellow", mag=0.02)
        pl.add_arrows(axes_cen, l_axes_dir, color="blue", mag=0.02)
    elif args.mode == "anchor":
        anchors = anchor_layer(verts)[0].numpy()
        n_achors = anchors.shape[0]
        for i in range(n_achors):
            pl.add_mesh(pv.Cube(center=anchors[i], x_length=3e-3, y_length=3e-3, z_length=3e-3),
                        color="yellow",
                        name=f"anchor{i}")

    pl.set_background('white')
    pl.add_camera_orientation_widget()
    pl.show(interactive=True)

    # ===== NOTE: common the above pl.show(), and uncommnet the following code to generate a gif >>>>>
    # path = pl.generate_orbital_path(factor=2.0, n_points=36, shift=0.1)
    # pl.open_gif("orbit.gif")
    # pl.orbit_on_path(path, write_frames=True, step=0.05)
    # pl.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["axis", "anchor"], default="axis", help="visualize axis or anchor")
    main(parser.parse_args())