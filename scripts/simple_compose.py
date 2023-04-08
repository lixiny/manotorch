from math import pi

import pyvista as pv
import torch
from trimesh import Trimesh

from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer, MANOOutput


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
    #*   3-- 2 -- 1 -----0   < NOTE: demo on this finger
    #   6 -- 5 -- 4 ----/
    #   12 - 11 - 10 --/
    #    9-- 8 -- 7 --/

    # NOTE: the ID: 1 joints have been rotated by pi/6 around spread-axis, and pi/2 around bend-axis
    composed_ee[:, 1] = torch.tensor([0, pi / 6, pi / 2]).unsqueeze(0)

    # NOTE: now, the ID: 2, 3 joints have been rotated by pi/2 around bend-axis
    composed_ee[:, 2] = torch.tensor([0, 0, pi / 2]).unsqueeze(0)
    composed_ee[:, 3] = torch.tensor([0, 0, pi / 2]).unsqueeze(0)

    composed_aa = axisFK.compose(composed_ee).clone()  # (B=1, 16, 3)
    composed_aa = composed_aa.reshape(1, -1)  # (1, 16x3)
    zero_shape = torch.zeros((1, 10))

    mano_output: MANOOutput = mano_layer(composed_aa, zero_shape)

    T_g_p = mano_output.transforms_abs  # (B=1, 16, 4, 4)
    T_g_a, R, ee = axisFK(T_g_p)
    T_g_a = T_g_a.squeeze(0)
    hand_verts = mano_output.verts.squeeze(0)  # (NV, 3)
    hand_faces = mano_layer.th_faces  # (NF, 3)
    mesh = pv.wrap(Trimesh(hand_verts, hand_faces))

    pl = pv.Plotter(off_screen=False, polygon_smoothing=True)
    pl.add_mesh(mesh, color=[0.9, 0, 0], name="mesh")
    pl.set_background('white')
    pl.add_camera_orientation_widget()
    pl.show(interactive=True)

    # ===== NOTE: common the above pl.show(), and uncommnet the following code to generate a gif >>>>>
    # view_up = [-1, 0, 0]
    # path = pl.generate_orbital_path(factor=2.0, n_points=36, viewup=view_up, shift=0.1)
    # pl.open_gif("orbit.gif")
    # pl.orbit_on_path(path, write_frames=True, step=0.05, viewup=view_up)
    # pl.close()


if __name__ == "__main__":
    main()
