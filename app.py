import torch
import argparse
from manotorch.manolayer import ManoLayer, MANOOutput
from manotorch.axislayer import AxisLayer
from manotorch.anchorlayer import AnchorLayer
from manotorch.utils.visutils import display_hand_matplot, display_hand_pyrender, display_hand_open3d


def main(args):
    # Initialize MANO layer
    ncomps = 15
    mano_layer = ManoLayer(
        rot_mode="axisang",
        use_pca=True,
        side="right",
        center_idx=None,
        mano_assets_root="assets/mano",
        flat_hand_mean=args.flat_hand_mean,
        ncomps=ncomps,
    )
    axis_layer = AxisLayer()
    anchor_layer = AnchorLayer(anchor_root="assets/anchor")

    batch_size = 1
    # Generate random shape parameters
    random_shape = torch.rand(batch_size, 10)
    # Generate random pose parameters, including 3 values for global axis-angle rotation
    random_pose = torch.rand(batch_size, 3 + ncomps)

    mano_results: MANOOutput = mano_layer(random_pose, random_shape)
    verts = mano_results.verts
    joints = mano_results.joints
    transforms_abs = mano_results.transforms_abs

    anchors = anchor_layer(verts)
    bul_axes = axis_layer(joints, transforms_abs)

    if args.display == "pyrender":
        display_hand_pyrender(mano_results, mano_layer.th_faces, bul_axes=bul_axes, anchors=anchors)
    elif args.display == "open3d":
        display_hand_open3d(mano_results, mano_layer.th_faces)
    elif args.display == "matplot":
        display_hand_matplot(mano_results, mano_layer.th_faces)
    else:
        print("Unknown display")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat_hand_mean", action="store_true", help="Use flat hand as mean")
    parser.add_argument("--display", choices=["matplot", "pyrender", "open3d"], default="pyrender", type=str)
    main(parser.parse_args())
