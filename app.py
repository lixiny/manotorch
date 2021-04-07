import torch
from manotorch.manolayer import ManoLayer
from manotorch.axislayer import AxisLayer
from manotorch.utils.visutils import display_hand_matplot, display_hand_pyrender,display_hand_open3d



batch_size = 1
# Select number of principal components for pose space
ncomps = 6

# Initialize MANO layer
mano_layer = ManoLayer(
    rot_mode="axisang",
    use_pca=True,
    side="right",
    center_idx=None,
    mano_assets_root="assets",
    flat_hand_mean=False,
    ncomps=18,
)

axis_layer = AxisLayer()

# Generate random shape parameters
random_shape = torch.rand(batch_size, 10)
# Generate random pose parameters, including 3 values for global axis-angle rotation
random_pose = torch.rand(batch_size, 21)

# Forward pass through MANO layer
mano_results = mano_layer(random_pose, random_shape)
verts = mano_results.verts
joints = mano_results.joints
full_poses = mano_results.full_poses
center_idx = mano_results.center_idx
transforms_abs = mano_results.transforms_abs

bul_axes = axis_layer(joints, transforms_abs)

display_hand_pyrender(mano_results, mano_layer.th_faces, bul_axes=bul_axes)


#
# a = torch.isclose(full_poses, full_poses_)
# diff = full_transforms_abs - full_transforms_abs_
# print(diff.max())