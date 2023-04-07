import numpy as np
import torch
import deprecation


class VizContext:

    def __init__(self, non_block=False) -> None:
        import open3d as o3d
        self.vis = o3d.visualization.VisualizerWithKeyCallback()

        self.running = True

        def shutdown_callback(vis):
            self.running = False

        self.vis.register_key_callback(ord("Q"), shutdown_callback)

        self.non_block = non_block

    def init(self, point_size=10.0):
        self.vis.create_window()
        self.vis.get_render_option().point_size = point_size
        self.vis.get_render_option().background_color = np.asarray([1, 1, 1])

    def deinit(self):
        self.vis.destroy_window()

    def add_geometry(self, pc):
        self.vis.add_geometry(pc)

    def add_geometry_list(self, pc_list):
        for pc in pc_list:
            self.vis.add_geometry(pc, reset_bounding_box=False)

    def update_geometry_list(self, pc_list):
        for pc in pc_list:
            self.vis.update_geometry(pc)

    def update_geometry(self, pc):
        self.vis.update_geometry(pc)

    def step(self):
        self.vis.poll_events()
        self.vis.update_renderer()

    def remove_geometry(self, pc):
        self.vis.remove_geometry(pc)

    def remove_geometry_list(self, pc_list):
        for pc in pc_list:
            self.vis.remove_geometry(pc, reset_bounding_box=False)

    def reset(self):
        self.running = True

    def condition(self):
        return self.running and (not self.non_block)


def caculate_align_mat(vec):
    vec = vec / np.linalg.norm(vec)
    z_unit_Arr = np.array([0, 0, 1])

    z_mat = np.array([
        [0, -z_unit_Arr[2], z_unit_Arr[1]],
        [z_unit_Arr[2], 0, -z_unit_Arr[0]],
        [-z_unit_Arr[1], z_unit_Arr[0], 0],
    ])

    z_c_vec = np.matmul(z_mat, vec)
    z_c_vec_mat = np.array([
        [0, -z_c_vec[2], z_c_vec[1]],
        [z_c_vec[2], 0, -z_c_vec[0]],
        [-z_c_vec[1], z_c_vec[0], 0],
    ])

    if np.dot(z_unit_Arr, vec) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, vec) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat, z_c_vec_mat) / (1 + np.dot(z_unit_Arr, vec))

    return qTrans_Mat


def create_coord_system_can(scale=1, transf=None):
    import open3d as o3d
    axis_list = []
    cylinder_radius = 0.0015 * scale
    cone_radius = 0.002 * scale
    cylinder_height = 0.05 * scale
    cone_height = 0.008 * scale
    resolution = int(20 * scale)
    cylinder_split = 4
    cone_split = 1

    x = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius,
        cone_radius=cone_radius,
        cylinder_height=cylinder_height,
        cone_height=cone_height,
        resolution=resolution,
        cylinder_split=cylinder_split,
        cone_split=cone_split,
    )
    x.paint_uniform_color([255 / 255.0, 0 / 255.0, 0 / 255.0])
    align_x = caculate_align_mat(np.array([1, 0, 0]))
    x = x.rotate(align_x, center=(0, 0, 0))
    x.compute_vertex_normals()
    axis_list.append(x)

    y = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius,
        cone_radius=cone_radius,
        cylinder_height=cylinder_height,
        cone_height=cone_height,
        resolution=resolution,
        cylinder_split=cylinder_split,
        cone_split=cone_split,
    )
    y.paint_uniform_color([0 / 255.0, 255 / 255.0, 0 / 255.0])

    align_y = caculate_align_mat(np.array([0, 1, 0]))
    y = y.rotate(align_y, center=(0, 0, 0))
    y.compute_vertex_normals()
    axis_list.append(y)

    z = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius,
        cone_radius=cone_radius,
        cylinder_height=cylinder_height,
        cone_height=cone_height,
        resolution=resolution,
        cylinder_split=cylinder_split,
        cone_split=cone_split,
    )
    z.paint_uniform_color([0 / 255.0, 0 / 255.0, 255 / 255.0])
    align_z = caculate_align_mat(np.array([0, 0, 1]))
    z = z.rotate(align_z, center=(0, 0, 0))
    z.compute_vertex_normals()
    axis_list.append(z)

    if transf is not None:
        assert transf.shape == (4, 4), "transf must be 4x4 Transformation matrix"
        for i, axis in enumerate(axis_list):
            axis.rotate(transf[:3, :3], center=(0, 0, 0))
            axis.translate(transf[:3, 3].T)
            axis_list[i] = axis

    return axis_list


@deprecation.deprecated(deprecated_in="0.0.2",
                        removed_in="0.0.3",
                        details="this function is deprecated due to the modification on bul axes")
def draw_axis(axis, transf, scene, color):
    import pyrender
    import trimesh
    end_points = np.concatenate([axis[np.newaxis] * 20.0, np.zeros((1, 3))], axis=0)
    end_points = np.concatenate([end_points, np.ones((2, 1))], axis=1)
    end_points = (transf @ end_points.T).T[:, :3]

    rot_matrix = np.concatenate([caculate_align_mat(axis), np.zeros((3, 1))], axis=1)
    rot_matrix = np.concatenate([rot_matrix, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)

    cylinder = trimesh.creation.cylinder(radius=0.9, segment=end_points)
    cylinder.visual.vertex_colors = color
    cylinder = pyrender.Mesh.from_trimesh(cylinder, smooth=False)
    scene.add(cylinder)


@deprecation.deprecated(deprecated_in="0.0.2",
                        removed_in="0.0.3",
                        details="this function is deprecated due to the modification on bul axes")
def display_hand_pyrender(mano_blob, faces=None, batch_idx=0, show_axis=True, anchors=None, bul_axes=None):
    import pyrender
    import trimesh

    # region Viewer Options >>>>>>>>>
    scene = pyrender.Scene()
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
    node_cam = pyrender.Node(camera=cam, matrix=np.eye(4))
    scene.add_node(node_cam)
    scene.set_pose(node_cam, pose=np.eye(4))
    vertex_colors = np.array([200, 200, 200, 150])
    joint_colors = np.array([10, 73, 233, 255])
    # endregion <<<<<<<<<<<<

    verts, joints = mano_blob.verts[batch_idx], mano_blob.joints[batch_idx]
    transforms = mano_blob.transforms_abs[batch_idx]
    dt = np.array([0, 0, -270.0])
    dt = dt[np.newaxis, :]

    joints = joints.detach().cpu().numpy()
    verts = verts.detach().cpu().numpy()
    transforms = np.array(transforms.detach().cpu())

    joints = joints * 1000.0 + dt
    verts = verts * 1000.0 + dt
    transforms[:, :3, 3] = transforms[:, :3, 3] * 1000.0 + dt

    tri_mesh = trimesh.Trimesh(verts, faces, vertex_colors=vertex_colors)
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    scene.add(mesh)

    # add joints
    for j in range(21):
        sm = trimesh.creation.uv_sphere(radius=2.5)
        sm.visual.vertex_colors = joint_colors
        tfs = np.tile(np.eye(4), (1, 1, 1))
        tfs[0, :3, 3] = joints[j]
        joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        scene.add(joints_pcl)

    # add transformation
    if show_axis:
        if bul_axes is not None:
            b_axis, u_axis, l_axis = bul_axes[0][batch_idx], bul_axes[1][batch_idx], bul_axes[2][batch_idx]
            b_axis = b_axis.detach().cpu().numpy()
            u_axis = u_axis.detach().cpu().numpy()
            l_axis = l_axis.detach().cpu().numpy()
            for i in range(15):
                draw_axis(b_axis[i], transforms[i + 1], scene, color=np.array([255, 42, 34, 255]))
                draw_axis(u_axis[i], transforms[i + 1], scene, color=np.array([190, 255, 0, 255]))
                draw_axis(l_axis[i], transforms[i + 1], scene, color=np.array([23, 217, 255, 255]))

        else:
            for i in range(16):
                axis = trimesh.creation.axis(transform=transforms[i], origin_size=3, axis_length=21)
                axis = pyrender.Mesh.from_trimesh(axis, smooth=False)
                scene.add(axis)

    if anchors is not None:
        anchors = anchors[batch_idx]
        for k in range(len(anchors)):
            anchor_sphere = trimesh.creation.box(extents=(3, 3, 3))
            anchor_sphere.visual.vertex_colors = np.array([250, 255, 0, 255])
            tfs = np.tile(np.eye(4), (1, 1, 1))
            tfs[0, :3, 3] = anchors[k] * 1000 + dt
            anchor_mesh = pyrender.Mesh.from_trimesh(anchor_sphere, poses=tfs)
            scene.add(anchor_mesh)

    pyrender.Viewer(scene, viewport_size=(1280, 768), use_raymond_lighting=True)
