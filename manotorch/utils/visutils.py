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
