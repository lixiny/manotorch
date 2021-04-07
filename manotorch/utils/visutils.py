import numpy as np
import torch


def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)


def display_hand_matplot(
    mano_blob,
    faces,
    ax=None,
    alpha=0.2,
    cam_view=False,
    batch_idx=0,
    show=True,
):
    """
    Displays hand batch_idx in batch of hand_info, hand_info as returned by
    generate_random_hand
    """
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    verts, joints = mano_blob.verts[batch_idx], mano_blob.joints[batch_idx]

    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    face_color = (141 / 255, 184 / 255, 226 / 255)
    edge_color = (50 / 255, 50 / 255, 50 / 255)
    mesh.set_edgecolor(edge_color)
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    # region: joint color >>>>>>>
    ax.scatter(joints[0, 0], joints[0, 1], joints[0, 2], s=42, color="c", marker="p")

    ax.scatter(joints[1, 0], joints[1, 1], joints[1, 2], color="y", marker="s")
    ax.scatter(joints[2, 0], joints[2, 1], joints[2, 2], color="y", marker="^")
    ax.scatter(joints[3, 0], joints[3, 1], joints[3, 2], color="y", marker="o")
    ax.scatter(joints[4, 0], joints[4, 1], joints[4, 2], color="y", marker="*")

    ax.scatter(joints[5, 0], joints[5, 1], joints[5, 2], color="r", marker="s")
    ax.scatter(joints[6, 0], joints[6, 1], joints[6, 2], color="r", marker="^")
    ax.scatter(joints[7, 0], joints[7, 1], joints[7, 2], color="r", marker="o")
    ax.scatter(joints[8, 0], joints[8, 1], joints[8, 2], color="r", marker="*")

    ax.scatter(joints[9, 0], joints[9, 1], joints[9, 2], color="b", marker="s")
    ax.scatter(joints[10, 0], joints[10, 1], joints[10, 2], color="b", marker="^")
    ax.scatter(joints[11, 0], joints[11, 1], joints[11, 2], color="b", marker="o")
    ax.scatter(joints[12, 0], joints[12, 1], joints[12, 2], color="b", marker="*")

    ax.scatter(joints[13, 0], joints[13, 1], joints[13, 2], color="g", marker="s")
    ax.scatter(joints[14, 0], joints[14, 1], joints[14, 2], color="g", marker="^")
    ax.scatter(joints[15, 0], joints[15, 1], joints[15, 2], color="g", marker="o")
    ax.scatter(joints[16, 0], joints[16, 1], joints[16, 2], color="g", marker="*")

    ax.scatter(joints[17, 0], joints[17, 1], joints[17, 2], color="m", marker="s")
    ax.scatter(joints[18, 0], joints[18, 1], joints[18, 2], color="m", marker="^")
    ax.scatter(joints[19, 0], joints[19, 1], joints[19, 2], color="m", marker="o")
    ax.scatter(joints[20, 0], joints[20, 1], joints[20, 2], color="m", marker="*")
    # endregion <<<<<<<<<<

    if cam_view:
        ax.view_init(azim=-90.0, elev=-90.0)
    cam_equal_aspect_3d(ax, verts.numpy())
    if show:
        plt.show()

def caculate_align_mat(vec):
    vec = vec / np.linalg.norm(vec)
    z_unit_Arr = np.array([0, 0, 1])

    z_mat = np.array(
        [[0, -z_unit_Arr[2], z_unit_Arr[1]], [z_unit_Arr[2], 0, -z_unit_Arr[0]], [-z_unit_Arr[1], z_unit_Arr[0], 0],]
    )

    z_c_vec = np.matmul(z_mat, vec)
    z_c_vec_mat = np.array([[0, -z_c_vec[2], z_c_vec[1]], [z_c_vec[2], 0, -z_c_vec[0]], [-z_c_vec[1], z_c_vec[0], 0],])

    if np.dot(z_unit_Arr, vec) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, vec) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat, z_c_vec_mat) / (1 + np.dot(z_unit_Arr, vec))

    return qTrans_Mat

def draw_axis(axis, transf, scene, color):
    import trimesh
    import pyrender
    end_points = np.concatenate([axis[np.newaxis] * 20.0, np.zeros((1, 3))], axis=0)
    end_points = np.concatenate([end_points, np.ones((2, 1))], axis=1)
    end_points = (transf @ end_points.T).T[:, :3]

    rot_matrix = np.concatenate([caculate_align_mat(axis), np.zeros((3, 1))], axis=1)
    rot_matrix = np.concatenate([rot_matrix, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)

    cylinder = trimesh.creation.cylinder(radius=0.9, segment=end_points)
    cylinder.visual.vertex_colors = color
    cylinder = pyrender.Mesh.from_trimesh(cylinder, smooth=False)
    scene.add(cylinder)


def display_hand_pyrender(
    mano_blob,
    faces=None,
    batch_idx=0,
    show_axis=True,
    anchors=None,
    bul_axes=None
):
    import trimesh
    import pyrender

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
            anchor_sphere = trimesh.creation.box(extents=(3,3,3))
            anchor_sphere.visual.vertex_colors = np.array([250, 255, 0, 255])
            tfs = np.tile(np.eye(4), (1, 1, 1))
            tfs[0, :3, 3] = anchors[k] * 1000 + dt
            anchor_mesh = pyrender.Mesh.from_trimesh(anchor_sphere, poses=tfs)
            scene.add(anchor_mesh)

    pyrender.Viewer(scene, viewport_size=(1280, 768), use_raymond_lighting=True)



def display_hand_open3d(
    mano_blob,
    faces=None,
    batch_idx=0,
):
    import open3d as o3d
    geometry = o3d.geometry.TriangleMesh()
    geometry.triangles = o3d.utility.Vector3iVector(faces)
    verts, joints = mano_blob.verts[batch_idx], mano_blob.joints[batch_idx]

    geometry.vertices = o3d.utility.Vector3dVector(verts.detach().cpu().numpy())
    geometry.compute_vertex_normals()
    vis = o3d.visualization.VisualizerWithKeyCallback()

    vis.create_window(
        window_name="display",
        width=1024,
        height=768,
    )
    vis.add_geometry(geometry)

    def kill(vis):
        exit(0)

    vis.register_key_callback(ord("Q"), kill)

    while True:
        geometry.compute_vertex_normals()
        vis.update_geometry(geometry)
        vis.update_renderer()
        vis.poll_events()



