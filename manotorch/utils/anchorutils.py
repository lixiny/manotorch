import torch
import numpy as np
import os
import pickle


def anchor_load_driver(inpath):
    """
    return: (face vert index, anchor weight, merged vertex assignment, anchor mapping)
    """
    anchor_root = os.path.join(inpath, "anchor")
    face_vert_idx, anchor_weight, merged_vertex_assignment, anchor_mapping = anchor_load(anchor_root)
    return face_vert_idx, anchor_weight, merged_vertex_assignment, anchor_mapping


def anchor_load(anchor_root):
    """
    this function exists for anchor layer use exclusively
    return: (face vert index, anchor weight, merged vertex assignment, anchor mapping)
    """
    # face vert idx
    face_vert_idx_path = os.path.join(anchor_root, "face_vertex_idx.txt")
    face_vert_idx = np.loadtxt(face_vert_idx_path, dtype=np.int)
    # anchor weight
    anchor_weight_path = os.path.join(anchor_root, "anchor_weight.txt")
    anchor_weight = np.loadtxt(anchor_weight_path)
    # vertex assignment
    vertex_assigned_path = os.path.join(anchor_root, "merged_vertex_assignment.txt")
    merged_vertex_assignment = np.loadtxt(vertex_assigned_path, dtype=np.int)
    # load the anchor mapping
    anchor_mapping_path = os.path.join(anchor_root, "anchor_mapping_path.pkl")
    with open(anchor_mapping_path, "rb") as fstream:
        anchor_mapping = pickle.load(fstream)
    return face_vert_idx, anchor_weight, merged_vertex_assignment, anchor_mapping


def recover_anchor(vertices, idx, weights):
    # weights = ARRAY[22, 2]
    indexed_vertices = vertices[idx]  # ARRAY[22, 3, 3]
    base_vec_1 = indexed_vertices[:, 1, :] - indexed_vertices[:, 0, :]  # ARRAY[22, 3]
    base_vec_2 = indexed_vertices[:, 2, :] - indexed_vertices[:, 0, :]  # ARRAY[22, 3]
    weights_1 = weights[:, 0:1]
    weights_2 = weights[:, 1:2]
    rebuilt_anchors = weights_1 * base_vec_1 + weights_2 * base_vec_2
    origins = indexed_vertices[:, 0, :]
    rebuilt_anchors = rebuilt_anchors + origins
    return rebuilt_anchors


def recover_anchor_batch(vertices, idx, weights):
    # vertices = TENSOR[NBATCH, 778, 3]
    # idx = TENSOR[1, 32, 3]
    # weights = TENSOR[1, 32, 2]
    batch_size = vertices.shape[0]
    batch_idx = torch.arange(batch_size)[:, None, None]  # TENSOR[NBATCH, 1, 1]
    indexed_vertices = vertices[batch_idx, idx, :]  # TENSOR[NBATCH, 32, 3, 3]
    base_vec_1 = indexed_vertices[:, :, 1, :] - indexed_vertices[:, :, 0, :]  # TENSOR[NBATCH, 32, 3]
    base_vec_2 = indexed_vertices[:, :, 2, :] - indexed_vertices[:, :, 0, :]  # TENSOR[NBATCH, 32, 3]
    weights_1 = weights[:, :, 0:1]  # TENSOR[1, 32, 1]
    weights_2 = weights[:, :, 1:2]  # TENSOR[1, 32, 1]
    rebuilt_anchors = weights_1 * base_vec_1 + weights_2 * base_vec_2  # TENSOR[NBATCH, 32, 3]
    origins = indexed_vertices[:, :, 0, :]  # TENSOR[NBATCH, 32, 3]
    rebuilt_anchors = rebuilt_anchors + origins
    return rebuilt_anchors


def region_select_and_mask(vertices, vertex_assignment, hand_palm_vert_idx):
    # there should be 17 diffrent regions
    # select the palm index
    # and get the vertex assignment done right
    selected_vertices = vertices[hand_palm_vert_idx]
    selected_vertices_assignment = vertex_assignment[hand_palm_vert_idx]
    return selected_vertices, selected_vertices_assignment


def get_region_size(vertex_assignment, n_region=None):
    if n_region is None:
        # get n_region
        n_region = len(np.unique(vertex_assignment))
    res = np.zeros((n_region,))
    for region_id in range(n_region):
        res[region_id] = np.sum((vertex_assignment == region_id).astype(np.int))
    return res


def get_region_size_masked_by_palm(vertex_assignment, hand_palm_vert_idx, n_region=None):
    if n_region is None:
        # get n_region
        n_region = len(np.unique(vertex_assignment))
    # first masked vetices & vertex assignment
    selected_vertices_assignment = vertex_assignment[hand_palm_vert_idx]
    res = np.zeros((n_region,))
    for region_id in range(n_region):
        res[region_id] = np.sum((selected_vertices_assignment == region_id).astype(np.int))
    return res


def get_rev_anchor_mapping(anchor_mapping, n_region=None):
    if n_region is None:
        # use mapping to figure out how many regions are there
        n_region = len(set(anchor_mapping.values()))
    res = {region_id: [] for region_id in range(n_region)}
    for anchor_id, region_id in anchor_mapping.items():
        res[region_id].append(anchor_id)
    return res

def get_mask_from_index(mask: np.ndarray, total: int):
    res = np.zeros((total,), dtype=np.int)
    res[mask] = 1
    return res


def get_region_palm_mask(n_region: int, palm, vertex_assignment_merged: np.ndarray, hand_palm_vert_mask: np.ndarray):
    if palm is not None:
        palm_id = 1 if palm else 0
        combined_mask = (vertex_assignment_merged == n_region) & (hand_palm_vert_mask == palm_id)
    else:
        combined_mask = vertex_assignment_merged == n_region
    return combined_mask


def masking_load_driver(anchor_path, palm_vert_idx_path):
    _, _, vertex_assignment_merged, _ = anchor_load(anchor_path)
    hand_palm_vert_idx = np.loadtxt(palm_vert_idx_path, dtype=np.int)
    n_vert = vertex_assignment_merged.shape[0]
    hand_palm_vert_mask = get_mask_from_index(hand_palm_vert_idx, n_vert)
    return vertex_assignment_merged, hand_palm_vert_mask


def test():
    res = anchor_load_driver("./assets/anchor")
    for x in res:
        try:
            print(x.shape)
        except AttributeError:
            print(x)
    print(get_rev_anchor_mapping(res[3]))

# testing
if __name__ == "__main__":
    test()
