import numpy as np
import torch
from torch.nn import Module


class UpSampleLayer(Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def calculate_faces(faces, vn):
        edges = {}
        new_faces = []

        def get_edge_id(e):
            if e not in edges:
                edges[e] = len(edges)
            return edges[e]

        for f in faces:
            a, b, c = f[0], f[1], f[2]
            e1, e2, e3 = tuple(sorted([a, b])), tuple(sorted([b, c])), tuple(sorted([c, a]))
            x = get_edge_id(e1) + vn
            y = get_edge_id(e2) + vn
            z = get_edge_id(e3) + vn
            new_faces.append(np.array([x, y, z]))
            new_faces.append(np.array([a, x, z]))
            new_faces.append(np.array([b, y, x]))
            new_faces.append(np.array([c, z, y]))

        new_faces = np.vstack(new_faces)
        new_vertices_idx = np.vstack([np.array(list(k)) for k in edges.keys()])
        return new_vertices_idx, new_faces

    def forward(self, vertices, faces):
        """
            *
           / \
          /   \
         /     \
        * ----- *
            |
            *
           / \
          o - o
         / \ / \
        * --o-- *
        """
        device = vertices.device
        new_vertices_idx_list, new_faces_list = [], []
        for i, fs in enumerate(faces):
            new_vertices_idx, new_faces = self.calculate_faces(fs.detach().cpu().numpy(), len(vertices[i]))
            new_vertices_idx_list.append(np.expand_dims(new_vertices_idx, axis=0))
            new_faces_list.append(np.expand_dims(new_faces, axis=0))
        new_vertices_idx_list = torch.from_numpy(np.vstack(new_vertices_idx_list)).long().to(device)
        new_faces_list = torch.from_numpy(np.vstack(new_faces_list)).long().to(device)

        expand_vertices = vertices.unsqueeze(1).expand(-1, new_vertices_idx_list.shape[1], -1, -1)
        expand_vertices_idx = new_vertices_idx_list.unsqueeze(-1).expand(-1, -1, -1, 3)
        new_verts = torch.mean(torch.gather(expand_vertices, 2, expand_vertices_idx), dim=-2)
        new_verts = torch.cat([vertices, new_verts], dim=1)
        return new_verts, new_faces_list
