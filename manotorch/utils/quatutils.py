# quat format w,x,y,z
import torch
import torch.nn.functional as torch_f


def normalize_quaternion(quaternion: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    r"""Normalizes a quaternion.
    The quaternion should be in (w, x, y, z) format.

    Args:
        quaternion (torch.Tensor): a tensor containing a quaternion to be
          normalized. The tensor can be of shape :math:`(*, 4)`.
        eps (Optional[bool]): small value to avoid division by zero.
          Default: 1e-12.

    Return:
        torch.Tensor: the normalized quaternion of shape :math:`(*, 4)`.

    Example:
        >>> quaternion = torch.tensor([0., 1., 0., 1.])
        >>> normalize_quaternion(quaternion)
        tensor([0.000, 0.7071, 0.0000, 0.7071])
    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape (*, 4). Got {}".format(quaternion.shape))
    return torch_f.normalize(quaternion, p=2, dim=-1, eps=eps)


def quaternion_inv(q):
    """
    inverse quaternion(s) q
    The quaternion should be in (w, x, y, z) format.
    Expects  tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4

    q_conj = q[..., 1:] * -1.0
    q_conj = torch.cat((q[..., 0:1], q_conj), dim=-1)
    q_norm = torch.norm(q, dim=-1, keepdim=True)
    return q_conj / q_norm


def quaternion_mul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    The quaternion should be in (w, x, y, z) format.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    # terms; ( * , 4, 4)
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] + terms[:, 2, 3] - terms[:, 3, 2]
    y = terms[:, 0, 2] - terms[:, 1, 3] + terms[:, 2, 0] + terms[:, 3, 1]
    z = terms[:, 0, 3] + terms[:, 1, 2] - terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def angle_axis_to_quaternion(angle_axis: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.
    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if not torch.is_tensor(angle_axis):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(angle_axis)))

    if not angle_axis.shape[-1] == 3:
        raise ValueError("Input must be a tensor of shape Nx3 or 3. Got {}".format(angle_axis.shape))

    angles = torch.norm(angle_axis, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48
    quaternions = torch.cat([torch.cos(half_angles), angle_axis * sin_half_angles_over_angles], dim=-1)
    return quaternions


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """

    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}".format(quaternion.shape))

    norms = torch.norm(quaternion[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternion[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48
    return quaternion[..., 1:] / sin_half_angles_over_angles


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    r"""Convert a quaternion to a rotation matrix.
    The quaternion vector has components in (w, x, y, z) format.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternion.

    Return:
        torch.Tensor: tensor with quaternion.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3, 3)`

    Example:
        >>> q = torch.rand(2, 4)  # Nx4
        >>> rotmat = quaternion_to_rotation_matrix(q)  # Nx3x3
    """
    original_shape = quaternion.shape  # (*, 4)
    asterisk_shape = original_shape[:-1]  # (*, )
    # split cols of q
    w, x, y, z = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    # convenient terms
    ww, wx, wy, wz = w * w, w * x, w * y, w * z
    xx, xy, xz = x * x, x * y, x * z
    yy, yz = y * y, y * z
    zz = z * z
    # compute normalizer
    q_norm_squared = ww + xx + yy + zz  # (*, )
    q_norm_squared = q_norm_squared.unsqueeze(-1)  # (*, 1) for broadcasting
    # stack
    rotation_matrix = torch.stack(
        (
            ww + xx - yy - zz,  # (0, 0)
            2 * (xy - wz),  # (0, 1)
            2 * (wy + xz),  # (0, 2)
            2 * (wz + xy),  # (1, 0)
            ww - xx + yy - zz,  # (1, 1)
            2 * (yz - wx),  # (1, 2)
            2 * (xz - wy),  # (2, 0)
            2 * (wx + yz),  # (2, 1)
            ww - xx - yy + zz,  # (2, 2)
        ),
        dim=-1,
    )  # (*, 9)
    # normalize
    rotation_matrix = rotation_matrix / q_norm_squared  # (*, 9)
    # reshape
    target_shape = tuple(list(asterisk_shape) + [3, 3])  # value = (*, 3, 3)
    rotation_matrix = rotation_matrix.reshape(target_shape)

    return rotation_matrix


def quaternion_norm(quaternion):
    r"""Computes norm of quaternion.
    The quaternion should be in (w, x, y, z) format.

    Args:
        quaternion (torch.Tensor): a tensor containing a quaternion to be
          normalized. The tensor can be of shape :math:`(*, 4)`.

    Return:
        torch.Tensor: the norm of shape :math:`(*)`.

    Example:
        >>> quaternion = torch.tensor([0., 1., 0., 1.])
        >>> quaternion_norm(quaternion)
        tensor(1.4142)
    """
    return torch.sqrt(torch.sum(torch.pow(quaternion, 2), dim=-1))


def quaternion_norm_squared(quaternion):
    r"""Computes norm of quaternion.
    The quaternion should be in (w, x, y, z) format.

    Args:
        quaternion (torch.Tensor): a tensor containing a quaternion to be
          normalized. The tensor can be of shape :math:`(*, 4)`.

    Return:
        torch.Tensor: the squared norm of shape :math:`(*)`.

    Example:
        >>> quaternion = torch.tensor([0., 1., 0., 1.])
        >>> quaternion_norm(quaternion)
        tensor(2.0)
    """
    return torch.sum(torch.pow(quaternion, 2), dim=-1)
