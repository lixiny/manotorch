import torch
import torch.nn as nn


class AnatomyConstraintLossEE(nn.Module):

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self._setup = False
        self._eps = 1e-6
        self.reduction = reduction

    def setup(
        self,
        thumb_cmc=["+-:45", "+:45,-:15", "+:45,-:0"],
        thumb_mcp=["+-:0", "+-:10", "+:90,-:0"],
        thumb_pip=["+-:0", "+-:0", "+:90,-:0"],
        finger_mcp=["+-:0", "+-:5", "+:90,-:0"],
        finger_pip=["+-:0", "+-:0", "+:90,-:0"],
        finger_dip=["+-:0", "+-:0", "+:90,-:0"],
    ):
        """Setup the angle degree limit for each types of joints.

        Args:
            thumb_cmc (list, optional): The angle limies of thumb CMC joint. 
                Defaults to ["+-:45", "+:45,-:15", "+:45,-:0"], for angles in 
                twist, spread, and bend direction, respectively.
            thumb_mcp (list, optional):  Defaults to ["+-:0", "+-:10", "+:90,-:0"].
            thumb_pip (list, optional):  Defaults to ["+-:0", "+-:0", "+:90,-:0"].
            finger_mcp (list, optional): Defaults to ["+-:0", "+-:5", "+:90,-:0"].
            finger_pip (list, optional): Defaults to ["+-:0", "+-:0", "+:90,-:0"].
            finger_dip (list, optional): Defaults to ["+-:0", "+-:0", "+:90,-:0"].
        """
        self.thumb_cmc = thumb_cmc
        self.thumb_mcp = thumb_mcp
        self.thumb_pip = thumb_pip

        self.finger_mcp = finger_mcp
        self.finger_pip = finger_pip
        self.finger_dip = finger_dip

        self._setup = True

    def _cal_loss_one_axis(self, ee, cfg):
        """Calculate the anatomy loss for one axis.

        Args:
            ee (torch.Tensor): (B, NJ), the euler angle of a certain axis. 
                NJ is number of joints that has this type of axis.
                NJ = 1 for thumb, NJ = 4 for other joints.
            cfg (str): The configuration of the angle limit on this axis.

        Returns:
            torch.Tensor: (B, NJ)
        """
        non_zero_mask = torch.abs(ee) > self._eps

        if "+-" in cfg:
            _, tolerance = cfg.split(":")
            tol = (float(tolerance) / 180.0) * torch.pi
            loss = torch.relu(torch.abs(ee) - tol)
        else:
            pos_cfg, neg_cfg = cfg.split(",")
            pos_tolerance, neg_tolerance = pos_cfg.split(":")[1], neg_cfg.split(":")[1]

            pos_tol = (float(pos_tolerance) / 180.0) * torch.pi
            neg_tol = (float(neg_tolerance) / 180.0) * torch.pi
            neg_mask = ee < 0
            pos_mask = ~neg_mask
            loss = torch.relu(-ee - neg_tol) * neg_mask.float() + \
                  torch.relu(ee - pos_tol) * pos_mask.float()

        loss = loss * non_zero_mask.float()
        return loss

    def _cal_loss_one_joint(self, ee, cfg):
        """Calculate the anatomy loss for one joint.
        as the sum of twist, spread, bend angles.

        Args:
            ee (torch.Tensor): (B, NJ, 3)
            cfg (str): config string for that joint type, e.g. ["+-:0", "+-:0", "+:90,-:0"]

        Returns:
            torch.Tensor: (B, NJ)
        """
        twist_loss = self._cal_loss_one_axis(ee[:, :, 0], cfg[0])
        spread_loss = self._cal_loss_one_axis(ee[:, :, 1], cfg[1])
        bend_loss = self._cal_loss_one_axis(ee[:, :, 2], cfg[2])

        spv = twist_loss + spread_loss + bend_loss
        return spv

    def forward(self, euler_angles, **kwargs):
        """Calculate the anatomy loss for the given euler angles.
        #  the euler-angles' order of the right hand:
        #         15-14-13-\
        #                   \
        #    3-- 2 -- 1 -----0
        #   6 -- 5 -- 4 ----/
        #   12 - 11 - 10 --/
        #    9-- 8 -- 7 --/ 

        Args:
            euler_angles (torch.Tensor): (B, NJ, 3), the euler angles of the joints.

        Raises:
            ValueError: If the setup function is not called before.

        Returns:
            torch.Tensor: a scalar tensor, the anatomy loss.
        """

        if self._setup is False:
            raise ValueError("Please setup the angle limit first.")

        finger_mcp_id = [1, 4, 10, 7]
        finger_pip_id = [2, 5, 11, 8]
        finger_dip_id = [3, 6, 12, 9]

        ee_mcps = euler_angles[:, finger_mcp_id]  # (B, 4, 3)
        ee_pips = euler_angles[:, finger_pip_id]  # (B, 4, 3)
        ee_dips = euler_angles[:, finger_dip_id]  # (B, 4, 3)

        loss_finger_mcps = self._cal_loss_one_joint(ee_mcps, self.finger_mcp)  # (B, 4)
        loss_finger_pips = self._cal_loss_one_joint(ee_pips, self.finger_pip)  # (B, 4)
        loss_finger_dips = self._cal_loss_one_joint(ee_dips, self.finger_dip)  # (B, 4)

        loss_thumb_cmc = self._cal_loss_one_joint(euler_angles[:, 13:14], self.thumb_cmc)  # (B, 1)
        loss_thumb_mcp = self._cal_loss_one_joint(euler_angles[:, 14:15], self.thumb_mcp)  # (B, 1)
        loss_thumb_pip = self._cal_loss_one_joint(euler_angles[:, 15:], self.thumb_pip)  # (B, 1)

        loss_all = torch.cat([
            loss_finger_mcps,
            loss_finger_pips,
            loss_finger_dips,
            loss_thumb_cmc,
            loss_thumb_mcp,
            loss_thumb_pip,
        ],
                             dim=1)  # (B, 15)
        if self.reduction == "none":
            return loss_all
        elif self.reduction == "mean":
            return loss_all.mean()
        elif self.reduction == "sum":
            return loss_all.sum()
        else:
            raise ValueError("Unknown reduction type.")
