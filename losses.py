import torch
from torch import nn
import numpy as np
import segmentation_models_pytorch as smp
from clDice.cldice_loss.pytorch.soft_skeleton import soft_skel


class LossMulti:
    def __init__(
        self, jaccard_weight=0, class_weights=None, num_classes=1, device=None
    ):
        self.device = device
        if class_weights is not None:
            nll_weight = torch.from_numpy(class_weights.astype(np.float32)).to(
                self.device
            )
        else:
            nll_weight = None

        self.nll_loss = nn.NLLLoss(weight=nll_weight)
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):

        targets = targets.squeeze(1)
        outputs = outputs.squeeze(1)
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)
        if self.jaccard_weight:
            eps = 1e-7
            for cls in range(self.num_classes):
                jaccard_target = (targets == cls).float()
                jaccard_output = outputs[:, cls].exp()
                intersection = (jaccard_output * jaccard_target).sum()

                union = jaccard_output.sum() + jaccard_target.sum()
                loss -= (
                    torch.log((intersection + eps) / (union - intersection + eps))
                    * self.jaccard_weight
                )

        return loss


class LossUNet:
    def __init__(self, weights=[1, 1, 1]):

        self.criterion = LossMulti(num_classes=2)

    def __call__(self, outputs, targets):

        criterion = self.criterion(outputs, targets)

        return criterion


class LossDCAN:
    def __init__(self, weights=[1, 1, 1]):

        self.criterion1 = LossMulti(num_classes=2)
        self.criterion2 = LossMulti(num_classes=2)
        self.weights = weights

    def __call__(self, outputs1, outputs2, targets1, targets2):

        criterion = self.weights[0] * self.criterion1(
            outputs1, targets1
        ) + self.weights[1] * self.criterion2(outputs2, targets2)

        return criterion


class LossDMTN:
    def __init__(self, weights=[1, 1, 1]):
        self.criterion1 = LossMulti(num_classes=2)
        self.criterion2 = nn.MSELoss()
        self.weights = weights

    def __call__(self, outputs1, outputs2, targets1, targets2):

        criterion = self.weights[0] * self.criterion1(
            outputs1, targets1
        ) + self.weights[1] * self.criterion2(outputs2, targets2)

        return criterion


class LossPsiNet:
    def __init__(self, weights=[1, 1, 1]):   # weights=[1,1,1]

        self.criterion1 = LossMulti(num_classes=2)
        self.criterion2 = LossMulti(num_classes=2)
        self.criterion3 = nn.MSELoss()
        # self.criterion3 = nn.SmoothL1Loss()
        self.weights = weights

    def __call__(self, outputs1, outputs2, outputs3, targets1, targets2, targets3):
        # print(self.weights)

        criterion = (
            self.weights[0] * self.criterion1(outputs1, targets1)
            + self.weights[1] * self.criterion2(outputs2, targets2)
            + self.weights[2] * self.criterion3(outputs3, targets3)
        )

        return criterion


class FAZ_multiLoss:
    def __init__(self, weights=[1, 1, 1]):   # weights=[1,1,1]

        self.criterion1 = smp.losses.DiceLoss(mode="binary")
        self.criterion2 = nn.MSELoss()
        self.criterion3 = nn.MSELoss()
        self.weights = weights

    def __call__(self, outputs1, outputs2, outputs3, targets1, targets2, targets3):
        # print(self.weights)

        criterion = (
            self.weights[0] * self.criterion1(outputs1, targets1)
            + self.weights[1] * self.criterion2(outputs2, targets2)
            + self.weights[2] * self.criterion3(outputs3, targets3)
        )

        return criterion
    
class VesselClDiceLoss(nn.Module):
    """Soft Dice + soft clDice (thesis Eq. 3.14) for a single-channel vessel prediction.

    Takes raw logits and a binary target of shape (B, 1, H, W). The vendored soft_dice_cldice slices
    [:, 1:] on the channel axis (it expects a background channel), which is empty for a single channel and
    silently turned the clDice term into zero.
    """

    def __init__(self, iter_=3, alpha=0.5, smooth=1.0):
        super().__init__()
        self.iter = iter_
        self.alpha = alpha
        self.smooth = smooth
        self.dice = smp.losses.DiceLoss(mode="binary", smooth=1)

    def forward(self, logits, target):
        eps = 1e-7
        prob = torch.sigmoid(logits)
        dice = self.dice(logits, target)
        skel_pred = soft_skel(prob, self.iter)
        skel_true = soft_skel(target, self.iter)
        tprec = (torch.sum(skel_pred * target) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(skel_true * prob) + self.smooth) / (torch.sum(skel_true) + self.smooth)
        cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens).clamp_min(eps)
        return (1.0 - self.alpha) * dice + self.alpha * cl_dice


class Vessel_FAZ_multiLoss:
    def __init__(self, weights=[1, 1, 1], vessel_bce=0.0, exclusion=0.0, aux_channels="both", bg_dice=1.0):
        # vessel_bce / exclusion / aux_channels go beyond the thesis loss (see the tracker):
        #  - vessel_bce: weight of a pixel-wise BCE on the vessel channel. With a dense vessel mask the soft clDice term
        #    is exactly 0 for an all-vessel prediction, so Dice + clDice alone rewards that trivial solution.
        #  - exclusion: weight of mean(p_faz * p_vessel), discouraging vessel predictions inside the FAZ.
        #  - aux_channels: "faz" applies the heatmap / signed-distance MSE to the FAZ channel only.
        self.bg_dice = bg_dice  # the segmentation head has 3 channels [FAZ, vessel, background]; the original loss summed all three
        self.vessel_bce = vessel_bce
        self.exclusion = exclusion
        self.aux_channels = aux_channels
        self.dice = smp.losses.DiceLoss(mode="multilabel")
        self.cl_dice = VesselClDiceLoss()
        self.criterion2 = nn.MSELoss()
        self.criterion3 = nn.MSELoss()
        self.weights = weights

    def __call__(self, outputs1, outputs2, outputs3, targets1, targets2, targets3):

        vessel_cldice = self.cl_dice(outputs1[:, 1:2, ...], targets1[:, 1:2, ...])
        faz_dice = self.dice(outputs1[:, 0:1, ...], targets1[:, 0:1, ...])
        background_dice = self.dice(outputs1[:, 2:3, ...], targets1[:, 2:3, ...])
        # thesis Eq. 3.14 (Dice on the FAZ channel + soft clDice on the vessel channel) plus the background channel
        # that the original code summed in as well, so that the 3-channel argmax is a meaningful class map
        seg_loss = vessel_cldice + faz_dice + self.bg_dice * background_dice
        if self.vessel_bce:
            seg_loss = seg_loss + self.vessel_bce * nn.functional.binary_cross_entropy_with_logits(outputs1[:, 1:2], targets1[:, 1:2])
        if self.exclusion:
            seg_loss = seg_loss + self.exclusion * (torch.sigmoid(outputs1[:, 0:1]) * torch.sigmoid(outputs1[:, 1:2])).mean()
        if self.aux_channels == "faz":
            outputs2, targets2, outputs3, targets3 = outputs2[:, 0:1], targets2[:, 0:1], outputs3[:, 0:1], targets3[:, 0:1]
        multi_loss = (
            self.weights[0] * (seg_loss)
            + self.weights[1] * self.criterion2(outputs2, targets2)
            + self.weights[2] * self.criterion3(outputs3, targets3)
        )

        return multi_loss, seg_loss , vessel_cldice, faz_dice, background_dice


# Lovasz loss
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSoftmax(nn.Module):
    def __init__(self, reduction='mean'):
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction

    def prob_flatten(self, input, target):
        assert input.dim() in [4, 5]
        num_class = input.size(1)
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        elif input.dim() == 5:
            input = input.permute(0, 2, 3, 4, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        target_flatten = target.view(-1)
        return input_flatten, target_flatten

    def lovasz_softmax_flat(self, inputs, targets):
        num_classes = inputs.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (targets == c).float()
            if num_classes == 1:
                input_c = inputs[:, 0]
            else:
                input_c = inputs[:, c]
            loss_c = (torch.autograd.Variable(target_c) - input_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(torch.dot(loss_c_sorted, torch.autograd.Variable(lovasz_grad(target_c_sorted))))
        losses = torch.stack(losses)

        if self.reduction == 'none':
            loss = losses
        elif self.reduction == 'sum':
            loss = losses.sum()
        else:
            loss = losses.mean()
        return loss

    def forward(self, inputs, targets):
        # print(inputs.shape, targets.shape) # (batch size, class_num, x,y,z), (batch size, 1, x,y,z)
        inputs, targets = self.prob_flatten(inputs, targets)
        # print(inputs.shape, targets.shape)
        losses = self.lovasz_softmax_flat(inputs, targets)
        return losses
