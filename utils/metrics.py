import torch


def _class_index_to_regions(target_lbl, region_label_values):
    region_channels = []
    for values in region_label_values:
        region_mask = torch.zeros_like(target_lbl, dtype=torch.bool)
        for v in values:
            region_mask |= (target_lbl == int(v))
        region_channels.append(region_mask.float())
    return torch.stack(region_channels, dim=1)


def prepare_target_for_loss(target, label_mode, num_classes, region_label_values=None):
    if label_mode == 'multiclass':
        # A one-channel model is binary segmentation and uses BCE/Dice.
        # BCE requires a floating target matching the [B, 1, H, W] logits.
        if num_classes == 1:
            if target.ndim == 3:
                target = target.unsqueeze(1)
            if target.ndim != 4 or target.shape[1] != 1:
                raise ValueError(
                    f"Binary segmentation expects target with shape [B,H,W] or "
                    f"[B,1,H,W], got {tuple(target.shape)}"
                )
            return target.float()

        if target.ndim == 4 and target.shape[1] == num_classes:
            target = torch.argmax(target, dim=1)
        elif target.ndim == 4 and target.shape[1] == 1:
            target = target[:, 0]
        return target.long()

    if target.ndim == 4 and target.shape[1] == num_classes:
        return target.float()
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]

    if target.ndim != 3:
        raise ValueError(f"Expected target with shape [B,H,W] or [B,C,H,W], got {tuple(target.shape)}")
    if not region_label_values:
        raise ValueError("multilabel mode requires region_label_values.")
    if len(region_label_values) != num_classes:
        raise ValueError(f"num_classes ({num_classes}) != region definitions ({len(region_label_values)})")
    return _class_index_to_regions(target.long(), region_label_values)


def _to_bool_masks(output, target, label_mode='multiclass', region_label_values=None):
    if target.ndim == output.ndim - 1:
        target = target.unsqueeze(1)

    c = output.shape[1]
    if label_mode == 'multilabel':
        pred = torch.sigmoid(output) > 0.5
        gt = prepare_target_for_loss(target, 'multilabel', c, region_label_values) > 0.5
        return pred.bool(), gt.bool()

    if c == 1:
        pred = torch.sigmoid(output) > 0.5
        gt = target > 0.5
        return pred.bool(), gt.bool()

    pred_labels = torch.argmax(output, dim=1)
    pred = torch.nn.functional.one_hot(pred_labels, num_classes=c).permute(0, 3, 1, 2).bool()

    if target.ndim == output.ndim and target.shape[1] == c:
        gt_labels = torch.argmax(target, dim=1)
    else:
        gt_labels = target[:, 0].long()
    gt = torch.nn.functional.one_hot(gt_labels, num_classes=c).permute(0, 3, 1, 2).bool()
    return pred, gt


def get_accuracy(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)
    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = 0
    SR = SR > threshold
    GT = GT == torch.max(GT)
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FN = ((SR == 0).byte() + (GT == 1).byte()) == 2
    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)
    return SE


def get_specificity(SR, GT, threshold=0.5):
    SP = 0
    SR = SR > threshold
    GT = GT == torch.max(GT)
    TN = ((SR == 0).byte() + (GT == 0).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    return SP


def get_precision(SR, GT, threshold=0.5):
    PC = 0
    SR = SR > threshold
    GT = GT== torch.max(GT)
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)
    return PC


def iou_score(output, target, label_mode='multiclass', region_label_values=None, ignore_empty=False):
    smooth = 1e-5
    pred, gt = _to_bool_masks(output, target, label_mode=label_mode, region_label_values=region_label_values)

    intersection = (pred & gt).sum(dim=(0, 2, 3)).float()
    union = (pred | gt).sum(dim=(0, 2, 3)).float()
    pred_sum = pred.sum(dim=(0, 2, 3)).float()
    gt_sum = gt.sum(dim=(0, 2, 3)).float()

    # For multi-class tasks, exclude background channel 0 for consistency with eval.
    if label_mode == 'multiclass' and output.shape[1] > 1:
        intersection = intersection[1:]
        union = union[1:]
        pred_sum = pred_sum[1:]
        gt_sum = gt_sum[1:]
        # Keep multiclass behavior fixed: do not ignore empty targets.
        ignore_empty = False

    iou_per_class = (intersection + smooth) / (union + smooth)
    dice_per_class = (2 * intersection + smooth) / (pred_sum + gt_sum + smooth)

    if ignore_empty:
        non_empty = gt_sum > 0
        if non_empty.any():
            iou = iou_per_class[non_empty].mean().item()
            dice = dice_per_class[non_empty].mean().item()
        else:
            iou = 0.0
            dice = 0.0
    else:
        iou = iou_per_class.mean().item()
        dice = dice_per_class.mean().item()

    tp = (pred & gt).sum().float()
    fp = (pred & ~gt).sum().float()
    fn = (~pred & gt).sum().float()
    tn = (~pred & ~gt).sum().float()

    SE = (tp / (tp + fn + 1e-6)).item()
    PC = (tp / (tp + fp + 1e-6)).item()
    SP = (tn / (tn + fp + 1e-6)).item()
    ACC = ((tp + tn) / (tp + tn + fp + fn + 1e-6)).item()
    F1 = (2 * SE * PC / (SE + PC + 1e-6))
    return iou, dice, SE, PC, F1, SP, ACC


def dice_coef(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
        
