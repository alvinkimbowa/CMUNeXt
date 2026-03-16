import os
import csv
import cv2
import random
import argparse
import json
import nibabel as nib
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader.dataset import MedicalDataSets, CMUNeXt_nnUNetDataset
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, Resize, Flip
from albumentations.augmentations.geometric.transforms import Affine

from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from scipy.ndimage import label
import torch.nn.functional as F

from utils.util import AverageMeter
import utils.losses as losses
from utils.metrics import iou_score, prepare_target_for_loss

from network.CMUNeXt import cmunext, cmunext_s, cmunext_l
from tqdm import tqdm

import matplotlib.pyplot as plt

def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


seed_torch(41)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="CMUNeXt",
                    choices=["CMUNeXt", "CMUNeXt-S", "CMUNeXt-L"], help='model')
parser.add_argument('--train_dataset_name', type=str, default="Dataset073_GE_LE", help='train dataset name')
parser.add_argument('--fold', type=str, default="all", help='fold')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--eval', type=int, default=0, help='eval')
parser.add_argument('--ckpt', type=str, default="checkpoint_best.pth", help='checkpoint')
parser.add_argument('--test_dataset', type=str, default="Dataset073_GE_LE", help='test dataset name')
parser.add_argument('--test_split', type=str, default="Tr", help='test split')
parser.add_argument('--save_preds', type=str2bool, default=False, help='save preds')
parser.add_argument('--overlay', type=str2bool, default=False, help='save overlay visualizations')
parser.add_argument('--data_augmentation', type=str2bool, default=False, help='data augmentation')
parser.add_argument('--largest_component', type=str2bool, default=False, help='largest component')
parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
parser.add_argument('--input_channels', type=int, default=3, help='number of input image channels')
parser.add_argument('--label_mode', type=str, default='multiclass', choices=['multiclass', 'multilabel'], help='label mode')
parser.add_argument('--plot_curves', type=str2bool, default=True, help='plot train/val loss and dice curves')
args = parser.parse_args()


def _load_multilabel_regions(dataset_name):
    nnunet_preprocessed = os.environ['nnUNet_preprocessed']
    nnunet_raw = os.environ.get('nnUNet_raw', None)

    candidates = [os.path.join(f'{nnunet_preprocessed}/{dataset_name}/dataset.json')]
    if nnunet_raw is not None:
        candidates.append(os.path.join(f'{nnunet_raw}/{dataset_name}/dataset.json'))

    dataset_json = None
    for p in candidates:
        if os.path.exists(p):
            dataset_json = p
            break
    if dataset_json is None:
        raise FileNotFoundError(f"Could not find dataset.json in: {candidates}")

    with open(dataset_json, 'r') as f:
        dataset_info = json.load(f)

    labels = dataset_info.get('labels', {})
    regions_class_order = dataset_info.get('regions_class_order', None)

    regions = []
    for k, v in labels.items():
        if str(k).lower() == 'background':
            continue
        if isinstance(v, (list, tuple)):
            regions.append((str(k), tuple(int(x) for x in v)))

    if not regions:
        raise ValueError("multilabel mode requires region-style labels in dataset.json.")

    if regions_class_order is not None:
        by_first = {}
        for name, vals in regions:
            if len(vals) > 0:
                by_first[int(vals[0])] = (name, vals)
        ordered = []
        used = set()
        for c in regions_class_order:
            c = int(c)
            if c in by_first:
                name, vals = by_first[c]
                ordered.append((name, vals))
                used.add(name)
        for name, vals in regions:
            if name not in used:
                ordered.append((name, vals))
        regions = ordered

    return [vals for _, vals in regions]


def _to_finite_scalar(x, default=0.0):
    if isinstance(x, torch.Tensor):
        x = x.detach().float().mean().item()
    x = float(x)
    if np.isnan(x) or np.isinf(x):
        return float(default)
    return x


def plot_training_curves(log_data, output_path):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    epochs = log_data['epoch']
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    line1 = ax1.plot(epochs, log_data['loss'], 'b-', label='Train Loss', alpha=0.7)
    line2 = ax1.plot(epochs, log_data['val_loss'], 'b--', label='Val Loss', alpha=0.7)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, axis='x', alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Dice', color='tab:red')
    line3 = ax2.plot(epochs, log_data['dice'], 'r-', label='Train Dice', alpha=0.7)
    line4 = ax2.plot(epochs, log_data['val_dice'], 'r--', label='Val Dice', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.grid(True, axis='y', alpha=0.3)

    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0, 1.12), ncols=2, frameon=False)

    plt.title('Training Curves')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def getDataloader(args):
    img_size = 256
    if args.data_augmentation:
        train_transform = Compose([
            Affine(rotate=(-15, 15), scale=(0.8, 1.2), p=0.8),
            Resize(img_size, img_size),
            transforms.Normalize(),
        ])
    else:
        train_transform = Compose([
            RandomRotate90(),
            Flip(),
            Resize(img_size, img_size),
            transforms.Normalize(),
        ])

    val_transform = Compose([
        Resize(img_size, img_size),
        transforms.Normalize(),
    ])
    db_train = CMUNeXt_nnUNetDataset(
        dataset_name=args.train_dataset_name,
        split="Tr",
        input_channels=args.input_channels,
        num_classes=args.num_classes,
        fold=args.fold,
        split_type="train",
        transform=train_transform,
    )
    db_val = CMUNeXt_nnUNetDataset(
        dataset_name=args.train_dataset_name,
        split="Tr",
        input_channels=args.input_channels,
        num_classes=args.num_classes,
        fold=args.fold,
        split_type="val",
        transform=val_transform,
    )
    print("train num:{}, val num:{}".format(len(db_train), len(db_val)))

    trainloader = DataLoader(db_train, batch_size=8, shuffle=True,
                             num_workers=8, pin_memory=False)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    return trainloader, valloader


def get_test_dataloader(args):
    img_size = 256
    
    val_transform = Compose([
        Resize(img_size, img_size),
        transforms.Normalize(),
    ])
    if args.test_dataset == args.train_dataset_name and args.test_split == "Tr":
        assert args.fold != 'all', "TODO: Implement validation for all folds"
        db_test = CMUNeXt_nnUNetDataset(
            dataset_name=args.test_dataset,
            split=args.test_split,
            input_channels=args.input_channels,
            num_classes=args.num_classes,
            fold=args.fold,
            split_type='val',
            transform=val_transform,
            eval=False,
            return_full_volume=True)
    else:
        db_test = CMUNeXt_nnUNetDataset(
            dataset_name=args.test_dataset,
            split=args.test_split,
            input_channels=args.input_channels,
            num_classes=args.num_classes,
            transform=val_transform,
            eval=True,
            return_full_volume=True)

    is_volume_eval = db_test.img_ext in ['.nii.gz', '.nii'] and db_test.return_full_volume
    test_batch_size = 1 if is_volume_eval else 8
    test_num_workers = 1 if is_volume_eval else 8
    testloader = DataLoader(db_test, batch_size=test_batch_size, shuffle=False,
                           num_workers=test_num_workers)
    return testloader


def get_model(args):
    if args.model == "CMUNeXt":
        model = cmunext(input_channel=args.input_channels, num_classes=args.num_classes)
    elif args.model == "CMUNeXt-S":
        model = cmunext_s(input_channel=args.input_channels, num_classes=args.num_classes)
    elif args.model == "CMUNeXt-L":
        model = cmunext_l(input_channel=args.input_channels, num_classes=args.num_classes)
    else:
        model = None
        print("model err")
        exit(0)
    return model.cuda()


def visualize_prediction(img, pred, gt=None, dice=None, masd=None, hd95=None, save_path=None):
    """
    Visualize prediction overlaid on input image with optional ground truth and metrics.
    
    Args:
        img: Input image as numpy array (H, W) for grayscale or (H, W, 3) for RGB
        pred: Prediction mask as numpy array (H, W) with values 0 or 1
        gt: Optional ground truth mask as numpy array (H, W) with values 0 or 1
        dice: Optional dice score (0-100)
        masd: Optional MASD score
        hd95: Optional HD95 score
        save_path: Path to save the visualized image
    """
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='gray')
    
    if gt is not None and gt.max() > 0:
        plt.contour(gt, colors='#90EE90', linewidths=1)
    
    # Overlay prediction (red)
    plt.contour(pred, colors='#FF0000', linewidths=1)
    
    # Add metrics text at bottom left
    metrics_text = []
    if dice is not None:
        metrics_text.append(f'Dice: {dice*100:.2f}%')
    if masd is not None:
        metrics_text.append(f'MASD: {masd:.2f}')
    if hd95 is not None:
        metrics_text.append(f'HD95: {hd95:.2f}')
        
    text_str = '\n'.join(metrics_text)
    img_height, img_width = img.shape[:2]
        
    plt.text(img_width * 0.02, img_height * 0.98, text_str,
            fontsize=10, color='white',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='black'))
    
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        print("Saving prediction to: ", save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


def train(args):
    base_lr = args.base_lr
    trainloader, valloader = getDataloader(args)
    model = get_model(args)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    region_label_values = None
    if args.label_mode == 'multilabel':
        region_label_values = _load_multilabel_regions(args.train_dataset_name)
        if len(region_label_values) != args.num_classes:
            raise ValueError(
                f"multilabel mode expects num_classes={len(region_label_values)} "
                f"from dataset.json regions, got {args.num_classes}"
            )
        criterion = losses.__dict__['BCEDiceLoss']().cuda()
    else:
        criterion = losses.__dict__['BCEDiceLoss']().cuda() if args.num_classes == 1 else losses.__dict__['CEDiceLoss']().cuda()
    print("{} iterations per epoch".format(len(trainloader)))
    best_iou = 0
    iter_num = 0
    max_epoch = 300
    max_iterations = len(trainloader) * max_epoch
    fold_str = str(args.fold)
    if args.data_augmentation:
        model_dir = f"models/{args.model}DA/{args.train_dataset_name}/fold_{fold_str}"
    else:
        model_dir = f"models/{args.model}/{args.train_dataset_name}/fold_{fold_str}"
    os.makedirs(model_dir, exist_ok=True)
    last_ckpt_path = f"{model_dir}/checkpoint_last.pth"

    log = {
        'epoch': [],
        'loss': [],
        'val_loss': [],
        'dice': [],
        'val_dice': [],
    }
    start_epoch = 0
    if os.path.exists(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = int(ckpt.get('epoch', -1)) + 1
        best_iou = float(ckpt.get('best_iou', 0.0))
        iter_num = int(ckpt.get('iter_num', start_epoch * len(trainloader)))
        ckpt_log = ckpt.get('log', None)
        if isinstance(ckpt_log, dict):
            for k in log.keys():
                if k in ckpt_log and isinstance(ckpt_log[k], list):
                    log[k] = ckpt_log[k]
        print(f"Resuming training from {last_ckpt_path} at epoch {start_epoch}")

    for epoch_num in range(start_epoch, max_epoch):
        model.train()
        avg_meters = {'loss': AverageMeter(),
                      'iou': AverageMeter(),
                      'dice': AverageMeter(),
                      'val_loss': AverageMeter(),
                      'val_iou': AverageMeter(),
                      'val_dice': AverageMeter(),
                      'SE': AverageMeter(),
                      'PC': AverageMeter(),
                      'F1': AverageMeter(),
                      'ACC': AverageMeter()
                      }
        for i_batch, sampled_batch in tqdm(enumerate(trainloader), total=len(trainloader)):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            label_for_loss = prepare_target_for_loss(
                label_batch,
                args.label_mode,
                args.num_classes,
                region_label_values=region_label_values,
            )

            outputs = model(volume_batch)
            
            loss = criterion(outputs, label_for_loss)
            iou, dice, _, _, _, _, _ = iou_score(
                outputs,
                label_batch,
                label_mode=args.label_mode,
                region_label_values=region_label_values,
            )
            iou = _to_finite_scalar(iou)
            dice = _to_finite_scalar(dice)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            avg_meters['loss'].update(loss.item(), volume_batch.size(0))
            avg_meters['iou'].update(iou, volume_batch.size(0))
            avg_meters['dice'].update(dice, volume_batch.size(0))

        model.eval()
        with torch.no_grad():
            for i_batch, sampled_batch in tqdm(enumerate(valloader), total=len(valloader)):
                input, target = sampled_batch['image'], sampled_batch['label']
                input = input.cuda()
                target = target.cuda()
                target_for_loss = prepare_target_for_loss(
                    target,
                    args.label_mode,
                    args.num_classes,
                    region_label_values=region_label_values,
                )
                output = model(input)
                loss = criterion(output, target_for_loss)
                
                iou, dice, SE, PC, F1, _, ACC = iou_score(
                    output,
                    target,
                    label_mode=args.label_mode,
                    region_label_values=region_label_values,
                )
                iou = _to_finite_scalar(iou)
                dice = _to_finite_scalar(dice)
                SE = _to_finite_scalar(SE)
                PC = _to_finite_scalar(PC)
                F1 = _to_finite_scalar(F1)
                ACC = _to_finite_scalar(ACC)
                avg_meters['val_loss'].update(loss.item(), input.size(0))
                avg_meters['val_iou'].update(iou, input.size(0))
                avg_meters['val_dice'].update(dice, input.size(0))
                avg_meters['SE'].update(SE, input.size(0))
                avg_meters['PC'].update(PC, input.size(0))
                avg_meters['F1'].update(F1, input.size(0))
                avg_meters['ACC'].update(ACC, input.size(0))

        print(
            'epoch [%d/%d]  train_loss : %.4f, train_iou: %.4f, train_dice: %.4f '
            '- val_loss %.4f - val_iou %.4f - val_dice %.4f - val_SE %.4f - val_PC %.4f - val_F1 %.4f - val_ACC %.4f'
            % (epoch_num, max_epoch, avg_meters['loss'].avg, avg_meters['iou'].avg, avg_meters['dice'].avg,
               avg_meters['val_loss'].avg, avg_meters['val_iou'].avg, avg_meters['val_dice'].avg, avg_meters['SE'].avg,
               avg_meters['PC'].avg, avg_meters['F1'].avg, avg_meters['ACC'].avg))
        
        log['epoch'].append(epoch_num)
        log['loss'].append(avg_meters['loss'].avg)
        log['val_loss'].append(avg_meters['val_loss'].avg)
        log['dice'].append(avg_meters['dice'].avg)
        log['val_dice'].append(avg_meters['val_dice'].avg)
        if args.plot_curves:
            plot_training_curves(log, f'{model_dir}/loss_curves.png')

        if avg_meters['val_iou'].avg > best_iou:
            torch.save(model.state_dict(), f'{model_dir}/checkpoint_best.pth')
            best_iou = avg_meters['val_iou'].avg
            print("=> saved best model")

        torch.save(
            {
                'epoch': epoch_num,
                'iter_num': iter_num,
                'best_iou': best_iou,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'log': log,
            },
            last_ckpt_path
        )
    
    torch.save(model.state_dict(), f'{model_dir}/checkpoint_final.pth')
    print("=> saved final model")
    
    return "Training Finished!"

def logits_to_pred_mask(logits, num_classes):
    if num_classes == 1:
        return (torch.sigmoid(logits) > 0.5).float()
    return torch.argmax(logits, dim=1).long()


def find_largest_component_per_class(segmentation, num_classes):
    label_map = segmentation.astype(np.uint8)
    output = np.zeros_like(label_map)
    for i in range(label_map.shape[0]):
        for cls in range(1, num_classes):
            binary_mask = (label_map[i] == cls).astype(np.uint8)
            labeled_array, num_features = label(binary_mask)
            if num_features == 0:
                continue
            largest_label = max(range(1, num_features + 1), key=lambda x: np.sum(labeled_array == x))
            output[i][labeled_array == largest_label] = cls
    return output


def find_largest_component_per_channel(segmentation):
    output = np.zeros_like(segmentation)
    for i in range(segmentation.shape[0]):
        for c in range(segmentation.shape[1]):
            binary_mask = segmentation[i, c].astype(np.uint8)
            labeled_array, num_features = label(binary_mask)
            if num_features == 0:
                continue
            largest_label = max(range(1, num_features + 1), key=lambda x: np.sum(labeled_array == x))
            output[i, c][labeled_array == largest_label] = 1
    return output


def label_to_one_hot(label, num_classes):
    if num_classes == 1:
        if label.ndim == 3:
            label = label.unsqueeze(1)
        return label.float()
    if label.ndim == 4 and label.shape[1] == num_classes:
        return label.float()
    if label.ndim == 4 and label.shape[1] == 1:
        label = label[:, 0]
    return F.one_hot(label.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

def eval(args):
    model = get_model(args)
    if args.data_augmentation:
        model_dir = f"models/{args.model}DA/{args.train_dataset_name}/fold_{args.fold}"
    else:
        model_dir = f"models/{args.model}/{args.train_dataset_name}/fold_{args.fold}"
    
    model.load_state_dict(torch.load(f'{model_dir}/checkpoint_best.pth'))
    valloader = get_test_dataloader(args)
    if len(valloader.dataset) == 0:
        print(
            f"No images found for test_dataset={args.test_dataset}, test_split={args.test_split}. "
            "Skipping evaluation."
        )
        return

    region_label_values = None
    if args.label_mode == 'multilabel':
        region_label_values = _load_multilabel_regions(args.test_dataset)
        if len(region_label_values) != args.num_classes:
            raise ValueError(
                f"multilabel mode expects num_classes={len(region_label_values)} "
                f"from dataset.json regions, got {args.num_classes}"
            )

    include_bg = True if args.label_mode == 'multilabel' else (args.num_classes == 1)
    dice_metric = DiceMetric(
        include_background=include_bg,
        reduction="mean",
        ignore_empty=False,
    )
    hd95_metric = HausdorffDistanceMetric(include_background=include_bg, reduction="mean", percentile=95)
    surface_dice_metric = SurfaceDistanceMetric(include_background=include_bg, reduction="mean")
    model.eval()
        
    if args.save_preds:
        save_dir = os.path.join(model_dir, 'test', args.test_dataset, 'preds')
        os.makedirs(save_dir, exist_ok=True)
        if args.overlay:
            overlay_dir = os.path.join(model_dir, 'test', args.test_dataset, 'overlays')
            os.makedirs(overlay_dir, exist_ok=True)
    
    image_ids = []
    pred_ext = valloader.dataset.img_ext
    overlay_written = False
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(valloader), total=len(valloader)):
            image, label, case = sampled_batch['image'], sampled_batch['label'], sampled_batch['case']
            is_volume = image.ndim == 5
            case_id = case[0] if isinstance(case, (list, tuple)) else case

            if is_volume:
                image_vol = image[0]
                d = image_vol.shape[0]
                logits_chunks = []
                infer_bs = 8
                for start in range(0, d, infer_bs):
                    end = min(start + infer_bs, d)
                    logits_chunks.append(model(image_vol[start:end].cuda()).cpu())
                logits = torch.cat(logits_chunks, dim=0)

                if args.label_mode == 'multilabel':
                    pred_oh_slices = (torch.sigmoid(logits) > 0.5).float()
                    gt_oh_slices = prepare_target_for_loss(
                        label[0].cpu(),
                        'multilabel',
                        args.num_classes,
                        region_label_values=region_label_values,
                    ).float()
                else:
                    pred = logits_to_pred_mask(logits, args.num_classes)
                    gt = label[0].cpu()

                    if args.largest_component:
                        if args.num_classes == 1:
                            pred = (pred > 0.5).float()
                        else:
                            pred_np = pred.numpy()
                            pred = torch.from_numpy(find_largest_component_per_class(pred_np, args.num_classes))

                    pred_oh_slices = label_to_one_hot(pred, args.num_classes)
                    gt_oh_slices = label_to_one_hot(gt, args.num_classes)

                if args.largest_component and args.label_mode == 'multilabel':
                    pred_np = pred_oh_slices.numpy()
                    pred_oh_slices = torch.from_numpy(find_largest_component_per_channel(pred_np)).float()

                pred_oh = pred_oh_slices.permute(1, 2, 3, 0).unsqueeze(0)
                gt_oh = gt_oh_slices.permute(1, 2, 3, 0).unsqueeze(0)
                image_ids.append(case_id)
            else:
                image = image.cuda()
                logits = model(image).cpu()
                if args.label_mode == 'multilabel':
                    pred_oh = (torch.sigmoid(logits) > 0.5).float()
                    gt_oh = prepare_target_for_loss(
                        label.cpu(),
                        'multilabel',
                        args.num_classes,
                        region_label_values=region_label_values,
                    ).float()
                else:
                    pred = logits_to_pred_mask(logits, args.num_classes)
                    gt = label.cpu()

                    if args.largest_component:
                        if args.num_classes == 1:
                            pred = (pred > 0.5).float()
                        else:
                            pred_np = pred.numpy()
                            pred = torch.from_numpy(find_largest_component_per_class(pred_np, args.num_classes))

                    pred_oh = label_to_one_hot(pred, args.num_classes)
                    gt_oh = label_to_one_hot(gt, args.num_classes)

                if args.largest_component and args.label_mode == 'multilabel':
                    pred_np = pred_oh.numpy()
                    pred_oh = torch.from_numpy(find_largest_component_per_channel(pred_np)).float()

                image_ids.extend(case)

            dice = dice_metric(pred_oh, gt_oh)
            hd95 = hd95_metric(pred_oh, gt_oh)
            masd = surface_dice_metric(pred_oh, gt_oh)

            output = pred_oh.numpy()
            gt_np = gt_oh.numpy()
            if args.save_preds:
                if is_volume:
                    affine = np.eye(4, dtype=np.float32)
                    if 'affine' in sampled_batch:
                        affine = sampled_batch['affine'][0].cpu().numpy().astype(np.float32)
                    output_vol = output[0]
                    if args.label_mode == 'multilabel':
                        for c in range(args.num_classes):
                            pred_filename = os.path.join(save_dir, f"{case_id}_c{c}{pred_ext}")
                            nib.save(nib.Nifti1Image(output_vol[c].astype(np.uint8), affine), pred_filename)
                    else:
                        if args.num_classes == 1:
                            label_vol = output_vol[0]
                        else:
                            label_vol = np.argmax(output_vol, axis=0)
                        pred_filename = os.path.join(save_dir, f"{case_id}{pred_ext}")
                        nib.save(nib.Nifti1Image(label_vol.astype(np.uint8), affine), pred_filename)
                else:
                    for i in range(len(output)):
                        for c in range(args.num_classes):
                            if args.overlay:
                                overlay_written = True
                                save_path = os.path.join(overlay_dir, case[i] + '.png')
                                visualize_prediction(
                                    img=image[i, 0, :, :].cpu().numpy(),
                                    gt=gt_np[i, c],
                                    pred=output[i, c],
                                    dice=dice[i].item(),
                                    masd=masd[i].item(),
                                    hd95=hd95[i].item(),
                                    save_path=save_path
                                )
                            else:
                                if args.num_classes > 1:
                                    pred_filename = os.path.join(save_dir, f"{case[i]}_c{c}{pred_ext}")
                                else:
                                    pred_filename = os.path.join(save_dir, f"{case[i]}{pred_ext}")
                                cv2.imwrite(pred_filename, (output[i, c] * 255).astype('uint8'))

        if args.save_preds and args.overlay and overlay_written:
            print("Overlay mode is enabled. Metrics will not be calculated.")
            return

        # Calculate metrics with finite-only reduction to avoid NaN/Inf in reports.
        dice_raw = dice_metric.get_buffer()
        hd95_raw = hd95_metric.get_buffer()
        masd_raw = surface_dice_metric.get_buffer()
        if dice_raw is None or hd95_raw is None or masd_raw is None:
            print("No valid metric buffers were produced. Skipping metric export.")
            return
        dice_buffer = dice_raw.detach().float() * 100
        hd95_buffer = hd95_raw.detach().float()
        masd_buffer = masd_raw.detach().float()

        def _finite_mean_std(buffer):
            x = buffer.reshape(-1)
            finite = torch.isfinite(x)
            if finite.any():
                v = x[finite]
                return v.mean().item(), v.std(unbiased=False).item()
            return 0.0, 0.0

        dice_score, dice_std = _finite_mean_std(dice_buffer)
        hd95_score, hd95_std = _finite_mean_std(hd95_buffer)
        masd_score, masd_std = _finite_mean_std(masd_buffer)

        # Print metrics
        print("\n")
        print(f"Dice: {dice_score:.2f}% ± {dice_std:.2f}%")
        print(f"HD95: {hd95_score:.2f} ± {hd95_std:.2f}")
        print(f"MASD: {masd_score:.2f} ± {masd_std:.2f}")

        # Save to CSV
        results_csv_path = f"{model_dir}/test/results{'_largest_component' if args.largest_component else ''}.csv"
        os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)
        csv_exists = os.path.exists(results_csv_path)
        with open(results_csv_path, 'a', newline='') as csvfile:
            fieldnames = ['test_dataset_name', 'dice', 'dice_std', 'hd95', 'hd95_std', 'masd', 'masd_std']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header if file doesn't exist
            if not csv_exists:
                writer.writeheader()
            
            # Write results
            writer.writerow({
                'test_dataset_name': args.test_dataset,
                'dice': f"{dice_score:.2f}",
                'dice_std': f"{dice_std:.2f}",
                'hd95': f"{hd95_score:.2f}",
                'hd95_std': f"{hd95_std:.2f}",
                'masd': f"{masd_score:.2f}",
                'masd_std': f"{masd_std:.2f}"
            })
        
        print(f"Results saved to: {results_csv_path}\n")

        # Save image-wise metrics to CSV (extract from metric buffers)
        if args.test_dataset == args.train_dataset_name:
            image_wise_csv_path = os.path.join(os.path.dirname(model_dir), f'image_wise_results{"_largest_component" if args.largest_component else ""}_{args.test_dataset}.csv')
        else:
            image_wise_csv_path = os.path.join(model_dir, 'test', f'image_wise_results{"_largest_component" if args.largest_component else ""}_{args.test_dataset}.csv')
        
        os.makedirs(os.path.dirname(image_wise_csv_path), exist_ok=True)
        
        # Per-image metrics from MONAI buffers

        def _buffer_row_to_scalar(buffer, idx):
            row = buffer[idx]
            if isinstance(row, torch.Tensor):
                row = row.detach().float().reshape(-1)
                # For multi-class metrics, store the mean across classes per image.
                finite = torch.isfinite(row)
                if finite.any():
                    return row[finite].mean().item()
                return 0.0
            return float(row)
        
        csv_exists = os.path.exists(image_wise_csv_path) and os.path.getsize(image_wise_csv_path) > 0
        with open(image_wise_csv_path, 'a', newline='') as csvfile:
            fieldnames = ['image_id', 'dice', 'hd95', 'masd']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header only if file doesn't exist or is empty
            if not csv_exists:
                writer.writeheader()
            
            # Write image-wise results
            for i, image_id in enumerate(image_ids):
                writer.writerow({
                    'image_id': image_id,
                    'dice': f"{_buffer_row_to_scalar(dice_buffer, i):.2f}",
                    'hd95': f"{_buffer_row_to_scalar(hd95_buffer, i):.2f}",
                    'masd': f"{_buffer_row_to_scalar(masd_buffer, i):.2f}"
                })
        
        print(f"Image-wise results saved to: {image_wise_csv_path}\n")


if __name__ == "__main__":
    if args.eval == 1:
        eval(args)
    else:
        train(args)
