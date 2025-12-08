import os
import csv
import cv2
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader.dataset import MedicalDataSets, CMUNeXt_nnUNetDataset
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, Resize, Flip

from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from scipy.ndimage import label

from utils.util import AverageMeter
import utils.losses as losses
from utils.metrics import iou_score

from network.CMUNeXt import cmunext, cmunext_s, cmunext_l
from tqdm import tqdm



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
parser.add_argument('--test_dataset', type=str, default="Dataset073_GE_LE", help='test dataset name')
parser.add_argument('--test_split', type=str, default="Tr", help='test split')
parser.add_argument('--save_preds', type=int, default=0, help='save preds')
args = parser.parse_args()


def getDataloader(args):
    img_size = 256
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
        fold=args.fold,
        split_type="train",
        transform=train_transform,
    )
    db_val = CMUNeXt_nnUNetDataset(
        dataset_name=args.train_dataset_name,
        split="Tr",
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
    if args.test_dataset == args.train_dataset_name:
        assert args.fold != 'all', "TODO: Implement validation for all folds"
        db_test = CMUNeXt_nnUNetDataset(
            dataset_name=args.test_dataset,
            split=args.test_split,
            fold=args.fold,
            split_type='val',
            transform=val_transform,
            eval=False)
    else:
        db_test = CMUNeXt_nnUNetDataset(
            dataset_name=args.test_dataset,
            split=args.test_split,
            transform=val_transform,
            eval=True)
    
    testloader = DataLoader(db_test, batch_size=8, shuffle=False,
                           num_workers=8)
    return testloader


def get_model(args):
    if args.model == "CMUNeXt":
        model = cmunext()
    elif args.model == "CMUNeXt-S":
        model = cmunext_s()
    elif args.model == "CMUNeXt-L":
        model = cmunext_l()
    else:
        model = None
        print("model err")
        exit(0)
    return model.cuda()


def train(args):
    base_lr = args.base_lr
    trainloader, valloader = getDataloader(args)
    model = get_model(args)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    criterion = losses.__dict__['BCEDiceLoss']().cuda()
    print("{} iterations per epoch".format(len(trainloader)))
    best_iou = 0
    iter_num = 0
    max_epoch = 300
    max_iterations = len(trainloader) * max_epoch
    for epoch_num in range(max_epoch):
        model.train()
        avg_meters = {'loss': AverageMeter(),
                      'iou': AverageMeter(),
                      'val_loss': AverageMeter(),
                      'val_iou': AverageMeter(),
                      'SE': AverageMeter(),
                      'PC': AverageMeter(),
                      'F1': AverageMeter(),
                      'ACC': AverageMeter()
                      }
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs = model(volume_batch)
            
            loss = criterion(outputs, label_batch)
            iou, dice, _, _, _, _, _ = iou_score(outputs, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            avg_meters['loss'].update(loss.item(), volume_batch.size(0))
            avg_meters['iou'].update(iou, volume_batch.size(0))

        model.eval()
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(valloader):
                input, target = sampled_batch['image'], sampled_batch['label']
                input = input.cuda()
                target = target.cuda()
                output = model(input)
                loss = criterion(output, target)
                
                iou, _, SE, PC, F1, _, ACC = iou_score(output, target)
                avg_meters['val_loss'].update(loss.item(), input.size(0))
                avg_meters['val_iou'].update(iou, input.size(0))
                avg_meters['SE'].update(SE, input.size(0))
                avg_meters['PC'].update(PC, input.size(0))
                avg_meters['F1'].update(F1, input.size(0))
                avg_meters['ACC'].update(ACC, input.size(0))

        print(
            'epoch [%d/%d]  train_loss : %.4f, train_iou: %.4f '
            '- val_loss %.4f - val_iou %.4f - val_SE %.4f - val_PC %.4f - val_F1 %.4f - val_ACC %.4f'
            % (epoch_num, max_epoch, avg_meters['loss'].avg, avg_meters['iou'].avg,
               avg_meters['val_loss'].avg, avg_meters['val_iou'].avg, avg_meters['SE'].avg,
               avg_meters['PC'].avg, avg_meters['F1'].avg, avg_meters['ACC'].avg))
        
        fold_str = str(args.fold)
        model_dir = f"models/{args.model}/{args.train_dataset_name}/fold_{fold_str}"
        os.makedirs(model_dir, exist_ok=True)

        if avg_meters['val_iou'].avg > best_iou:
            torch.save(model.state_dict(), f'{model_dir}/checkpoint_best.pth')
            best_iou = avg_meters['val_iou'].avg
            print("=> saved best model")
    
    torch.save(model.state_dict(), f'{model_dir}/checkpoint_final.pth')
    print("=> saved final model")
    
    return "Training Finished!"


def eval(args):
    model = get_model(args)
    model_dir = f"models/{args.model}/{args.train_dataset_name}/fold_{args.fold}"
    model.load_state_dict(torch.load(f'{model_dir}/checkpoint_best.pth'))
    valloader = get_test_dataloader(args)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=95)
    surface_dice_metric = SurfaceDistanceMetric(include_background=False, reduction="mean")
    model.eval()
        
    if args.save_preds:
        save_dir = os.path.join(model_dir, 'test', args.test_dataset, 'preds')
        os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(valloader), total=len(valloader)):
            image, label = sampled_batch['image'], sampled_batch['label']
            image = image.cuda()
            output = model(image).cpu()
            output = torch.sigmoid(output)
            output = (output > 0.5).float()

            dice_metric(output, label)
            hd95_metric(output, label)
            surface_dice_metric(output, label)

            output = output.numpy()
            if args.save_preds:
                for i in range(len(output)):
                    cv2.imwrite(os.path.join(save_dir, sampled_batch['case'][i] + '.png'),
                            (output[i][0] * 255).astype('uint8'))

        # Calculate metrics
        dice_score = dice_metric.aggregate().item() * 100
        dice_std = dice_metric.get_buffer().std().item() * 100
        hd95_score = hd95_metric.aggregate().item()
        hd95_std = hd95_metric.get_buffer().std().item()
        masd_score = surface_dice_metric.aggregate().item()
        masd_std = surface_dice_metric.get_buffer().std().item()

        # Print metrics
        print("\n")
        print(f"Dice: {dice_score:.2f}% ± {dice_std:.2f}%")
        print(f"HD95: {hd95_score:.2f} ± {hd95_std:.2f}")
        print(f"MASD: {masd_score:.2f} ± {masd_std:.2f}")

        # Save to CSV
        results_csv_path = f"{model_dir}/test/results.csv"
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


if __name__ == "__main__":
    if args.eval == 1:
        eval(args)
    else:
        train(args)
