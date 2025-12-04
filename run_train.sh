#!/bin/bash

nnUNet_raw="/home/ultrai/UltrAi/knee_us_segmentation/data/nnUNet_raw"
nnUNet_preprocessed="/home/ultrai/UltrAi/knee_us_segmentation/data/nnUNet_preprocessed"

export nnUNet_raw=$nnUNet_raw
export nnUNet_preprocessed=$nnUNet_preprocessed

train=1
eval=0
train_dataset_name="Dataset073_GE_LE"
model="CMUNeXt-S"
fold=0

echo "nnUNet_raw: $nnUNet_raw"
echo "nnUNet_preprocessed: $nnUNet_preprocessed"
echo "train: $train"
echo "eval: $eval"
echo "train_dataset_name: $train_dataset_name"
echo "model: $model"
echo "fold: $fold"

# Evaluation settings
test_datasets=("Dataset073_GE_LE" "Dataset072_GE_LQP9" "Dataset070_Clarius_L15" "Dataset078_KneeUS_OtherDevices")

if [[ $train -eq 1 ]]; then
    echo "Training..."
    python main.py \
        --model $model \
        --train_dataset_name $train_dataset_name \
        --fold $fold
fi