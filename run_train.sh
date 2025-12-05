#!/bin/bash

nnUNet_raw="nnUNet_raw"
nnUNet_preprocessed="nnUNet_preprocessed"

export nnUNet_raw=$nnUNet_raw
export nnUNet_preprocessed=$nnUNet_preprocessed

train=0
eval=1
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
save_preds=1

if [[ $train -eq 1 ]]; then
    echo "Training..."
    python main.py \
        --model $model \
        --train_dataset_name $train_dataset_name \
        --fold $fold
fi

if [[ $eval -eq 1 ]]; then
    for test_dataset in ${test_datasets[@]}; do
        echo "Evaluating $test_dataset"
        if [[ $test_dataset == "Dataset078_KneeUS_OtherDevices" ]]; then
            test_split="Ts"
        else
            test_split="Tr"
        fi
        python main.py \
            --model $model \
            --train_dataset_name $train_dataset_name \
            --fold $fold \
            --test_dataset $test_dataset \
            --test_split $test_split \
            --eval 1 \
            --save_preds $save_preds
    done
fi