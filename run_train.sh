#!/bin/bash

nnUNet_raw="nnUNet_raw"
nnUNet_preprocessed="nnUNet_preprocessed"

export nnUNet_raw=$nnUNet_raw
export nnUNet_preprocessed=$nnUNet_preprocessed

train=0
eval=0
analyze=0
train_dataset_name="Dataset072_GE_LQP9"
model="CMUNeXt-S"
fold=4
data_augmentation=true
num_classes=1
# Evaluation settings
test_datasets=("Dataset072_GE_LQP9" "Dataset073_GE_LE" "Dataset070_Clarius_L15" "Dataset078_KneeUS_OtherDevices")
save_preds=false
largest_component=true
# Analysis defaults
input_channels=3
gpu=0

echo "nnUNet_raw: $nnUNet_raw"
echo "nnUNet_preprocessed: $nnUNet_preprocessed"
echo "train: $train"
echo "eval: $eval"
echo "train_dataset_name: $train_dataset_name"
echo "model: $model"
echo "fold: $fold"
echo "data_augmentation: $data_augmentation"
echo "largest_component: $largest_component"
echo "save_preds: $save_preds"
echo "test_datasets: ${test_datasets[@]}"
echo "test_split: $test_split"
echo "gpu: $gpu"
echo "input_channels: $input_channels"
echo "num_classes: $num_classes"
echo "input_h: $input_h"
echo "input_w: $input_w"

if [[ $train -eq 1 ]]; then
    echo "Training..."
    python main.py \
        --model $model \
        --train_dataset_name $train_dataset_name \
        --fold $fold \
        --num_classes $num_classes \
        --input_channels $input_channels \
        --data_augmentation $data_augmentation
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
            --save_preds $save_preds \
            --data_augmentation $data_augmentation \
            --largest_component $largest_component \
                --num_classes $num_classes \
                --input_channels $input_channels
    done
fi


if [[ $analyze -eq 1 ]]; then
    current_arch=$model
    analyze_input_h=256
    analyze_input_w=256
    analyze_deep_supervision=False
    
    analyze_args="--arch $current_arch --input_channels $input_channels --input_h $analyze_input_h --input_w $analyze_input_w --gpu $gpu"
    
    # Save analysis to model directory if it exists
    model_dir="models/$current_arch"
    if [[ $data_augmentation == true ]]; then
        model_dir="${model_dir}DA"
    fi
    if [[ -d "$model_dir" ]]; then
        analyze_args="$analyze_args --save_path $model_dir/model_analysis.json"
    fi
    
    python analyze_model.py $analyze_args
    
    echo "✓ Completed analysis for $current_arch"
    
    echo ""
    echo "============================================================"
    echo "All models analyzed!"
    echo "============================================================"
fi