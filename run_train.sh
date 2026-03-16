#!/bin/bash

source /home/ultrai/UltrAi/UNeXt/.venv/bin/activate

nnUNet_raw="/home/ultrai/UltrAi/nntinyunet/data/nnUNet_raw"
nnUNet_preprocessed="/home/ultrai/UltrAi/nntinyunet/data/nnUNet_preprocessed"

export nnUNet_raw=$nnUNet_raw
export nnUNet_preprocessed=$nnUNet_preprocessed

train=0
eval=1
analyze=0
ckpt="checkpoint_best.pth"
train_dataset_name="Dataset072_GE_LQP9"
# train_dataset_name="Dataset073_GE_LE"
# train_dataset_name="Dataset070_Clarius_L15"

# train_dataset_name="Dataset301_busbra"
# train_dataset_name="Dataset300_isic2018"
# train_dataset_name="Dataset302_EchoNet-Dynamic"
# train_dataset_name="Dataset309_FIVES"
# train_dataset_name="Dataset306_ACDC"
# train_dataset_name="Dataset137_BraTS2021"

model="CMUNeXt-S"
fold=0
data_augmentation=false
label_mode="multiclass"
num_classes=1
# Evaluation settings
# test_datasets=("Dataset072_GE_LQP9" "Dataset073_GE_LE" "Dataset070_Clarius_L15" "Dataset078_KneeUS_OtherDevices" "Dataset079_KneeUS_Ilker")
# test_datasets=("Dataset070_Clarius_L15" "Dataset079_KneeUS_Ilker")
# test_datasets=("Dataset073_GE_LE")
# test_datasets=("Dataset079_KneeUS_Ilker")
test_datasets=($train_dataset_name)

save_preds=true
overlay=false
largest_component=false
# Analysis defaults
input_channels=3
gpu=0

# Dataset-specific overrides
if [[ $train_dataset_name == "Dataset137_BraTS2021" ]]; then
    input_channels=4
    num_classes=3
    label_mode="multilabel"
elif [[ $train_dataset_name == "Dataset306_ACDC" ]]; then
    input_channels=1
    num_classes=4
fi

export CUDA_VISIBLE_DEVICES=$gpu
export NO_ALBUMENTATIONS_UPDATE=1

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
echo "overlay: $overlay"
echo "test_datasets: ${test_datasets[@]}"
echo "test_split: $test_split"
echo "gpu: $gpu"
echo "input_channels: $input_channels"
echo "num_classes: $num_classes"
echo "label_mode: $label_mode"
echo "input_h: $input_h"
echo "input_w: $input_w"

if [[ $train -eq 1 ]]; then
    echo "Training..."
    python main.py \
        --model $model \
        --train_dataset_name $train_dataset_name \
        --fold $fold \
        --num_classes $num_classes \
        --label_mode $label_mode \
        --input_channels $input_channels \
        --data_augmentation $data_augmentation
fi

for fold in {0..0}; do
    if [[ $eval -eq 1 ]]; then
        for test_dataset in ${test_datasets[@]}; do
            # echo "Evaluating $test_dataset"
            # if [[ $test_dataset == "Dataset078_KneeUS_OtherDevices" || $test_dataset == "Dataset079_KneeUS_Ilker" ]]; then
            #     test_split="Ts"
            # else
            #     test_split="Tr"
            # fi
            test_split="Ts"
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
                --label_mode $label_mode \
                --input_channels $input_channels \
                --overlay $overlay \
                --ckpt $ckpt
        done
    fi


    if [[ $analyze -eq 1 ]]; then
        current_arch=$model
        analyze_input_h=256
        analyze_input_w=256
        analyze_deep_supervision=False
        
        analyze_args="--arch $current_arch --input_channels $input_channels --num_classes $num_classes --input_h $analyze_input_h --input_w $analyze_input_w --gpu $gpu"
        
        # Save analysis to model directory if it exists
        model_dir="models/$current_arch"
        if [[ $data_augmentation == true ]]; then
            model_dir="${model_dir}DA"
        fi
        if [[ -d "$model_dir" ]]; then
            analyze_args="$analyze_args --save_path $model_dir/$train_dataset_name/model_analysis.json"
        fi
        
        python analyze_model.py $analyze_args
        
        echo "✓ Completed analysis for $current_arch"
        
        echo ""
        echo "============================================================"
        echo "All models analyzed!"
        echo "============================================================"
    fi
done
