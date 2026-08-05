#!/bin/bash

source ../UNeXt/.venv/bin/activate

nnUNet_raw="../mononunetv2/data/nnUNet_raw"
nnUNet_preprocessed="../mononunetv2/data/nnUNet_preprocessed"

export nnUNet_raw=$nnUNet_raw
export nnUNet_preprocessed=$nnUNet_preprocessed

train=1
eval=1
analyze=0
# train_dataset_name="Dataset072_GE_LQP9"
# train_dataset_name="Dataset073_GE_LE"
# train_dataset_name="Dataset070_Clarius_L15"
train_dataset_name="Dataset080_BUSBRA_GE_Logiq_5"
# train_dataset_name="Dataset081_BUSBRA_GE_Logiq_7"
# train_dataset_name="Dataset082_BUSBRA_Toshiba_Aplio_300"
# train_dataset_name="Dataset083_BUSBRA_U_Systems"
# train_dataset_name="Dataset084_KidneyUS_Philips"
# train_dataset_name="Dataset085_KidneyUS_Other_Devices"
# train_dataset_name="Dataset086_MMOTU_2D"
# train_dataset_name="Dataset087_MMOTU_CEUS"
model="CMUNeXt-S"
data_augmentation=false
num_classes=1
# Evaluation settings
# test_datasets=("Dataset072_GE_LQP9" "Dataset073_GE_LE" "Dataset070_Clarius_L15") # "Dataset079_KneeUS_Ilker")
test_datasets=("Dataset080_BUSBRA_GE_Logiq_5" "Dataset081_BUSBRA_GE_Logiq_7" "Dataset082_BUSBRA_Toshiba_Aplio_300" "Dataset083_BUSBRA_U_Systems")
# test_datasets=("Dataset084_KidneyUS_Philips" "Dataset085_KidneyUS_Other_Devices")
# test_datasets=("Dataset086_MMOTU_2D" "Dataset087_MMOTU_CEUS")
save_preds=true
largest_component=true
# Analysis defaults
input_channels=3
gpu=1

export CUDA_VISIBLE_DEVICES=$gpu

for fold in {0..0}; do
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
        if [[ $test_dataset == "Dataset078_KneeUS_OtherDevices" || $test_dataset == "Dataset079_KneeUS_Ilker" ]]; then
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
done

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
        analyze_args="$analyze_args --save_path $model_dir/$train_dataset_name/model_analysis.json"
    fi
    
    python analyze_model.py $analyze_args
    
    echo "✓ Completed analysis for $current_arch"
    
    echo ""
    echo "============================================================"
    echo "All models analyzed!"
    echo "============================================================"
fi
