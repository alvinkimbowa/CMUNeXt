#!/bin/bash

set -e

source ../UNeXt/.venv/bin/activate

nnUNet_raw="../monounetv2/data/nnUNet_raw"
nnUNet_preprocessed="../monounetv2/data/nnUNet_raw"

export nnUNet_raw=$nnUNet_raw
export nnUNet_preprocessed=$nnUNet_preprocessed

train=${TRAIN:-0}
eval=${EVAL:-1}
analyze=${ANALYZE:-0}
# train_dataset_name="Dataset072_GE_LQP9"
# train_dataset_name="Dataset073_GE_LE"
# train_dataset_name="Dataset070_Clarius_L15"
# train_dataset_name="Dataset080_BUSBRA_GE_Logiq_5"
# train_dataset_name="Dataset081_BUSBRA_GE_Logiq_7"
# train_dataset_name="Dataset082_BUSBRA_Toshiba_Aplio_300"
# train_dataset_name="Dataset083_BUSBRA_U_Systems"
# train_dataset_name="Dataset084_KidneyUS_Philips"
# train_dataset_name="Dataset085_KidneyUS_Other_Devices"
# train_dataset_name="Dataset086_MMOTU_2D"
# train_dataset_name="Dataset087_MMOTU_CEUS"
train_dataset_name="${TRAIN_DATASET_NAME:-Dataset089_Echo_CardiacUDA}"
model="${MODEL:-CMUNeXt-S}"
data_augmentation=${DATA_AUGMENTATION:-false}
num_classes=${NUM_CLASSES:-6}
# Evaluation settings
# test_datasets=("Dataset072_GE_LQP9" "Dataset073_GE_LE" "Dataset070_Clarius_L15") # "Dataset079_KneeUS_Ilker")
# test_datasets=("Dataset080_BUSBRA_GE_Logiq_5" "Dataset081_BUSBRA_GE_Logiq_7" "Dataset082_BUSBRA_Toshiba_Aplio_300" "Dataset083_BUSBRA_U_Systems")
# test_datasets=("Dataset084_KidneyUS_Philips" "Dataset085_KidneyUS_Other_Devices")
# test_datasets=("Dataset086_MMOTU_2D" "Dataset087_MMOTU_CEUS")
# test_datasets=("Dataset090_Echo_EchoCP" "Dataset093_Echo_CardiacNet")
read -r -a test_datasets <<< "${TEST_DATASETS:-Dataset089_Echo_CardiacUDA}"
save_preds=true
largest_component=${LARGEST_COMPONENT:-true}
# Prediction and scoring are separate steps: main.py writes the label maps and
# compute_metrics.py scores them, which is the same scorer monounetv2 and xtinyunet use.
compute_metrics=${COMPUTE_METRICS:-true}
# Analysis defaults
input_channels=${INPUT_CHANNELS:-3}
gpu=${GPU:-1}

export CUDA_VISIBLE_DEVICES=$gpu

read -r -a folds <<< "${FOLDS:-0}"

for fold in "${folds[@]}"; do
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
echo "compute_metrics: $compute_metrics"
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

        model_dir="models/$model"
        if [[ $data_augmentation == true ]]; then
            model_dir="${model_dir}DA"
        fi
        test_dir="$model_dir/$train_dataset_name/fold_${fold}/test/${test_dataset}"
        # The post-processing is baked into the saved predictions, so the filename
        # records it. The test dataset is already the directory, so it is not repeated.
        metrics_csv="image_wise_results.csv"
        if [[ $largest_component == true || $largest_component == True ]]; then
            metrics_csv="image_wise_results_largest_component.csv"
        fi

        if [[ $compute_metrics == true && -d "$test_dir/preds" ]]; then
            python compute_metrics.py \
                --pred_dir "$test_dir/preds" \
                --gt_dir "$nnUNet_raw/${test_dataset}/labels${test_split}" \
                --dataset_json "$nnUNet_raw/${test_dataset}/dataset.json" \
                --results_csv "$test_dir/$metrics_csv" \
                --title "$model / $train_dataset_name -> $test_dataset"
        fi
    done
fi
done

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
