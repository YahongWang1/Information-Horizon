#!/bin/bash
CKPT=/llava-v1.5-7b/ # Model weights

attn=flash_attention_2

# Pruning based on token information
size=2
id_prune_layer=3 # the layer to be pruned, ranging from 1 to 31
save_list=(72 144 288 432) 

for save_num in "${save_list[@]}"; 
do

    # MME
    Model="llava-v1.5-7b-info-prune-K$id_prune_layer-save$save_num-size$size"

    dataset=MME
    DATASET=$dataset python -m llava.eval.model_vqa_loader \
        --model-path $CKPT \
        --question-file ./playground/data/eval/MME/llava_mme.jsonl \
        --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
        --answers-file ./playground/data/eval/MME/answers/$Model.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --sparse \
        --attn_implementation $attn \
        --pruned_layer $id_prune_layer \
        --max_num_trunction $save_num 

    cd ./playground/data/eval/MME

    python convert_answer_to_mme.py --experiment $Model

    cd eval_tool

    python calculation.py --results_dir answers/$Model

    cd /Information-Horizon/

done

