#!/bin/bash
CKPT=/llava-v1.5-7b/ # Model weights

attn=flash_attention_2

# Pruning based on token information
size=2
id_prune_layer=3 # the layer to be pruned, ranging from 1 to 31
save_list=(72 144 288 432)

for save_num in "${save_list[@]}"; 
do
    # TextVQA
    Model="llava-v1.5-7b-info-prune-K$id_prune_layer-save$save_num-size$size"

    dataset=TextVQA
    DATASET=$dataset python -m llava.eval.model_vqa_loader \
        --model-path $CKPT \
        --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
        --image-folder ./playground/data/eval/textvqa/train_images \
        --answers-file ./playground/data/eval/textvqa/answers/$Model.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --sparse \
        --attn_implementation $attn \
        --pruned_layer $id_prune_layer \
        --max_num_trunction $save_num 

    python -m llava.eval.eval_textvqa \
        --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
        --result-file ./playground/data/eval/textvqa/answers/$Model.jsonl

done