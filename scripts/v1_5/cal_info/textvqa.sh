#!/bin/bash
CKPT=/llava-v1.5-7b/ # Model weights
MODEL=llava-v1.5-7b

attn=flash_attention_2

layer_list=($(seq 1 31))

# LLaVA-1.5 encodes each image into a 24x24=576 visual token grid
# size: window size for computing token information, i.e., size*size tokens per group
# x_list, y_list: window coordinates on the 24x24 grid, ranging from 1 to 24/size
grid=24
size=2
num_windows=$((grid / size))

x_list=($(seq 1 $num_windows))
y_list=($(seq 1 $num_windows))




# # Calculate the information of text tokens
for id_prune_layer in ${layer_list[@]}; do
    # TextVQA
    dataset=TextVQA
    DATASET=$dataset LAYER_INDEX=$id_prune_layer SIZE=$size python -m llava.eval.model_vqa_loader \
        --model-path $CKPT \
        --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
        --image-folder ./playground/data/eval/textvqa/train_images \
        --answers-file /$dataset/answers/$MODEL/save_answers/size_$size/layer_$id_prune_layer/systext.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --sparse \
        --attn_implementation $attn \
        --pruned_layer $id_prune_layer 
done

# Calculate the information of each visual token

for id_prune_layer in ${layer_list[@]}; do
    for x in ${x_list[@]}; do
        for y in ${y_list[@]}; do
            # TextVQA
            dataset=TextVQA
            DATASET=$dataset LAYER_INDEX=$id_prune_layer SIZE=$size X=$x  Y=$y  python -m llava.eval.model_vqa_loader \
                --model-path $CKPT \
                --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
                --image-folder ./playground/data/eval/textvqa/train_images \
                --answers-file /$dataset/answers/$MODEL/save_answers/size_$size/layer_$id_prune_layer/x_$x/y_$y.jsonl \
                --temperature 0 \
                --conv-mode vicuna_v1 \
                --sparse \
                --attn_implementation $attn \
                --pruned_layer $id_prune_layer
        done
    done
done


