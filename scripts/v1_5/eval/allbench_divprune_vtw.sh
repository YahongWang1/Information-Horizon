#!/bin/bash
# DivPrune+VTW 两阶段剪枝实验

CKPT=/llava-v1.5-7b/  
baseModel=llava-v1.5-7b

attn=flash_attention_2

# 两阶段剪枝配置：
# 格式：divprune_layer divprune_save vtw_layer vtw_save avg_tkn
# 192配置：layer0保留284，layer20随机保留0
# 64配置：layer0保留98，layer21随机保留0
configs=(
    "0 284 20 0 192"
    "0 98 21 0 64"
)

seq_num=1 # 重复次数

for i in $(seq 1 $seq_num); do
    for config in "${configs[@]}"; do
        # 解析配置：divprune_layer divprune_save vtw_layer vtw_save avg_tkn
        read -r divprune_layer divprune_save vtw_layer vtw_save avg_tkn <<< "$config"

        Model="$baseModel-DivPrune+VTW-avg_tkn${avg_tkn}-DivPrune-K${divprune_layer}-save${divprune_save}-VTW-K${vtw_layer}-save${vtw_save}-$i"

        export SAVE_NUM_LIST="[$divprune_save, $vtw_save]"
        export PRUNED_LAYER_LIST="[$divprune_layer, $vtw_layer]"
        export METHOD_PRE=DivPrune
        export PRUNE_STG=VTW

        # MME
        echo "DivPrune Layer$divprune_layer save$divprune_save -> VTW Layer$vtw_layer save$vtw_save"

        python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file ./playground/data/eval/MME/llava_mme.jsonl \
            --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
            --answers-file ./playground/data/eval/MME/answers/$Model.jsonl \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --sparse \
            --attn_implementation $attn \
            --pruned_layer -1 \
            --image_token_start_index 35 \
            --image_token_length 576 \
            --pivot_image_token 4 \
            --pivot_text_token 4 \
            --max_num_trunction 576

        cd ./playground/data/eval/MME
        python convert_answer_to_mme.py --experiment $Model
        cd eval_tool
        python calculation.py --results_dir answers/$Model
        cd /Information-Horizon


        # SQA
        echo "DivPrune Layer$divprune_layer save$divprune_save -> VTW Layer$vtw_layer save$vtw_save"

        python -m llava.eval.model_vqa_science \
            --model-path $CKPT \
            --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
            --image-folder ./playground/data/eval/scienceqa/images/test \
            --answers-file ./playground/data/eval/scienceqa/answers/$Model.jsonl \
            --single-pred-prompt \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --sparse \
            --attn_implementation $attn \
            --pruned_layer -1 \
            --image_token_start_index 35 \
            --image_token_length 576 \
            --pivot_image_token 4 \
            --pivot_text_token 4 \
            --max_num_trunction 576

        python llava/eval/eval_science_qa.py \
            --base-dir ./playground/data/eval/scienceqa \
            --result-file ./playground/data/eval/scienceqa/answers/$Model.jsonl \
            --output-file ./playground/data/eval/scienceqa/answers/${Model}_output.jsonl \
            --output-result ./playground/data/eval/scienceqa/answers/${Model}_result.json


        # POPE
        echo "DivPrune Layer$divprune_layer save$divprune_save -> VTW Layer$vtw_layer save$vtw_save"

        python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
            --image-folder ./playground/data/eval/pope/val2014 \
            --answers-file ./playground/data/eval/pope/answers/$Model.jsonl \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --sparse \
            --attn_implementation $attn \
            --pruned_layer -1 \
            --image_token_start_index 35 \
            --image_token_length 576 \
            --pivot_image_token 4 \
            --pivot_text_token 4 \
            --max_num_trunction 576

        python llava/eval/eval_pope.py \
            --annotation-dir ./playground/data/eval/pope/coco \
            --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
            --result-file ./playground/data/eval/pope/answers/$Model.jsonl


        # TextVQA
        echo "DivPrune Layer$divprune_layer save$divprune_save -> VTW Layer$vtw_layer save$vtw_save"

        python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
            --image-folder ./playground/data/eval/textvqa/train_images \
            --answers-file ./playground/data/eval/textvqa/answers/$Model.jsonl \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --sparse \
            --attn_implementation $attn \
            --pruned_layer -1 \
            --image_token_start_index 35 \
            --image_token_length 576 \
            --is_textvqa \
            --pivot_image_token 4 \
            --pivot_text_token 4 \
            --max_num_trunction 576

        python -m llava.eval.eval_textvqa \
            --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
            --result-file ./playground/data/eval/textvqa/answers/$Model.jsonl


        # MM-VET
        echo "DivPrune Layer$divprune_layer save$divprune_save -> VTW Layer$vtw_layer save$vtw_save"
        python -m llava.eval.model_vqa \
            --model-path $CKPT \
            --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
            --image-folder ./playground/data/eval/mm-vet/images \
            --answers-file ./playground/data/eval/mm-vet/answers/$Model.jsonl \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --sparse \
            --attn_implementation $attn \
            --pruned_layer -1 \
            --image_token_start_index 35 \
            --image_token_length 576 \
            --pivot_image_token 4 \
            --pivot_text_token 4 \
            --max_num_trunction 576

        mkdir -p ./playground/data/eval/mm-vet/results
        python scripts/convert_mmvet_for_eval.py \
            --src ./playground/data/eval/mm-vet/answers/$Model.jsonl \
            --dst ./playground/data/eval/mm-vet/results/$Model.json


        # MMB
        SPLIT="mmbench_dev_20230712"
        echo "DivPrune Layer$divprune_layer save$divprune_save -> Random Layer$random_layer save$random_save"

        python -m llava.eval.model_vqa_mmbench \
            --model-path $CKPT \
            --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
            --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$Model.jsonl \
            --single-pred-prompt \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --sparse \
            --attn_implementation $attn \
            --pruned_layer -1 \
            --image_token_start_index 35 \
            --image_token_length 576 \
            --pivot_image_token 4 \
            --pivot_text_token 4 \
            --max_num_trunction 576

        mkdir -p ./playground/data/eval/mmbench/answers_upload/$SPLIT
        python scripts/convert_mmbench_for_submission.py \
            --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
            --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
            --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
            --experiment $Model


        # MMBENCH_DEV_CN
        SPLIT="mmbench_dev_cn_20231003"
        echo "DivPrune Layer$divprune_layer save$divprune_save -> VTW Layer$vtw_layer save$vtw_save"

        python -m llava.eval.model_vqa_mmbench \
            --model-path $CKPT \
            --question-file ./playground/data/eval/mmbench_cn/$SPLIT.tsv \
            --answers-file ./playground/data/eval/mmbench_cn/answers/$SPLIT/$Model.jsonl \
            --single-pred-prompt \
            --lang cn \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --sparse \
            --attn_implementation $attn \
            --pruned_layer -1 \
            --image_token_start_index 35 \
            --image_token_length 576 \
            --pivot_image_token 4 \
            --pivot_text_token 4 \
            --max_num_trunction 576

        mkdir -p ./playground/data/eval/mmbench_cn/answers_upload/$SPLIT
        python scripts/convert_mmbench_for_submission.py \
            --annotation-file ./playground/data/eval/mmbench_cn/$SPLIT.tsv \
            --result-dir ./playground/data/eval/mmbench_cn/answers/$SPLIT \
            --upload-dir ./playground/data/eval/mmbench_cn/answers_upload/$SPLIT \
            --experiment $Model

    done
done
