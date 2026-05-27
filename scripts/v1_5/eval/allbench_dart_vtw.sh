#!/bin/bash
# # DART+VTW两阶段剪枝实验

CKPT=/llava-v1.5-7b/
baseModel=llava-v1.5-7b

attn=flash_attention_2

# 两阶段剪枝配置：
# 格式：dart_layer dart_save vtw_layer vtw_save avg_tkn
# 192配置：layer2保留260，layer21全部剪枝
# 64配置：layer1保留60，layer26全部剪枝
configs=(
    "2 260 21 0 192"
    "1 60 26 0 64"
)

seq_num=1 # 重复次数

for i in $(seq 1 $seq_num); do
    for config in "${configs[@]}"; do
        # 解析配置：dart_layer dart_save vtw_layer vtw_save avg_tkn
        read -r dart_layer dart_save vtw_layer vtw_save avg_tkn<<< "$config"

        Model="$baseModel-DART+VTW-avg_tkn${avg_tkn}-DART-K${dart_layer}-save${dart_save}-VTW-K${vtw_layer}-save${vtw_save}-$i"

        export SAVE_NUM_LIST="[$dart_save, $vtw_save]"
        export PRUNED_LAYER_LIST="[$dart_layer, $vtw_layer]"
        export METHOD_LLM=DART
        export PRUNE_STG=VTW

        # MME
        echo "DART Layer$dart_layer save$dart_save -> VTW Layer$vtw_layer save$vtw_save"

        python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file ./playground/data/eval/MME/llava_mme.jsonl \
            --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
            --answers-file ./playground/data/eval/MME/answers/$Model.jsonl \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --sparse \
            --attn_implementation $attn \
            --pruned_layer $dart_layer \
            --image_token_start_index 35 \
            --image_token_length 576 \
            --pivot_image_token 4 \
            --pivot_text_token 4  \
            --max_num_trunction $dart_save

        cd ./playground/data/eval/MME

        python convert_answer_to_mme.py --experiment $Model

        cd eval_tool

        python calculation.py --results_dir answers/$Model

        cd /Information-Horizon




        # SQA
        echo "DART Layer$dart_layer save$dart_save -> VTW Layer$vtw_layer save$vtw_save"

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
            --pruned_layer $dart_layer \
            --image_token_start_index 35 \
            --image_token_length 576 \
            --pivot_image_token 4 \
            --pivot_text_token 4  \
            --max_num_trunction $dart_save

        python llava/eval/eval_science_qa.py \
            --base-dir ./playground/data/eval/scienceqa \
            --result-file ./playground/data/eval/scienceqa/answers/$Model.jsonl \
            --output-file ./playground/data/eval/scienceqa/answers/${Model}_output.jsonl \
            --output-result ./playground/data/eval/scienceqa/answers/${Model}_result.json




        # POPE
        echo "DART Layer$dart_layer save$dart_save -> VTW Layer$vtw_layer save$vtw_save"

        python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
            --image-folder ./playground/data/eval/pope/val2014 \
            --answers-file ./playground/data/eval/pope/answers/$Model.jsonl \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --sparse \
            --attn_implementation $attn \
            --pruned_layer $dart_layer \
            --image_token_start_index 35 \
            --image_token_length 576 \
            --pivot_image_token 4 \
            --pivot_text_token 4  \
            --max_num_trunction $dart_save

        python llava/eval/eval_pope.py \
            --annotation-dir ./playground/data/eval/pope/coco \
            --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
            --result-file ./playground/data/eval/pope/answers/$Model.jsonl



        # TextVQA
        echo "DART Layer$dart_layer save$dart_save -> VTW Layer$vtw_layer save$vtw_save"

        python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
            --image-folder ./playground/data/eval//textvqa/train_images \
            --answers-file ./playground/data/eval/textvqa/answers/$Model.jsonl \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --sparse \
            --attn_implementation $attn \
            --pruned_layer $dart_layer \
            --image_token_start_index 35 \
            --image_token_length 576 \
            --is_textvqa \
            --pivot_image_token 4 \
            --pivot_text_token 4  \
            --max_num_trunction $dart_save

        python -m llava.eval.eval_textvqa \
            --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
            --result-file ./playground/data/eval/textvqa/answers/$Model.jsonl


        
        # MM-VET
        echo "DART Layer$dart_layer save$dart_save -> VTW Layer$vtw_layer save$vtw_save"

        python -m llava.eval.model_vqa \
            --model-path $CKPT  \
            --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
            --image-folder ./playground/data/eval/mm-vet/images \
            --answers-file ./playground/data/eval/mm-vet/answers/$Model.jsonl \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --sparse \
            --attn_implementation $attn \
            --pruned_layer $dart_layer \
            --image_token_start_index 35 \
            --image_token_length 576 \
            --pivot_image_token 4 \
            --pivot_text_token 4  \
            --max_num_trunction $dart_save 

        mkdir -p ./playground/data/eval/mm-vet/results

        python scripts/convert_mmvet_for_eval.py \
            --src ./playground/data/eval/mm-vet/answers/$Model.jsonl \
            --dst ./playground/data/eval/mm-vet/results/$Model.json



        # MMB
        SPLIT="mmbench_dev_20230712"
        echo "DART Layer$dart_layer save$dart_save -> VTW Layer$vtw_layer save$vtw_save"

        python -m llava.eval.model_vqa_mmbench \
            --model-path $CKPT \
            --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
            --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$Model.jsonl \
            --single-pred-prompt \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --sparse \
            --attn_implementation $attn \
            --pruned_layer $dart_layer \
            --image_token_start_index 35 \
            --image_token_length 576 \
            --pivot_image_token 4 \
            --pivot_text_token 4  \
            --max_num_trunction $dart_save

        mkdir -p ./playground/data/eval/mmbench/answers_upload/$SPLIT

        python scripts/convert_mmbench_for_submission.py \
            --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
            --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
            --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
            --experiment $Model


        

        # # MMBENCH_DEV_CN
        SPLIT="mmbench_dev_cn_20231003"
        echo "DART Layer$dart_layer save$dart_save -> VTW Layer$vtw_layer save$vtw_save"

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
            --pruned_layer $dart_layer \
            --image_token_start_index 35 \
            --image_token_length 576 \
            --pivot_image_token 4 \
            --pivot_text_token 4  \
            --max_num_trunction $dart_save 

        mkdir -p ./playground/data/eval/mmbench_cn/answers_upload/$SPLIT

        python scripts/convert_mmbench_for_submission.py \
            --annotation-file ./playground/data/eval/mmbench_cn/$SPLIT.tsv \
            --result-dir ./playground/data/eval/mmbench_cn/answers/$SPLIT \
            --upload-dir ./playground/data/eval/mmbench_cn/answers_upload/$SPLIT \
            --experiment $Model

    done
done
