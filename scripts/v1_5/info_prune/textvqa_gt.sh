#!/bin/bash

# 1. 计算 TextVQA 验证集的最佳 GT 答案

annotation_file=./playground/data/eval/textvqa/TextVQA_0.5.1_val.json
result_file=./playground/data/eval/textvqa/answers/llava-v1.5-7b.jsonl # You can choose any answer's jsonl file.
python -m llava.eval.textvqa_best_answer \
    --annotation-file $annotation_file \
    --result-file $result_file \

# 2. 计算最佳 GT 答案对应的tokenizer标签

CKPT=/llava-v1.5-7b/ # Model weights
attn=flash_attention_2
python -m llava.eval.llava_tokenizer \
    --model-path $CKPT \
    --attn_implementation $attn \
