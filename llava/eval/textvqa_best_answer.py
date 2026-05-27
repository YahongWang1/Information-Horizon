"""
功能说明：
    本脚本用于为 TextVQA 数据集的每个问题生成"最佳 GT 答案"(Best Ground Truth Answer)。

    TextVQA 数据集每个问题有 10 个标注答案，按照 EvalAI 的软评分规则,同一个答案
    可能获得不同的准确率分数(0~1)。本脚本的核心逻辑是：
    1. 读取模型预测结果和数据集标注文件
    2. 使用 TextVQAAccuracyEvaluator 计算每个问题的 10 个 GT 答案中得分最高的作为 best_answer
    3. 特殊处理：若 best_answer 为空字符串，则用该问题的 10 个答案中的众数（出现次数最多的答案）替换
    4. 将所有问题的 best_answer 保存为 pickle 文件，供后续评估流程使用

输入：
    --annotation-file: TextVQA 官方标注文件(TextVQA_0.5.1_val.json)
    --result-file: 模型的预测结果文件(JSONL 格式)，用于匹配问题 ID 与标注答案，随机选取任意一个模型的预测结果即可，不影响输出的正确性

输出：
    ./playground/data/eval/textvqa/nonull_best_gts.pkl: 包含所有问题 best_answer 的列表

"""

import os
import argparse
import json
import re
import pickle
from collections import Counter

from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--result-file', type=str)
    return parser.parse_args()


def prompt_processor(prompt):
    if prompt.startswith('OCR tokens: '):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        question = match.group(1)
    elif 'Reference OCR token: ' in prompt and len(prompt.split('\n')) == 3:
        if prompt.startswith('Reference OCR token:'):
            question = prompt.split('\n')[1]
        else:
            question = prompt.split('\n')[0]
    elif len(prompt.split('\n')) == 2:
        question = prompt.split('\n')[0]
    else:
        assert False

    return question.lower()


def eval_single(annotation_file, result_file):
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    annotations = json.load(open(annotation_file))['data']
    annotations = {(annotation['image_id'], annotation['question'].lower()): annotation for annotation in annotations}
    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    for result in results:
        annotation = annotations[(result['question_id'], prompt_processor(result['prompt']))]
        pred_list.append({
            "pred_answer": result['text'],
            "gt_answers": annotation['answers'],
        })

    # HACK: 计算每个question的best answer
    evaluator = TextVQAAccuracyEvaluator()
    best_gts = evaluator.best_answer(pred_list)

    # 找到空字符串的位置
    empty_positions = [index for index, value in enumerate(best_gts) if value == '']

    # 打印空字符串对应的answers
    annotations = json.load(open(annotation_file))['data']
    for index in empty_positions:
        char_list = annotations[index]['answers']

        # 使用 Counter 统计出现次数
        counter = Counter(char_list)

        # 获取出现次数最多的字符串及其出现次数
        most_common_string, most_common_count = counter.most_common(1)[0]

        # 赋值
        best_gts[index] = most_common_string

    save_path = "./playground/data/eval/textvqa/nonull_best_gts.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(best_gts, f)

if __name__ == "__main__":
    args = get_args()
    eval_single(args.annotation_file, args.result_file)

