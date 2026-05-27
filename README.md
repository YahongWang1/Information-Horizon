<div align="center">
  <h1 style="display: inline-block; margin: 0;">🛸When Token Pruning is Worse than Random: <br>Understanding Visual Token Information in VLLMs</h1>
</div>

<h4 align="center"> 

[Yahong Wang*](https://scholar.google.com/citations?user=ps7AntYAAAAJ&hl=en)<sup>1</sup>,
[Juncheng Wu*](https://chtholly17.github.io/)<sup>2,3</sup>,
[Zhangkai Ni](https://eezkni.github.io/)<sup>1✉</sup>,
Longzhen Yang<sup>1</sup>, 
[Yihang Liu](https://scholar.google.com/citations?user=Qsl7mMgAAAAJ&hl=zh-CN)<sup>1</sup>,<br>
Chengmei Yang<sup>1</sup>, 
Ying Wen<sup>4</sup>, 
Lianghua He<sup>1,5✉</sup>, 
[Xianfeng Tang](https://scholar.google.com/citations?user=u1PEv-QAAAAJ&hl=zh-CN),
[Hui Liu](https://huil.io/)<sup>3</sup>,
[Yuyin Zhou](https://yuyinzhou.github.io/)<sup>3</sup>,


<sup>1</sup>Tongji University, <sup>2</sup>University of California, Santa Cruz, <sup>3</sup>Amazon,<br>
<sup>4</sup>East China Normal University, <sup>5</sup>Shanghai Eye Disease Prevention and Treatment Center

</h4>

<div align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2512.07580-AD1C18.svg?logo=arXiv)](https://arxiv.org/pdf/2512.07580)
</div>

## 📢 News

- **`2026.05.27`** [Code](https://github.com/YahongWang1/Information-Horizon) is available!
- **`2026.02.21`** Our paper is accepted at CVPR 2026 main!
- **`2025.12.08`** Our [paper](https://arxiv.org/abs/2512.07580) is released!

Star 🌟 us if you think it is helpful!!

## ⚡Introduction
<p align='center'>
<img src='https://github.com/YahongWang1/Information-Horizon/blob/main/images/overview.png' alt='mask' width='1000px'>
</p>

> **TLDR:** By quantifying a visual token's information content(measured by the change in model output probabilities upon its removal), We identify an "information horizon" in Vision Large Language Models, beyond which deep-layer tokens lose salience and become redundant. Crucially, this horizon is dynamic: it extends deeper for visually intensive tasks (e.g., OCR vs. simple VQA) and strongly correlates with higher model capacity. Based on these findings, applying simple random pruning beyond this horizon outperforms complex training-free methods.

## 🛠 Preparation
### LLaVA
1. Clone this repository.

```bash
git clone https://github.com/YahongWang1/Information-Horizon
cd Information-Horizon
```

2. Environment Setup.

```Shell
 conda create -n Info python=3.10 -y
 conda activate Info
 pip install -e .
 pip install flash_attn==2.5.9.post1 --no-build-isolation
```

3. Download Benchmark.

Please follow instructions in [LLaVA-Evaluation](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).



## 🎯 Usage
### LLaVA
### 📖 Calculate Token Information (Figure 4)
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/cal_info/mme.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/cal_info/textvqa.sh
```

### ✂️ Prune with Token Information (Figure 6)
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/info_prune/mme.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/info_prune/textvqa_gt.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/info_prune/textvqa.sh
```

`textvqa_gt.sh` analyzes TextVQA ground-truth answers to calculate per-token information across all samples.

### 🎲 Random Pruning with DART/DivPrune (Table 1)
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/allbench_dart_random.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/allbench_dart_vtw.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/allbench_divprune_random.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/allbench_divprune_vtw.sh
```



## 📌 Citation

```bibtex
@misc{wang2026tokenpruningworserandom,
      title={When Token Pruning is Worse than Random: Understanding Visual Token Information in VLLMs}, 
      author={Yahong Wang and Juncheng Wu and Zhangkai Ni and Longzhen Yang and Yihang Liu and Chengmei Yang and Ying Wen and Lianghua He and Xianfeng Tang and Hui Liu and Yuyin Zhou},
      year={2026},
      eprint={2512.07580},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.07580}, 
}

```


## ❤️ Acknowledgment
Thanks to the open-source contributions of [LLaVA](https://github.com/haotian-liu/LLaVA), [DivPrune](https://github.com/vbdi/divprune), and [DART](https://github.com/ZichenWen1/DART).



## 📭 Contact
For any questions about our paper or code, please email `yahongwang@tongji.edu.cn`.
