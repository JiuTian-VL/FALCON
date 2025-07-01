# Evaluation

To ensure the reproducibility, we evaluate the models with greedy decoding.

We have provided scripts for conveniently running the evaluations.

## High-resolution Benchmarks

### MME-RealWorld

1. Prepare data following the official instruction of [MME-RealWorld](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld).
2. Single- or Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/jiutian/eval/mme_realworld.sh
```

### Vstar

1. Prepare data following the official instruction of [Vstar](https://github.com/penghao-wu/vstar?tab=readme-ov-file#benchmark).
2. Single-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/jiutian/eval/vstar.sh
```

### Document

We mainly follow [mPLUG-Docowl1.5](https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/DocOwl1.5#model-evaluation) for evaluation.

1. Install the following packages.
```shell
pip install jsonlines
pip install textdistance
pip install editdistance
pip install pycocoevalcap
```
2. Single-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/jiutian/eval/doc.sh
```

## General Benchmarks

For the evaluation of SQA, TextVQA, POPE and MMBench, we follow the implementation of [LLaVA](https://github.com/haotian-liu/LLaVA).

Please first download [eval.zip](https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing) provided by LLaVA. Then modify the path in the script to the corresponding directory.

### ScienceQA

1. Download `pid_splits.json`, `problems.json` from the `data/scienceqa` folder of the ScienceQA [repo](https://github.com/lupantech/ScienceQA).
2. Download ScienceQA images following official [instructions](https://github.com/lupantech/ScienceQA/tree/main#ghost-download-the-dataset).
3. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/jiutian/eval/sqa.sh
```

### TextVQA

1. Download [`TextVQA_0.5.1_val.json`](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json) and [images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip).
2. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/jiutian/eval/textvqa.sh
```

### POPE

1. Download [coco2014 images](https://cocodataset.org/#download).
2. Download `coco` from [POPE](https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco).
3. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/jiutian/eval/pope.sh
```

### MMBench

1. Single-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/jiutian/eval/mmbench.sh
```
2. Submit the results to the [evaluation server](https://opencompass.org.cn/leaderboard-multimodal).

### SEED-Bench

1. Following the official [instructions](https://github.com/AILab-CVC/SEED-Bench/blob/main/DATASET.md) to download the images.
2. Multiple-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/jiutian/eval/seed.sh
```

### GQA

1. Download the [data](https://cs.stanford.edu/people/dorarad/gqa/download.html) and [evaluation scripts](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html) following the official instructions. You may need to modify `eval.py` as [this](https://gist.github.com/haotian-liu/db6eddc2a984b4cbcc8a7f26fd523187) due to the missing assets in the GQA v1.2 release.
2. Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/jiutian/eval/gqa.sh
```


## Inference on Custom Datasets

We have developed a well-encapsulated class `JiutianHDInfer` specifically designed for model inference in `jiutian/eval/model_infer.py`.

Below is an example of how to use the `JiutianHDInfer` class. By calling the `inference` method, you can easily obtain the model's inference results.

Note that it is important to set `conv_mode='llama_3_1'` for `Falcon-8B`.

```python
from jiutian.eval.model_infer import JiutianHDInfer

model_infer = JiutianHDInfer(
    model_path='/path/to/ckpt',
    model_base='/path/to/base_ckpt or None',
    conv_mode='llama_3_1',
)

image_file = '/path/to/image'
question = 'question'
model_infer.inference(image_file, question)
```