# Agricultural_Metaverse_Retrieval

This repository contains the code of the paper ["Hierarchical Vision-Language Retrieval of Educational Metaverse Content in Agriculture"](https://arxiv.org/abs/2508.13713) accepted at the [ICIAP 2025](https://sites.google.com/view/iciap25) conference.

## Museums
The collected Museums used in the paper are saved in the [museums.json](https://github.com/aliabdari/Agricultural_Metaverse_Retrieval/blob/main/museums.json) file

## Feature Generation
All of the modules regarding generating embeddings using different models are presented at the [feature_generation](https://github.com/aliabdari/Agricultural_Metaverse_Retrieval/tree/main/feature_generation) directory.

## Train and Evaluation
Different architectures presented in the paper, are in the [train_evaluation](https://github.com/aliabdari/Agricultural_Metaverse_Retrieval/tree/main/train_evaluation) directory. In order to execute one of the models like MobileCLIP you can use the following command:

```
python train_mobile_clip.py
```
