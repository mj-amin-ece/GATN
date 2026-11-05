# GATN: Graph Attention Transformer Network for Retinal Disease Classification

**Multi-label classification of 20 retinal conditions from fundus images. An EfficientNet image encoder is combined with a graph over the disease labels, and a transformer learns how the labels relate.**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![timm](https://img.shields.io/badge/timm-EfficientNet-orange)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black)
![Status](https://img.shields.io/badge/status-research%20in%20progress-yellow)

## Motivation

Retinal diseases often occur together. For example, diabetic retinopathy frequently appears alongside macular edema. A standard CNN classifier predicts each label independently and ignores this structure. Graph-based multi-label methods (ML-GCN, GATN) instead model the labels as nodes in a graph and learn classifiers that share information along label correlations. This project tests whether that helps on medical images, using a strong CNN baseline for comparison.

## Method

```text
fundus image ──► EfficientNet-B2 (timm) ──► image feature ─────────────────┐
                                                                          ▼
label names ──► text embeddings (PubMedBERT / OpenAI) ──► label nodes ──► dot product ──► 20 sigmoid outputs
                                                            ▲
label statistics ──► co-occurrence / correlation adjacency ─┘
                     refined by a transformer (attention over the graph) + GCN layers
```

1. **Image branch:** an EfficientNet-B2 backbone (260 × 260 input, fundus-specific crop-to-square preprocessing) produces a global image embedding.
2. **Label graph:** each of the 20 labels (for example *Diabetic Retinopathy*, *Media Haze*, *Optic Disc Cupping*, *Age-Related Macular Degeneration*) is embedded from its expanded clinical name with a biomedical language model (PubMed BERT) or OpenAI embeddings. The base adjacency comes from label **co-occurrence** or **correlation** statistics, thresholded and re-weighted.
3. **Graph attention transformer:** a transformer refines the adjacency adaptively, and graph convolutions map the label embeddings to per-class classifiers that are applied to the image embedding.
4. **Evaluation:** mAP, overall and per-class precision, recall and F1 (OP/OR/OF1, CP/CR/CF1), against an EfficientNet baseline.

## Status

This is research in progress. The notebook contains the full pipeline: dataset analysis and visualization, adjacency construction, the label-embedding cache, the GATN model, and training and evaluation with mixed precision. Early training runs are logged in the notebook, and a full comparison with the baseline is planned.

## Repository structure

```text
.
├── GATN_code.ipynb    # data analysis, model, training and evaluation
└── requirements.txt
```

## Running

The notebook was developed on Google Colab with a GPU. Set `TrainConfig.data_root` to your dataset folder, which must contain `train_data.csv`, `val_data.csv` (one binary column per label) and the images. Then run the cells in order.

```bash
pip install -r requirements.txt
```

Credentials are read from the environment and never stored in the notebook:
- `HF_TOKEN`: Hugging Face token, needed only if the chosen text model is gated.
- `OPENAI_API_KEY`: needed only for `embed_type = "openai"`.

## References

- Veličković et al., *Graph Attention Networks*, ICLR 2018. [arXiv:1710.10903](https://arxiv.org/abs/1710.10903)
- Chen et al., *Multi-Label Image Recognition with Graph Convolutional Networks*, CVPR 2019.
- Chen et al., *Learning Category Correlations for Multi-Label Image Recognition with Graph Networks*, 2019.
- Zhang et al., *Graph Attention Transformer Network for Multi-Label Image Classification*, 2024.
- Tan & Le, *EfficientNetV2: Smaller Models and Faster Training*, ICML 2021. [arXiv:2104.00298](https://arxiv.org/abs/2104.00298)

## Author

**Mohammad Javad Amin** · [GitHub](https://github.com/mj-amin-ece)
