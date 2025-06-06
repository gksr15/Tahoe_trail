# Tahoe Trail - Predicting Mechanism of Action (MOA) using Tahoe-100M Dataset and Transformer Models

## Introduction

This project aims to explore the effectiveness of transformer-based machine learning models for predicting the broad mechanism of action (MOA-broad) labels from transcriptional profiles of cells using the single-cell RNA-seq data from the Tahoe-100M dataset (https://huggingface.co/datasets/tahoebio/Tahoe-100M). Specifically, the focus is on cell lines derived from the pancreas with the oncogenic driver gene KRAS to accurately predict if the drugs in the study will have an inhibitory/antagonist effect, act as an activator/agonist, or if the MOA is unclear. By accurately predicting MOA labels, the goal is to demonstrate that transformer models can effectively learn meaningful biological signals from gene expression profiles influenced by various drug perturbations.

## Data Source and Selection

The Tahoe-100M dataset, hosted by the Arc Institute and available via Hugging Face, was utilized. The three files streamed contain a large expression data file consisting of sparse gene expression data for all the genes in a given cell line ID and its associated drug perturbations, a cell line metadata file with the driver gene and organ information associated with the cell line, and a third drug metadata file with different drugs and their effects and characteristics. For the purpose of computational ease, we utilized a small subset of pancreatic cell lines with KRAS driver mutations.

KRAS is one of the most frequently mutated oncogenes in pancreatic cancers; thus, predicting MOA specifically for KRAS-driven pancreatic cells provides clinically relevant insights.

**Data characteristics:**

- Large-scale single-cell RNA-seq (`expression_data`)
- `cell_line_metadata`
- `drug_metadata`

## Data Preparation

As mentioned earlier, to focus the analysis on a specific biological context, samples were filtered to include only those from pancreatic cell lines with KRAS as the identified driver gene. This was done using the `get_dataframe.py` script, which accessed and subsetted the dataset metadata to retain only the relevant cell lines based on the filtering criteria (`Organ == Pancreas` and `Driver_Gene_Symbol == KRAS`).

After filtering, the `dataset_updated.py` script was used to define a custom PyTorch dataset (`TahoeDataset`). This dataset class handled gene expression preprocessing, including log-normalization and top-K gene selection using the `f_classif` method based on ANOVA F-statistics. It also generated sparse expression vectors from the selected genes and mapped each sample to its corresponding MOA-broad label, which included three classes: `activator/agonist`, `inhibitor/antagonist`, and `unclear`.

The `dataset.py` script thus implemented batching, normalization, and dynamic label encoding to prepare the data for input into the transformer model. By processing data in a streaming manner and selecting a fixed number of the most informative genes, the approach remained memory-efficient while preserving the structure needed for downstream classification. Thus, the final input for the transformer model included:

- Sparse representation of gene expressions for selected genes
- MOA labels (broad categories: `inhibitor/antagonist`, `activator/agonist`, `unclear`)

## Transformer Model

**Model architecture (`model.py`):**

- `GeneExpressionTransformer` (Encoder)
- Input projection of gene expression vectors
- Positional encoding for gene sequences
- 4-layer Transformer with multi-head self-attention
- Global average pooling for sequence embedding
- Output: Predict MOA labels (multi-class classification)

**Reasoning for Transformer:**

Transformers effectively capture complex gene-gene interactions through self-attention mechanisms, making them suitable for high-dimensional expression data.

## Training and Evaluation

**Training setup (`trainer_updated.py`):**

- Optimizer: Adam
- Learning rate schedule: Cosine Annealing
- Loss function: Cross-Entropy Loss
- Batch size: 32
- Mixed precision training for efficiency

**Data split:**

- Train/validation/test split (90/5/5) for robust evaluation

**Evaluation metrics (`trainer.py` + `evaluate.py`):**

- Accuracy
- Cross-Entropy Loss
- TensorBoard logging for real-time monitoring

## Outcomes

Our pipeline with dataset, model, and trainer works end-to-end, but due to computational and resource constraints, we trained our model for one epoch on 500,000 samples. The resulting performance was:

- **Train Loss:** 0.7761  
- **Train Accuracy:** 0.7006

The checkpointed model was then evaluated on a test set of 10,000 samples using the `evaluate.py` script:

- **Test Accuracy:** 0.6653

This indicates that the model generalizes reasonably well to unseen data.

## Implementation Workflow (Summary)

| Step | Description                                             | Script/Code         |
|------|---------------------------------------------------------|---------------------|
| 1    | Filtering and Subsetting: Stream data, select KRAS pancreatic cells | `get_dataframe.py`  |
| 2    | Feature Engineering: Normalize, select top-K genes      | `dataset_updated.py`|
| 3    | Dataset Preparation: Custom PyTorch dataset and dataloader | `dataset_updated.py`|
| 4    | Model Definition: Transformer architecture              | `model.py`          |
| 5    | Training and Logging: Train the model, log metrics      | `trainer_updated.py`|
| 6    | Evaluation: Evaluate on held-out test set               | `evaluate.py`       |

## Rationale

The ultimate purpose is twofold:

- Demonstrate the capability of transformer models in decoding biological insights from transcriptional profiles of isolated cells from single-cell RNA-seq data.
- Understand how effectively gene expression perturbations induced by drugs can predict their MOA, particularly in a clinically relevant oncogenic context — in our case, KRAS-mutant pancreatic cancer.

## Future Directions

- Investigate whether attention heads align with KRAS downstream effectors (e.g., MAPK, PI3K, MYC signatures)
- Use this framework to identify off-target activators/inhibitors in KRAS-like expression states
- Extend to other RAS-family drivers to compare transformer attention maps across oncogene subtypes
