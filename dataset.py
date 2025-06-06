import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import pandas as pd
from typing import Optional, Dict, Any, Union, List
import os
from collections import defaultdict
import loompy as lp
import numpy as np
import tempfile
from pathlib import Path
from tqdm import tqdm
from sklearn.feature_selection import f_classif
import json

def compute_topk_genes(
    all_gene_ids: list[list[int]],
    all_expr_vals: list[list[float]],
    all_labels: list[int],
    vocab_size: int = 62000,
    top_k: int = 1000,
    method: str = "f_classif" 
):
    # Create the dense matrix (N, vocab_size)
    N = len(all_labels)
    expr_matrix = np.zeros((N, vocab_size), dtype=np.float32)

    for i, (gene_ids, expr_vals) in enumerate(zip(all_gene_ids, all_expr_vals)):
        expr_matrix[i, gene_ids] = expr_vals

    # Normalize 
    total_counts = expr_matrix.sum(axis=1, keepdims=True)
    expr_matrix = np.divide(expr_matrix, total_counts, out=np.zeros_like(expr_matrix), where=total_counts!=0)
    expr_matrix *= 1e4
    expr_matrix = np.log1p(expr_matrix)

    # Compute selection score
    if method == "f_classif":
        scores, _ = f_classif(expr_matrix, all_labels)
    elif method == "variance":
        scores = expr_matrix.var(axis=0)
    else:
        raise ValueError("Unsupported method")

    # Pick top-K gene indices
    topk_gene_ids = np.argsort(-scores)[:top_k]  # descending sort

    return topk_gene_ids.tolist()

class TahoeDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        output_dir: Optional[str] = None,
        batch_size: int = 32,
        gene_max_cnt: int = 1000,
        gene_selection_method: str = "f_classif",  # or 'variance'
        **kwargs
    ):
        """     Initialize the Tahoe-100M dataset from HuggingFace.     """
        self.split = split
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.gene_max_cnt = gene_max_cnt
        self.gene_selection_method = gene_selection_method
        
        # Set up output directory for tokenized data
        self.output_dir = output_dir or os.path.join(os.getcwd(), "tokenized_data")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize drug mapping
        self.drug_to_idx = {}
        self.next_drug_idx = 0
        
        # Define target cell lines as class attribute
        self.target_cell_lines = [
            "CVCL_0428",
            "CVCL_1119",
            "CVCL_0152",
            "CVCL_1634",
            "CVCL_0334",
            "CVCL_0480",
            "CVCL_1639",
            "CVCL_1635",
            "CVCL_1638"
        ]
        
        # Load gene metadata first
        self._load_gene_metadata()
        
        # Load drug metadata
        self._load_drug_metadata()
        
        # Load dataset from HuggingFace
        self._load_from_huggingface(**kwargs)
        
        # Initialize MOA mapping
        self.moa_to_idx = {}
        self.next_idx = 0
        print("Initializing dynamic MOA mapping for streaming data")
        
        # Load or compute top K genes
        self._load_or_compute_topk_genes()

    def _load_drug_metadata(self):
        """Load drug metadata and create mapping for broad MOA categories."""
        print("Loading drug metadata...")
        
        drug_metadata = load_dataset("tahoebio/Tahoe-100m", data_files="metadata/drug_metadata.parquet", split="train")
        
        # Create mapping from drug to broad MOA
        self.drug_to_broad_moa = {}
        for item in drug_metadata:
            self.drug_to_broad_moa[item['drug']] = item['moa-broad']
        
        print(f"Loaded metadata for {len(self.drug_to_broad_moa)} drugs")
        
    def _load_gene_metadata(self):
        """Load gene metadata and create mapping dictionaries."""
        print("Loading gene metadata...")
        
        gene_metadata = load_dataset("tahoebio/Tahoe-100m", data_files="metadata/gene_metadata.parquet", split="train")
        
        
        # Create mapping dictionaries
        self.token_to_symbol = {}
        self.token_to_ensembl = {}
        
        for item in gene_metadata:
            token_id = item['token_id']
            self.token_to_symbol[token_id] = item['gene_symbol']
            self.token_to_ensembl[token_id] = item['ensembl_id']
        
        print(f"Loaded metadata for {len(self.token_to_symbol)} genes")

    def _load_from_huggingface(self, **kwargs):
        """Load dataset from HuggingFace datasets."""
        # Load the dataset with streaming
        dataset = load_dataset("tahoebio/Tahoe-100m", split=self.split, streaming=True, **kwargs)
        
        # Filter for specific cell lines
        dataset = dataset.filter(self._filter_cell_line)
        
        # Shuffle after filtering since we have fewer samples
        dataset = dataset.shuffle(buffer_size=100000, seed=42)
        
        if self.max_samples is not None:
            dataset = dataset.take(self.max_samples)

        self.data = dataset

    def _filter_cell_line(self, sample):
        """Filter function for cell lines."""
        return sample["cell_line_id"] in self.target_cell_lines

    def _get_moa_idx(self, moa_label: str) -> int:
        """Get or create index for a MOA label."""
        if moa_label not in self.moa_to_idx:
            self.moa_to_idx[moa_label] = self.next_idx
            self.next_idx += 1
        return self.moa_to_idx[moa_label]

    def _get_drug_idx(self, drug_label: str) -> int:
        """Get or create index for a drug label."""
        if drug_label not in self.drug_to_idx:
            self.drug_to_idx[drug_label] = self.next_drug_idx
            self.next_drug_idx += 1
        return self.drug_to_idx[drug_label]

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return float('inf') if self.max_samples is None else self.max_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from the dataset."""
        if not hasattr(self, '_iterator'):
            self._iterator = iter(self.data)
        try:
            sample = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.data)
            sample = next(self._iterator)
        
        return self._process_sample(sample)

    def _load_or_compute_topk_genes(self):
        """Load pre-computed top K genes or compute them if not available."""
        topk_genes_path = os.path.join(self.output_dir, f"topk_genes_{self.gene_max_cnt}_{self.gene_selection_method}.json")
        
        if os.path.exists(topk_genes_path):
            print(f"Loading pre-computed top {self.gene_max_cnt} genes from {topk_genes_path}")
            with open(topk_genes_path, 'r') as f:
                self.topk_gene_ids = json.load(f)
        elif self.split == "train":
            print(f"Computing top {self.gene_max_cnt} genes using {self.gene_selection_method} method...")
            self.topk_gene_ids = self._compute_topk_genes()
            
            # Save the computed gene IDs
            print(f"Saving top {self.gene_max_cnt} genes to {topk_genes_path}")
            with open(topk_genes_path, 'w') as f:
                json.dump(self.topk_gene_ids, f)
        else:
            raise ValueError(f"Top K genes not found at {topk_genes_path}. Please run training split first.")
        

    def _compute_topk_genes(self):
        """Compute the top K most informative genes based on the selected method."""
        # Collect data for gene selection
        all_gene_ids = []
        all_expr_vals = []
        all_labels = []
        
        # subset of data for gene selection
        max_selection_samples = min(5000, self.max_samples if self.max_samples else 5000)
        
        iterator = iter(self.data)
        for _ in tqdm(range(max_selection_samples)):
            try:
                sample = next(iterator)
            except StopIteration:
                break
                
            # Skip first token (CLS token)
            genes = sample['genes'][1:]
            counts = sample['expressions'][1:]
            moa_idx = self._get_moa_idx(sample['moa-fine'])
            
            all_gene_ids.append(genes)
            all_expr_vals.append(counts)
            all_labels.append(moa_idx)
        
        # Compute top K genes
        vocab_size = max(max(ids) for ids in all_gene_ids) + 1
        topk_gene_ids = compute_topk_genes(
            all_gene_ids=all_gene_ids,
            all_expr_vals=all_expr_vals,
            all_labels=all_labels,
            vocab_size=vocab_size,
            top_k=self.gene_max_cnt,
            method=self.gene_selection_method
        )
        
        return topk_gene_ids

    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """   Process a single sample from the dataset.        """
        
        # Get genes for metadata
        genes = sample['genes'][1:] 
        counts = sample['expressions'][1:]
        
        # Create sparse matrix in torch
        sparse = torch.zeros(self.gene_max_cnt)
        
        # Fill sparse matrix with counts at gene indices
        genes = torch.tensor(genes, dtype=torch.long)
        values = torch.tensor(counts, dtype=torch.float)
        
        # Only keep genes that are in our top K selection
        mask = torch.isin(genes, torch.tensor(self.topk_gene_ids))
        indices = genes[mask]
        values = values[mask]
        
        # Map indices to their positions in the top K genes
        mapped_indices = torch.tensor([self.topk_gene_ids.index(idx.item()) for idx in indices])
        sparse.scatter_(0, mapped_indices, values)

        # Normalize sparse 
        total_counts = sparse.sum()
        
        if total_counts > 0:
            sparse = sparse / total_counts
        
            sparse = sparse * 1e4
        
            sparse = torch.log1p(sparse)  # log1p(x) = log(1 + x)

        # Get gene symbols and ensembl IDs for the selected genes
        gene_symbols = [self.token_to_symbol[g] for g in self.topk_gene_ids]
        gene_ensembl_ids = [self.token_to_ensembl[g] for g in self.topk_gene_ids]

        # Encode MOA label
        moa_label = sample['moa-fine']
        moa_idx = self._get_moa_idx(moa_label)
        
        # Get drug index and broad MOA
        drug_label = sample['drug']
        drug_idx = self._get_drug_idx(drug_label)
        moa_broad = self.drug_to_broad_moa.get(drug_label, "unclear")
        
        return {
            'gene_exp_norm': sparse,
            'moa_idx': torch.tensor(moa_idx, dtype=torch.long),
            'drug_idx': torch.tensor(drug_idx, dtype=torch.long),
            'gene_symbols': gene_symbols,
            'gene_ensembl_ids': gene_ensembl_ids,
            'cell_line_id': sample['cell_line_id'],
            'moa_label': moa_label,
            'drug': drug_label,
            'moa_broad': moa_broad,
        }

def custom_collate_fn(batch):
    """Custom collate function to handle both tensors and string lists.    """
    batched = {}
    
    # Get all keys from the first sample
    keys = batch[0].keys()
    
    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            batched[key] = torch.stack([sample[key] for sample in batch])
        else:
            # Each inner list represents one sample's genes
            batched[key] = [sample[key] for sample in batch]
    
    return batched

def get_dataloader(
    dataset: TahoeDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """     Create a DataLoader for the TahoeDataset.    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn, 
        **kwargs
    )

if __name__ == "__main__":
    # Example usage of loading data from HuggingFace
    print("Loading Tahoe-100M dataset from HuggingFace...")
    
    # Create dataset instance with a small sample
    dataset = TahoeDataset(
        split="train",
        max_samples=10, 
        gene_max_cnt=5000
    )
    
    # Create dataloader
    dataloader = get_dataloader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=1
    )
    
    # Get and print the first batch
    print("\nFirst batch from dataloader:")
    first_batch = next(iter(dataloader))
    
    # Print batch structure
    print("\nBatch shapes:")
    for key, value in first_batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {len(value)}")
    
    # Print sample values
    print("\nSample values from first item in batch:")
    for key, value in first_batch.items():
        if isinstance(value, torch.Tensor):
            print(key, value.shape)
        else:
            print(key, len(value), len(value[0]))  
        
