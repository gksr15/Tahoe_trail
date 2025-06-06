
import dask.dataframe as dd
import json

# # Login using e.g. `huggingface-cli login` to access this dataset
# print("Reading expression data")
# expression_df = dd.read_parquet("hf://datasets/tahoebio/Tahoe-100M/data/train-*.parquet")
# print("Reading cell line data")
# cell_df = dd.read_parquet("hf://datasets/tahoebio/Tahoe-100M/metadata/cell_line_metadata.parquet")
# print("Filtering cell lines")
# kras_pancreas_cells = cell_df[(cell_df['Driver_Gene_Symbol'] == 'KRAS') & (cell_df['Organ'] == 'Pancreas')]
# print("Filtering expression data")
# expression_subset_df = expression_df[expression_df['cell_line_id'].isin(kras_pancreas_cells['Cell_ID_Cellosaur'])]
# req_cols =['genes', 'expressions', 'moa-fine', 'drug', 'cell_line_id']
# pd_df = expression_subset_df[req_cols]
# #print(pd_df.shape)
# pd_df.to_csv('./merged_subset_total_df.csv')

print("Reading cell line metadata...")
cell_df = dd.read_parquet("hf://datasets/tahoebio/Tahoe-100M/metadata/cell_line_metadata.parquet")

print("Filtering for KRAS-driven pancreatic cancer cell lines...")
kras_pancreas_cells = cell_df[(cell_df['Driver_Gene_Symbol'] == 'KRAS') & (cell_df['Organ'] == 'Pancreas')]

# Convert to pandas to get the list of cell IDs
cell_ids = kras_pancreas_cells['Cell_ID_Cellosaur'].compute().tolist()

print(f"\nFound {len(cell_ids)} KRAS-driven pancreatic cancer cell lines:")
for cell_id in cell_ids:
    print(f"- {cell_id}")

# Save the cell IDs to a JSON file
output_file = 'kras_pancreas_cell_ids.json'
with open(output_file, 'w') as f:
    json.dump(cell_ids, f, indent=2)

print(f"\nSaved cell IDs to {output_file}")

