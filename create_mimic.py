import argparse
import os
import json
import pandas as pd
import numpy as np
from datasets import load_dataset, DatasetDict

def trim_text_to_tokens(text, max_tokens=64):
    # Split by whitespace and join the first max_tokens tokens.
    tokens = text.split()
    return " ".join(tokens[:max_tokens])

def process_chunk(chunk, df_names, df_medcat_grouped, medcat_desc_map,
                  first_name_to_index, last_name_to_index, full_name_to_index, medcat_to_index,
                  supervised_patients, unsupervised_patients):
    # Merge NOTES chunk with names using SUBJECT_ID.
    merged = pd.merge(chunk, df_names, on="SUBJECT_ID", how="left")
    # Merge with MedCAT data (grouped by SUBJECT_ID).
    merged = pd.merge(merged, df_medcat_grouped, on="SUBJECT_ID", how="left")
    
    # Create a column that maps the MedCAT codes to their descriptions.
    merged["medcat_desc_list"] = merged["CODE"].apply(
        lambda codes: [medcat_desc_map.get(code, "UNK") for code in codes] if isinstance(codes, list) else []
    )
    # Create a full name column.
    merged["full_name"] = merged["FIRST_NAME"].astype(str) + " " + merged["LAST_NAME"].astype(str)
    
    rows = []
    for _, row in merged.iterrows():
        note = row["TEXT"]
        # Trim the note text to 64 tokens.
        trimmed_note = trim_text_to_tokens(note, max_tokens=64)
        
        first_name = row["FIRST_NAME"]
        last_name = row["LAST_NAME"]
        full_name = row["full_name"]
        medcat_list = row["medcat_desc_list"] if isinstance(row["medcat_desc_list"], list) else []
        
        # Instead of one-hot vectors, store the integer index.
        fn_index = first_name_to_index.get(first_name, None)
        ln_index = last_name_to_index.get(last_name, None)
        full_index = full_name_to_index.get(full_name, None)
        medcat_indices = [medcat_to_index[m] for m in medcat_list if m in medcat_to_index]
        
        # Determine split based on SUBJECT_ID.
        subject_id = row["SUBJECT_ID"]
        if subject_id in supervised_patients:
            split = "supervised"
        elif subject_id in unsupervised_patients:
            split = "unsupervised"
        else:
            split = "evaluation"
        
        entry = {
            "text": trimmed_note,
            "first_name_index": fn_index,
            "last_name_index": ln_index,
            "full_name_index": full_index,
            "medcat_indices": medcat_indices,
            "split": split
        }
        rows.append(entry)
    return rows

def main(data_dir, cache_dir, output_jsonl, chunksize):
    print("Starting the dataset creation process...")
    # Build full paths for each CSV file.
    path_names = os.path.join(data_dir, "SUBJECT_ID_to_NAME.csv")
    path_medcat = os.path.join(data_dir, "SUBJECT_ID_to_MedCAT.csv")
    path_medcat_desc = os.path.join(data_dir, "MedCAT_Descriptions.csv")
    path_notes = os.path.join(data_dir, "SUBJECT_ID_to_NOTES_templates.csv")
    
    print("Loading names from:", path_names)
    df_names = pd.read_csv(path_names)
    
    print("Loading MedCAT data from:", path_medcat)
    df_medcat = pd.read_csv(path_medcat)
    
    print("Loading MedCAT descriptions from:", path_medcat_desc)
    df_medcat_desc = pd.read_csv(path_medcat_desc)
    
    print("Grouping MedCAT codes by SUBJECT_ID...")
    df_medcat_grouped = df_medcat.groupby("SUBJECT_ID")["CODE"].apply(list).reset_index()
    
    print("Building medcat description mapping...")
    medcat_desc_map = dict(zip(df_medcat_desc["CODE"], df_medcat_desc["DESCRIPTION"]))
    
    print("Building mapping dictionaries from names and MedCAT descriptions...")
    unique_first_names = sorted(df_names["FIRST_NAME"].dropna().unique())
    unique_last_names = sorted(df_names["LAST_NAME"].dropna().unique())
    unique_full_names = sorted((df_names["FIRST_NAME"].astype(str) + " " + df_names["LAST_NAME"].astype(str)).unique())
    unique_medcats = sorted(set(medcat_desc_map.values()))
    
    first_name_to_index = {name: idx for idx, name in enumerate(unique_first_names)}
    last_name_to_index = {name: idx for idx, name in enumerate(unique_last_names)}
    full_name_to_index = {name: idx for idx, name in enumerate(unique_full_names)}
    medcat_to_index = {med: idx for idx, med in enumerate(unique_medcats)}
    
    print("Saving mapping files...")
    pd.DataFrame(list(first_name_to_index.items()), columns=["first_name", "index"]) \
      .to_csv(os.path.join(cache_dir, "first_name_mapping.csv"), index=False)
    pd.DataFrame(list(last_name_to_index.items()), columns=["last_name", "index"]) \
      .to_csv(os.path.join(cache_dir, "last_name_mapping.csv"), index=False)
    pd.DataFrame(list(full_name_to_index.items()), columns=["full_name", "index"]) \
      .to_csv(os.path.join(cache_dir, "full_name_mapping.csv"), index=False)
    pd.DataFrame(list(medcat_to_index.items()), columns=["medcat_description", "index"]) \
      .to_csv(os.path.join(cache_dir, "medcat_mapping.csv"), index=False)
    
    # Sample patients for each split.
    print("Sampling patients for supervised and unsupervised splits...")
    unique_patients = df_names["SUBJECT_ID"].unique()
    np.random.seed(42)  # for reproducibility
    shuffled_patients = np.random.permutation(unique_patients)
    supervised_patients = set(shuffled_patients[:20000])
    unsupervised_patients = set(shuffled_patients[20000:40000])
    # Evaluation patients: those not in either supervised or unsupervised.
    
    print("Processing NOTES file in chunks from:", path_notes)
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        chunk_num = 0
        for chunk in pd.read_csv(path_notes, chunksize=chunksize):
            chunk_num += 1
            print(f"Processing chunk {chunk_num} (rows {chunk.index.min()} - {chunk.index.max()})...")
            rows = process_chunk(
                chunk,
                df_names,
                df_medcat_grouped,
                medcat_desc_map,
                first_name_to_index,
                last_name_to_index,
                full_name_to_index,
                medcat_to_index,
                supervised_patients,
                unsupervised_patients
            )
            for row in rows:
                fout.write(json.dumps(row) + "\n")
    print("Finished writing processed data to JSONL file:", output_jsonl)
    
    print("Loading dataset from JSONL file using HuggingFace datasets...")
    # Load the full dataset.
    full_dataset = load_dataset('json', data_files=output_jsonl, split='train')
    
    # Filter into splits.
    supervised_ds = full_dataset.filter(lambda x: x['split'] == 'supervised')
    unsupervised_ds = full_dataset.filter(lambda x: x['split'] == 'unsupervised')
    evaluation_ds = full_dataset.filter(lambda x: x['split'] == 'evaluation')
    
    dataset = DatasetDict({
        'supervised': supervised_ds,
        'unsupervised': unsupervised_ds,
        'evaluation': evaluation_ds
    })
    
    os.makedirs(cache_dir, exist_ok=True)
    dataset.save_to_disk(cache_dir)
    print("Dataset with splits created and saved to cache at:", cache_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create a HuggingFace-compatible dataset with dense (index-based) representations and three splits."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the input CSV files.")
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="Directory where the HuggingFace dataset should be saved (cache).")
    parser.add_argument("--output_jsonl", type=str, default="output_dataset.jsonl",
                        help="Path to the intermediate JSONL file.")
    parser.add_argument("--chunksize", type=int, default=1000,
                        help="Number of rows to process per chunk.")
    args = parser.parse_args()
    
    main(args.data_dir, args.cache_dir, args.output_jsonl, args.chunksize)
