import re
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi

# Load your dataset
dataset = load_dataset("snoop2head/enron_aeslc_emails", split="train")

# Updated regex pattern with groups for "From:" and "To:"
email_structure_pattern = re.compile(
    r"(?i)^Date: .+\n"
    r"From: (?P<from>.+@.+)\n"
    r"To: (?P<to>.+)\n"
    r"Subject:.*\n"
    r"Body:\s*\n"
    r"(.|\n)+",
    re.MULTILINE
)

# Process emails to filter for @enron.com addresses only
def process_email(example):
    email_text = example["text"].strip()
    match = email_structure_pattern.search(email_text)
    if not match:
        return {"is_valid": False, "email_addresses": []}
    
    # Extract and lowercase the "From:" email
    from_email = match.group("from").strip().lower()
    
    # Extract and split the "To:" emails (assumed comma-separated)
    to_field = match.group("to").strip().lower()
    to_emails = [addr.strip() for addr in to_field.split(",")]
    
    # Filter for only @enron.com addresses
    enron_emails = []
    if from_email.endswith("@enron.com"):
        enron_emails.append(from_email)
    enron_emails.extend([addr for addr in to_emails if addr.endswith("@enron.com")])
    
    enron_emails = sorted(set(enron_emails))  # Deduplicate and sort
    is_valid = bool(enron_emails)
    return {"is_valid": is_valid, "email_addresses": enron_emails}

# Process the dataset with parallel processing
processed_dataset = dataset.map(process_email, num_proc=8)
filtered_dataset = processed_dataset.filter(lambda x: x["is_valid"])
filtered_dataset = filtered_dataset.remove_columns(["is_valid"])

# Create email-to-index mapping for @enron.com emails only
all_enron_emails = sorted(set(
    email for row in filtered_dataset["email_addresses"] for email in row
))
email_to_index = {email: idx for idx, email in enumerate(all_enron_emails)}
index_to_email = {idx: email for email, idx in email_to_index.items()}
print(f"Number of unique @enron.com email addresses: {len(email_to_index)}")

# Add one-hot labels based on the filtered @enron.com emails
def add_labels(example):
    indices = [email_to_index[email] for email in example["email_addresses"]]
    one_hot = np.zeros(len(all_enron_emails), dtype=int)
    one_hot[indices] = 1
    return {"label": one_hot.tolist()}

filtered_dataset = filtered_dataset.map(add_labels, num_proc=8)

# Create the main dataset (only one split) as a DatasetDict
dataset_dict = DatasetDict({"train": filtered_dataset})

# Push the main dataset to Hugging Face Hub
dataset_repo = "rishi-jha/filtered_enron"  # Change this as needed
dataset_dict.push_to_hub(dataset_repo)
print(f"Dataset uploaded to: https://huggingface.co/datasets/{dataset_repo}")

# Save the email mapping as a separate JSON file
import json
mapping_file = "data/email_to_index.json"
with open(mapping_file, "w") as f:
    json.dump(email_to_index, f, indent=4)

with open("data/index_to_email.json", "w") as f:
    json.dump(index_to_email, f, indent=4)

# Upload the mapping file as an artifact to your repository
# api = HfApi()
# api.upload_file(
#     path_or_fileobj=mapping_file,
#     path_in_repo="email_to_index.json",
#     repo_id=dataset_repo,
# )
# print(f"Email mapping uploaded to: https://huggingface.co/datasets/{dataset_repo}/blob/main/email_to_index.json")
