import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv
from pydriller import Repository
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
import torch


device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")



def generate_llm_message(diff_content, filename, model, tokenizer, device):
    inputs = tokenizer(diff_content[:3000], return_tensors="pt", max_length=2048,
                       truncation=True, padding="max_length").to(device)
    # inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_new_tokens=10,
                                 num_beams=5, do_sample=False, early_stopping=True)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
    return pred if pred else "unknown"



def generate_rect_msg(diff_content, filename, model, tokenizer, device, 
                      src_before=None, src_after=None, llm_inference=None, human_commit=None):
    
    try:
    
        use_after = src_after and len(src_after) < 2000
        use_before = not use_after and src_before and len(src_before) < 2000

        if use_after:
            code_context = f"Source after (full):\n{src_after}"
            prompt = f"""
You are helping refine commit messages. 
Focus on the dominant change made in the code. 
Here is the file diff, the modified source code (after changes), 
the previous human-written commit message, and LLM's inference.

Diff:
{diff_content}

{code_context}

Human commit message: {human_commit if human_commit else "N/A"}
LLM inference: {llm_inference if llm_inference else "N/A"}

Now, generate a **concise and precise commit message (max 12 words)** 
focusing only on the dominant change.
Do not use vague terms like 'update', 'add', 'fix', 'change'.
"""
        elif use_before:
            code_context = f"Source before (full):\n{src_before}"
            prompt = f"""
You are helping refine commit messages. 
Focus on the dominant fix or modification. 
Here is the file diff, the original source code (before changes), 
the previous human-written commit message, and LLM's inference.

Diff:
{diff_content}

{code_context}

Human commit message: {human_commit if human_commit else "N/A"}
LLM inference: {llm_inference if llm_inference else "N/A"}

Now, generate a **concise and precise commit message (max 12 words)** 
highlighting the main bug fix or feature introduced. 
Avoid vague terms like 'update', 'add', 'fix', 'change'.
"""
        else:
            prompt = f"""
Generate a concise and precise commit message (max 12 words) 
based only on the file diff, human commit message, and LLM inference. 
Focus on the dominant change. Avoid vague words like 'update' or 'fix'.

Diff:
{diff_content}

Human commit message: {human_commit if human_commit else "N/A"}
LLM inference: {llm_inference if llm_inference else "N/A"}
"""

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True, padding="max_length").to(device)
        # inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=16,
                num_beams=6,
                do_sample=False,
                length_penalty=0.8,
                early_stopping=True
            )

        msg = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return msg if msg else f"update {filename}"

    except Exception as e:
        print(f"Error generating rectified message for {filename}: {e}")
        return f"update {filename}"



MODEL_NAME = "mamiksik/CommitPredictorT5"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to(device)
print("Model loaded.")

MODEL_NAME_2 = "SEBIS/code_trans_t5_base_commit_generation" 
tokenizer_2 = T5Tokenizer.from_pretrained(MODEL_NAME_2, use_fast=False)
model_2 = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_2)
model_2.to(device)

print("Model and tokenizer for retification loaded successfully")


REPO_PATH = '/Users/tejasmacipad/Desktop/Third_year/STT/lab2/boxmot'
commits_csv = '/Users/tejasmacipad/Desktop/Third_year/STT/lab2/commits.csv'
output_csv = '/Users/tejasmacipad/Desktop/Third_year/STT/lab2/diffanalysis.csv'



with open(commits_csv, 'r', encoding='utf-8') as infile, \
     open(output_csv, 'w', newline='', encoding='utf-8') as outfile:

    reader = csv.DictReader(infile)
    writer = csv.writer(outfile)
    writer.writerow([
        'Hash', 'Message', 'Filename', 'Source Code (prev)',
        'Source Code (current)', 'Diff', 'LLM Inference',
        'rectified message'
    ])

    count = 0
    for row in reader:
        commit_hash = row['Hash']
        commit_message = row['Message']
        count += 1
        print(f"Processing commit {count}: {commit_hash}")

        try:
            for commit in Repository(REPO_PATH, single=commit_hash).traverse_commits():
                for modified_file in commit.modified_files:
                    filename = modified_file.new_path or modified_file.old_path or modified_file.filename
                    if not filename:
                        continue

                    source_before = (modified_file.source_code_before or "").replace('\n', '\\n').replace('\r', '').replace('"', '""')
                    source_current = (modified_file.source_code or "").replace('\n', '\\n').replace('\r', '').replace('"', '""')
                    diff_content = (modified_file.diff or "").replace('\n', '\\n').replace('\r', '').replace('"', '""')

                    # llm_inference = generate_llm_message(diff_content, filename, model, tokenizer, device)
                    # rectified_msg = generate_rect_msg(diff_content, filename, model_2, tokenizer_2, device)

                    llm_inference = generate_llm_message(diff_content, filename, model, tokenizer, device)

                    rectified_msg = generate_rect_msg(
                        diff_content=diff_content,
                        filename=filename,
                        model=model_2,
                        tokenizer=tokenizer_2,
                        device=device,
                        src_before=source_before,
                        src_after=source_current,
                        llm_inference=llm_inference,
                        human_commit=commit_message
                    )

                    print(llm_inference, rectified_msg)

                    writer.writerow([
                        commit_hash,
                        commit_message,
                        filename,
                        source_before,
                        source_current,
                        diff_content,
                        llm_inference,
                        rectified_msg
                    ])
        except Exception as e:
            print(f"Error processing commit {commit_hash[:8]}: {e}")
            continue

print(f"Saved output to {output_csv}")