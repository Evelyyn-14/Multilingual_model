# Multilingual_model
User Manual: Running the Multilingual Training Script
This guide explains how to set up your environment, install the required packages, arrange your data files, and run the datasetModel.py training script that demonstrates masked‐LM, data augmentation, and fairness‐aware training on small FLORES‑101 text splits.

Multilingual.py has the data hardcoded into it so byy simply runiing python Multilingual.py will run the code  You stillneed sentencepieces however  

1. Requirements: 
Python: version 3.7 or higher
Git (optional, for cloning)
Disk Space: ~100 MB for model caches, plus your dataset files
Memory: ≥4 GB RAM (more is better)

2. Create & Activate a Virtual Environment (Recommended)

python3 -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate     # Windows PowerShell

3. Install Python Packages
With your virtualenv active, run:

pip install torch transformers sentencepiece datasets

torch: for deep learning
transformers: for XLM‑RoBERTa and tokenizers
sentencepiece: required by the tokenizer
datasets: (only if you later use load_dataset)

4. Prepare Your Dataset Files
This script reads four text files (one sentence per line) from a local FLORES‑101 download:

<project_root>/
  Multilingual.py
  flores101_dataset/
    dev/
      eng.dev          # English development set
      swh.dev          # Swahili development set
    devtest/
      eng.devtest      # English devtest set
      swh.devtest      # Swahili devtest set
Download FLORES‑101 from https://github.com/facebookresearch/flores

Extract or copy the four files into the folders above.
You may add more languages by placing additional .dev/.devtest files and updating the script’s file paths and labels.

5. Review & (Optional) Edit the Script
Open datasetModel.py and check:

File paths in the load_text_file(...) calls match your folder structure.
Language labels (e.g., label=0 for English, label=1 for Swahili).
Hyperparameters near the bottom:
batch_size=16
num_epochs=3
fairness_weight=0.1
max_length=128
Adjust these to your hardware and needs.

6. Run the Training Script
From the project root (with venv active):

python Multilingual.py
You should see console output like:


Epoch 1: MLM=41.6489, Fairness=0.0006, Total=41.6489
…
Epoch 3: MLM=13.9772, Fairness=0.0012, Total=13.9774

7. Understanding the Output
MLM: Masked‑language modeling loss—lower is better.
Fairness: Penalty for representation variance—ideally near zero.
Total: Sum of the two losses guiding training.

8. Troubleshooting
ModuleNotFoundError:
Ensure you ran pip install torch transformers sentencepiece datasets in your active venv.
Out of Memory / Killed Process:
Lower batch_size (e.g. to 4 or 1).
Set num_workers=0 in the DataLoader call.
Reduce max_length.
FileNotFoundError:
Confirm your flores101_dataset/... folder and file names exactly match those in the script.