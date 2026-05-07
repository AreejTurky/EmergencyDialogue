# Llama-3.1-8B QLoRA — Bilingual 911 Emergency Dispatch

Fine-tuned on ~11,500 Arabic/English emergency dispatch dialogues using QLoRA on GCP Vertex AI.

## Model weights
Adapter hosted on Hugging Face:
👉 https://huggingface.co/your-username/llama31-911-qlora

## Load the model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import snapshot_download
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "https://anonymous-hf.up.railway.app/a/mwv18npgrwji/")
tokenizer = AutoTokenizer.from_pretrained("[your-username/llama31-911-qlora](https://anonymous-hf.up.railway.app/a/mwv18npgrwji/)")
```

## Training
- Base model: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- Method: QLoRA (r=16, alpha=32, dropout=0.05)
- Hardware: GCP Vertex AI — A100 40GB
- Data: 11,500 bilingual Arabic/English 911 dialogues
- Epochs: 1 | Batch size: 2 | Grad accumulation: 8

## Reproduce training
```bash
pip install -r requirements.txt
# Upload your sft_chatml.jsonl to GCS first, then:
jupyter notebook notebooks/01_finetune.ipynb
```
