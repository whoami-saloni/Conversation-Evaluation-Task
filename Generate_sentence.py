from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from tqdm import tqdm
from Facet_preprocessing import df


def generate_text(facet):
    model_id = "mistralai/Mistral-7B-Instruct-v0.1"  # Lighter than Mixtral
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
    inputs = tokenizer(prompt(facet), return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=40)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

def prompt(facet):
    prompt = f"""You are a high-quality dataset generator for a conversation evaluation AI benchmark.

Given the psychological or linguistic facet: **"{facet}"**, generate **10 short, natural-sounding conversation sentences**.

Each sentence should reflect the facet to **a different degree** — from very strong to irrelevant — but the order should be **mixed**, not sequential.

Make sure:
- All sentences sound like real dialogue
- The degree of relevance to the facet varies (some strong, some moderate, some unrelated)
- Behind the scenes, assign each sentence a score from 1 (not related) to 5 (very strongly related)

Facet: **{facet}**

Return a numbered list like this:

1. "..."
2. "..."
3. "..."
4. "..."
5. "..."
6. "..."
7. "..." (score: X)
8. "..." (score: X)
9. "..." (score: X)
10. "..." (score: X)
"""
    return prompt

def add_sentence():
    texts = []
    for facet in tqdm(df["Facets"]):
        try:
            samples = generate_text(facet)
            texts.extend(samples)
        except Exception as e:
            print(f"Error generating for facet: {facet} — {e}")
    df_texts = pd.DataFrame(texts)
    return df_texts
