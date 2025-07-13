from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
from tqdm import tqdm
import pandas as pd
from Facet_preprocessing import df
from Generate_sentence import add_sentence

def build_prompt(text, facet):
    return f"""You are a facet evaluation assistant.

Your job is to score the following **conversation turn** on the facet **"{facet}"**.

Rate from 1 to 5 based on how strongly the text reflects the facet:
1 = Not at all
3 = Moderately
5 = Very strongly

Conversation: "{text}"

Only reply with a number from 1 to 5.
Score:"""

def model_creation():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"  
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model

def generate_score(text, facet):
    prompt = build_prompt(text, facet)
    tokenizer, model = model_creation()
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=10)
    result = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract last number as score
    matches = re.findall(r'\b[1-5]\b', result)
    if matches:
        return int(matches[-1])
    return 3  # default fallback
def score(df):
    results = []
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        score = generate_score(row.text, row.facet)
        results.append({
            "facet": row.facet,
            "text": row.text,
            "score": score
        })
    df_scored = pd.DataFrame(results)

    return df_scored