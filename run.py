from Facet_preprocessing import load_facet
from Generate_sentence import add_sentence
from training import train
from score_generation import score
import pandas as pd

if __name__ == "__main__":
    # Load facets
    df = load_facet()
    
    # Generate sentences based on facets
    df_sentences = add_sentence()
    
    # Score the generated sentences
    df_scored = score(df_sentences)
    
    # Train the model with the scored sentences
    model, tokenizer = train(df_scored)
    
    print("Training complete. Model and tokenizer saved.")
