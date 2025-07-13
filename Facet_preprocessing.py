import pandas as pd
import re
df = pd.read_csv("Facets Assignment.csv")
df.head()


def clean_facet(facet):
    """
    Removes special characters from facet names while keeping alphanumerics, spaces, dashes, and parentheses.
    """
    return re.sub(r'[^a-zA-Z0-9 ()\-]', '', facet).strip()

def load_facet():
    df = pd.read_csv("Facets Assignment.csv")
    df['Facets'] = df['Facets'].apply(clean_facet)
    return df