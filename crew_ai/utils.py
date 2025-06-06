import nltk
from nltk.corpus import stopwords
import re

# Download once if not already done
# nltk.download('stopwords')

# Load default English stop words
nltk_stopwords = set(stopwords.words('english'))

# Add your custom domain-specific stop words
custom_stopwords = {
    "site", "sites", "task", "tasks", "show", "me", "details", "of", "the",
    "and", "please", "give", "info", "information", "about", "list", "all", "get"
}

# Combined stop words set
all_stopwords = nltk_stopwords.union(custom_stopwords)

import json

def search_json_objects(data, query):
    """
    Search JSON objects for any keyword from cleaned query.

    Args:
        data (list[dict]): List of JSON objects.
        query (str): User query string.
        stopwords_set (set): Set of stopwords to filter from query.

    Returns:
        list[dict]: List of matched JSON objects.
    """
    import re

    # Clean query tokens
    tokens = re.findall(r'\w+', query.lower())
    keywords = [t for t in tokens if t not in all_stopwords]

    if not keywords:
        # fallback: return all if no keywords
        return data

    matched = []
    for obj in data:
        dumped = json.dumps(obj).lower()
        # If any keyword is substring in dumped JSON, keep object
        if any(k in dumped for k in keywords):
            matched.append(obj)

    return matched

# with open("knowledge/data.json", "r") as f:
#     hidden_json = json.load(f)

# filtered = search_json_objects(hidden_json["data"], "show me details of ATH2", all_stopwords)

# print(filtered)