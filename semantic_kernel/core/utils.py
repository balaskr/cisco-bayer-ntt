import json
import re


def load_stopwords_from_file(file_path: str) -> set:
    reconstructed_stopwords = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    reconstructed_stopwords.add(word)
    except FileNotFoundError:
        pass
    return reconstructed_stopwords

all_stopwords = load_stopwords_from_file("knowledge/stopwords.txt")

def search_json_objects(data, query):
    """
    Search JSON objects for any keyword from cleaned query.

    Args:
        data (list[dict]): List of JSON objects.
        query (str): User query string.

    Returns:
        list[dict]: List of matched JSON objects.
    """
    # Clean query tokens
    tokens = re.findall(r'\w+', query.lower())
    keywords = [t for t  in tokens if t not in all_stopwords]

    if not keywords:
        # If no keywords, it means the query might have been entirely stopwords,
        # or implies a broad request that should be handled by LISTALL or OVERALL.
        # For filtering purposes, if there are no keywords, this function yields no matches.
        # The routing logic in run_multimodal_query should handle "list all" via LISTALL label.
        return [] # Return empty list if no meaningful keywords for specific filtering

    matched = []
    for obj in data:
        dumped = json.dumps(obj).lower()
        
        # If any keyword is substring in dumped JSON, keep object
        if any(k in dumped for k in keywords):
            matched.append(obj)

    return matched
