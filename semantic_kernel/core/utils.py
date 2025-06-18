import json
import re

from nltk.corpus import stopwords

# Download once if not already done
# nltk.download('stopwords') # Uncomment if you haven't run this before

# Load default English stop words
nltk_stopwords = set(stopwords.words('english'))

# Add your custom domain-specific stop words
custom_stopwords = {
    "site", "sites", "task", "tasks", "show", "me", "details", "of", "the",
    "and", "please", "give", "info", "information", "about", "list", "all", "get",
    "status", "a", "an", "with", "my", "in", "a", "table", "entries", "everything", "full" # Added more presentation/meta words
}

# Combined stop words set
all_stopwords = nltk_stopwords.union(custom_stopwords)

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
