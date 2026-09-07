import re
from typing import Optional

# Corpus 2 actual participants as requested by the user
PARTICIPANTS = [
    "Priya",
    "Meera",
    "Aditya",
    "Neha",
    "Simran",
    "Kunal",
    "Akhil",
    "Rohan"
]

def extract_sender(query: str) -> Optional[str]:
    """
    Robustly extracts a sender's name from a query string.
    Only extracts names explicitly found in the PARTICIPANTS list.
    """
    query_lower = query.lower()
    
    # We use a word boundary regex to avoid partial matches
    # (e.g. matching "an" inside "another")
    for participant in PARTICIPANTS:
        pattern = rf'\b{re.escape(participant.lower())}\b'
        if re.search(pattern, query_lower):
            return participant
            
    return None
