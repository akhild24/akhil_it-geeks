from .models import QueryClassification
from .entity_extraction import extract_sender
from .temporal_parser import parse_temporal_cues

def classify_query(query: str) -> QueryClassification:
    """
    Classifies a query into semantic, attributed, temporal, or mixed.
    Extracts the sender and date range if available.
    """
    sender = extract_sender(query)
    date_start, date_end = parse_temporal_cues(query)
    
    if sender and (date_start or date_end):
        q_type = "mixed"
    elif sender:
        q_type = "attributed"
    elif (date_start or date_end):
        q_type = "temporal"
    else:
        q_type = "semantic"
        
    return QueryClassification(
        query_type=q_type,
        sender=sender,
        date_start=date_start,
        date_end=date_end,
        original_query=query
    )
