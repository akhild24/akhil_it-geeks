import json
import os
from typing import List, Dict, Any

_CORPUS_MESSAGES = None

def _ensure_corpus_loaded():
    global _CORPUS_MESSAGES
    if _CORPUS_MESSAGES is None:
        corpus_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'chat_corpus_2.json'))
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
            
        _CORPUS_MESSAGES = {msg['id']: msg for msg in corpus_data}

def build_context(message_id: int, window: int = 5) -> List[Dict[str, Any]]:
    """
    Builds conversational context around a specific message.
    Returns up to 'window' messages before and after, following prev_id and next_id.
    """
    _ensure_corpus_loaded()
    
    if message_id not in _CORPUS_MESSAGES:
        return []
        
    context_msgs = []
    
    # 1. Trace backward
    current_id = message_id
    before_msgs = []
    for _ in range(window):
        msg = _CORPUS_MESSAGES.get(current_id)
        if not msg or msg.get('prev_id') is None:
            break
        prev_id = msg['prev_id']
        prev_msg = _CORPUS_MESSAGES.get(prev_id)
        if not prev_msg:
            break
        before_msgs.append(prev_msg)
        current_id = prev_id
        
    # Reverse to make it chronological
    before_msgs.reverse()
    
    # 2. Add target message
    target_msg = _CORPUS_MESSAGES[message_id]
    
    # 3. Trace forward
    current_id = message_id
    after_msgs = []
    for _ in range(window):
        msg = _CORPUS_MESSAGES.get(current_id)
        if not msg or msg.get('next_id') is None:
            break
        next_id = msg['next_id']
        next_msg = _CORPUS_MESSAGES.get(next_id)
        if not next_msg:
            break
        after_msgs.append(next_msg)
        current_id = next_id
        
    # Combine and format
    all_raw_msgs = before_msgs + [target_msg] + after_msgs
    
    for raw_msg in all_raw_msgs:
        is_match = (raw_msg['id'] == message_id)
        context_msgs.append({
            'id': raw_msg['id'],
            'sender': raw_msg['sender'],
            'timestamp': raw_msg['timestamp'],
            'text': raw_msg['text'],
            'is_match': is_match
        })
        
    return context_msgs
