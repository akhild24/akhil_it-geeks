import json
import os
from datetime import datetime
from .participants import PARTICIPANTS

def validate_corpus(filepath="data/chat_corpus.json"):
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return
        
    with open(filepath, "r", encoding="utf-8") as f:
        corpus = json.load(f)
        
    num_messages = len(corpus)
    
    senders = set()
    invalid_ids = 0
    broken_links = 0
    invalid_timestamps = 0
    empty_messages = 0
    
    seen_ids = set()
    
    # Check IDs and collect data
    for msg in corpus:
        if not msg.get("id"):
            invalid_ids += 1
        elif msg["id"] in seen_ids:
            invalid_ids += 1
        else:
            seen_ids.add(msg["id"])
            
        if not msg.get("text") or str(msg["text"]).strip() == "":
            empty_messages += 1
            
        senders.add(msg.get("sender"))
        
    # Check links and timestamps
    start_date = None
    end_date = None
    
    for i, msg in enumerate(corpus):
        try:
            ts = datetime.fromisoformat(msg["timestamp"].replace("Z", "+00:00"))
            if start_date is None or ts < start_date:
                start_date = ts
            if end_date is None or ts > end_date:
                end_date = ts
        except Exception:
            invalid_timestamps += 1
            
        # Check chronological order locally
        if i > 0:
            prev_ts = datetime.fromisoformat(corpus[i-1]["timestamp"].replace("Z", "+00:00"))
            ts = datetime.fromisoformat(msg["timestamp"].replace("Z", "+00:00"))
            if ts < prev_ts:
                invalid_timestamps += 1
                
        # Link check
        if msg.get("prev_id") is not None:
            if msg["prev_id"] not in seen_ids:
                broken_links += 1
        if msg.get("next_id") is not None:
            if msg["next_id"] not in seen_ids:
                broken_links += 1
                
        if i > 0 and msg.get("prev_id") != corpus[i-1]["id"]:
            broken_links += 1
        if i < len(corpus) - 1 and msg.get("next_id") != corpus[i+1]["id"]:
            broken_links += 1

    # Date validation
    current_date = datetime.utcnow().replace(tzinfo=None)
    future_dates = 0
    if end_date and end_date.replace(tzinfo=None) > current_date:
        future_dates = 1
        
    duration = (end_date - start_date).days if (end_date and start_date) else 0

    # Missing participants?
    missing_participants = set(PARTICIPANTS) - senders
    extra_participants = senders - set(PARTICIPANTS)
    
    status = "PASS"
    if num_messages < 4000: status = "FAIL (Not enough messages)"
    if len(senders) != 8: status = "FAIL (Participants mismatch)"
    if invalid_ids > 0: status = "FAIL (Invalid IDs)"
    if broken_links > 0: status = "FAIL (Broken links)"
    if invalid_timestamps > 0: status = "FAIL (Invalid timestamps)"
    if empty_messages > 0: status = "FAIL (Empty messages)"
    if missing_participants or extra_participants: status = "FAIL (Participant mismatch)"
    if future_dates > 0: status = "FAIL (Future timestamps detected)"
    if not (150 <= duration <= 210): status = f"FAIL (Duration {duration} days is not approx 6 months)"
    
    print("## CORPUS VALIDATION")
    print("-" * 25)
    print(f"Messages: {num_messages}")
    print(f"Participants: {len(senders)} {list(senders)}")
    if missing_participants:
        print(f"Missing: {missing_participants}")
    if extra_participants:
        print(f"Extra: {extra_participants}")
    print(f"Start date: {start_date.strftime('%Y-%m-%d') if start_date else 'N/A'}")
    print(f"End date: {end_date.strftime('%Y-%m-%d') if end_date else 'N/A'}")
    print(f"Duration: approximately {duration // 30} months ({duration} days)")
    print(f"Future dates: {'Yes' if future_dates > 0 else 'No'}")
    print(f"Invalid IDs: {invalid_ids}")
    print(f"Broken links: {broken_links}")
    print(f"Invalid timestamps: {invalid_timestamps}")
    print(f"Empty messages: {empty_messages}")
    print("-" * 25)
    print(f"Status: {status}")

if __name__ == "__main__":
    validate_corpus()
