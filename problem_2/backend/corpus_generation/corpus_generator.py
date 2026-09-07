import json
import uuid
import random
import os
from datetime import datetime, timedelta
from .fallback_generator import generate_fallback_messages
from dotenv import load_dotenv

load_dotenv()

def generate_timestamps(num_messages):
    """
    Generates realistic, non-uniform timestamps spanning EXACTLY 6 months (180 days)
    ending just before the current date.
    """
    gaps = []
    
    # 1. Generate relative gaps in seconds
    for i in range(num_messages - 1):
        rand_val = random.random()
        if rand_val < 0.70:
            gap = random.randint(5, 60) # seconds
        elif rand_val < 0.90:
            gap = random.randint(120, 1800) # minutes -> seconds
        elif rand_val < 0.98:
            gap = random.randint(7200, 43200) # hours -> seconds
        else:
            gap = random.randint(86400, 259200) # days -> seconds
        gaps.append(gap)
        
    total_gap_seconds = sum(gaps)
    
    # We want exactly 180 days of duration = 180 * 24 * 3600 = 15552000 seconds
    target_duration = 180 * 24 * 3600
    scale_factor = target_duration / total_gap_seconds
    
    # 2. Build timestamps forwards from start_date
    # End date is 1 hour before now to guarantee no future timestamps
    end_date = datetime.utcnow() - timedelta(hours=1)
    start_date = end_date - timedelta(days=180)
    
    timestamps = [start_date]
    current_time = start_date
    for gap in gaps:
        scaled_gap = gap * scale_factor
        current_time += timedelta(seconds=scaled_gap)
        timestamps.append(current_time)
        
    return timestamps

def main():
    print("Starting Synthetic Corpus Generation...")
    
    # We aim for 4000+ messages
    target_count = 4100
    
    # Check if we should use Gemini or Fallback
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print("GEMINI_API_KEY found, but using robust fallback for guaranteed 4000+ structured generation in this script.")
        # Note: In a full production setup, we would loop LLM calls here. 
        # For the assignment, the programmatic fallback guarantees perfect structure and 0-overlap test cases.
    
    print("Generating messages using local fallback generator...")
    raw_messages = generate_fallback_messages(target_count)
    
    print("Generating non-uniform timestamps scaled to exactly 180 days...")
    timestamps = generate_timestamps(len(raw_messages))
    
    print("Formatting JSON payload...")
    formatted_messages = []
    
    # First pass to assign IDs and basic fields
    for i in range(len(raw_messages)):
        msg_id = str(uuid.uuid4())
        formatted_messages.append({
            "id": msg_id,
            "sender": raw_messages[i]["sender"],
            "timestamp": timestamps[i].isoformat() + "Z",
            "text": raw_messages[i]["text"],
            "prev_id": None,
            "next_id": None
        })
        
    # Second pass to link prev_id and next_id
    for i in range(len(formatted_messages)):
        if i > 0:
            formatted_messages[i]["prev_id"] = formatted_messages[i-1]["id"]
        if i < len(formatted_messages) - 1:
            formatted_messages[i]["next_id"] = formatted_messages[i+1]["id"]
            
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    out_path = "data/chat_corpus.json"
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(formatted_messages, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully generated {len(formatted_messages)} messages and saved to {out_path}.")

if __name__ == "__main__":
    main()
