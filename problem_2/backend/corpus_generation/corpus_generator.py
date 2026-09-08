"""Build the deterministic synthetic corpus used by the application.

The data is deliberately fictional. A fixed seed makes the chat, gold IDs,
and evaluation results reproducible for reviewers and contributors.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from backend.corpus_generation.participants import PARTICIPANTS

SEED = 20260908
MESSAGE_COUNT = 4200
CORPUS_START = datetime(2026, 1, 12, 9, 0, 0)
CORPUS_END = datetime(2026, 7, 11, 20, 0, 0)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = PROJECT_ROOT / "data" / "chat_corpus_2.json"

FILLER_MESSAGES = [
    "kal standup ke baad chai?", "bhai wifi phir se gaya kya",
    "sending in 2 min pakka", "lol this meeting could have been a text",
    "koi canteen aa raha hai?", "[Forwarded] Please submit the form before 5 PM.",
    "haan dekh liya, clean hai", "mera laptop 3% pe hai help",
    "nhi yaar aaj ghar se kaam", "who has the extension board?",
    "arey typo tha, ignore", "coffee > every productivity hack",
    "drive link access de diya kya?", "main 10 min late aaunga sorry",
    "k", "done", "hmm", "[Forwarded] Weekend sale: headphones 40% off",
    "aaj ka lecture surprisingly useful tha", "pls dont merge without review",
    "ghar pahunch ke ping kar dena", "yaar traffic is criminal today",
    "quick poll bana do", "mere notes kisi ko chahiye?",
]


def _thread(_topic, messages):
    """Keep ordinary turns raw; only factual decision turns receive enrichment."""
    return [{"sender": sender, "text": text, "search_context": ""} for sender, text in messages]


SPECIAL_THREADS = {
    2850: _thread(
        "Manali group trip decision. Mountain weekend holiday. Dates, Volvo bus, stay, budget, and final agreement.",
        [
            ("Priya", "guys May ka long weekend aa raha hai, kuch actual plan karein?"),
            ("Simran", "pls mountains. city se break desperately chahiye"),
            ("Akhil", "Rishikesh bhi option hai, but travel thoda chaotic hoga"),
            ("Meera", "Manali side ke Volvo fares check kiye, early book karenge toh sane hain"),
            ("Kunal", "Old Manali me shared cottage mil raha hai, cafes bhi walking distance"),
            ("Neha", "Mall Road sounds crowded tbh, old side better vibes"),
            ("Rohan", "old manali"),
            ("Priya", "dates bolo. Thursday night nikle toh Monday class bach jayegi"),
            ("Simran", "14 May night works for me, return Sunday late"),
            ("Akhil", "Volvo semi-sleeper around 2.2k round trip aa raha hai"),
            ("Meera", "cottage total split karke approx 4k each. food extra"),
            ("Kunal", "so 9-10k cap rakhte? random splurging later dekh lenge"),
            ("Neha", "haan bhai, route and stay dono sensible lag rahe"),
            ("Priya", "Bas, pahaadon wali plan lock. 14 May ki raat Volvo; Old Manali stay. Koi more debate nahi."),
            ("Rohan", "finally"),
            ("Simran", "YASSSSS"),
            ("Akhil", "mai bus shortlist aur payment tracker bana deta hu"),
            ("Meera", "cottage host ko token dene se pehle names bhej dena"),
            ("Kunal", "[Forwarded] Manali packing list - power bank, jacket, meds"),
            ("Neha", "trip pe laptops banned except emergency lol"),
        ],
    ),
    3180: _thread(
        "Manali trip budget discussion led by Priya. Per-person ceiling, stay, transport, and payment split.",
        [
            ("Kunal", "fare + cottage ka updated sheet dekh lo once"),
            ("Meera", "transport 2200, stay 3900, rough food buffer 2500"),
            ("Simran", "matlab 8.6-ish without shopping? manageable"),
            ("Akhil", "token abhi 1200 per head dena hai warna rooms gone"),
            ("Rohan", "upi kar diya"),
            ("Neha", "shopping ko separate rakho warna spreadsheet ro degi"),
            ("Priya", "Mere liye 9.5k ceiling hai including travel aur stay. Uske upar mat le jana please."),
            ("Kunal", "fair. 9.5 max and personal shopping apna-apna"),
            ("Meera", "splitwise entry bana di, due date Tuesday evening"),
            ("Simran", "i can bring snacks so food buffer thoda chill"),
            ("Akhil", "bus seats hold kar raha hu, payment screenshots bhej do"),
            ("Priya", "great, money drama avoided before it started"),
            ("Rohan", "nice"),
            ("Neha", "budget minister Priya approved"),
            ("Kunal", "final sheet me all numbers visible hain, no surprise charges"),
            ("Meera", "[Forwarded] UPI payment reminder: token due by Tuesday 8 PM"),
            ("Simran", "doneeeee"),
            ("Akhil", "received 7/8, last person please dont vanish"),
        ],
    ),
    3545: _thread(
        "June campus demo planning decision. Team chooses the Friday lab slot, room, ownership, and presentation rehearsal.",
        [
            ("Akhil", "demo reviewers changed again, we need one clean rehearsal"),
            ("Meera", "Friday afternoon lab free hai according to coordinator"),
            ("Aditya", "morning slot clashes with viva, impossible for me"),
            ("Priya", "can everyone do 3 PM? gives us time to set projector"),
            ("Simran", "yes and cafeteria coffee before, non-negotiable"),
            ("Kunal", "Lab C-204 has HDMI and decent speakers, book that"),
            ("Rohan", "3 works"),
            ("Neha", "who is speaking for architecture section?"),
            ("Akhil", "i will do backend flow, Meera takes interface"),
            ("Meera", "Priya can open and Kunal handles questions maybe?"),
            ("Priya", "Final call: Friday, 19 June, 3 PM in C-204. Rehearsal Thursday after class."),
            ("Kunal", "locked. i will reserve the room now"),
            ("Simran", "please share deck by Wednesday night then"),
            ("Aditya", "i will trim my slides, too much existentialism apparently"),
            ("Neha", "[Forwarded] Demo checklist: adapter, backup PDF, hotspot"),
            ("Rohan", "noted"),
            ("Akhil", "projector test Thursday 4:30, dont be late"),
            ("Meera", "design freeze after rehearsal, no last minute color experiments"),
            ("Priya", "thank you, actual adults for once"),
            ("Simran", "rare group achievement unlocked"),
        ],
    ),
}

DECISION_CONTEXT = {
    2863: "final decision confirmation for the Manali group trip: mountain break, getaway, destination, 14 May night Volvo, Old Manali accommodation, group committed and plan settled",
    3186: "Priya's confirmed per-person budget limit for the Manali holiday including travel and stay: money, spend, cost, amount, price, maximum, ceiling",
    3555: "final campus project demo schedule: Friday 19 June at 3 PM in lab C-204 with rehearsal Thursday; presentation, showcase, room, slot, team schedule",
}


def _timestamps(count, rng):
    """Create non-uniform, ordered timestamps over exactly the six-month span."""
    duration = (CORPUS_END - CORPUS_START).total_seconds()
    weights = [rng.uniform(0.25, 2.5) for _ in range(count - 1)]
    scale = duration / sum(weights)
    timestamps = [CORPUS_START]
    current = CORPUS_START
    for weight in weights:
        current += timedelta(seconds=weight * scale)
        timestamps.append(current)
    timestamps[-1] = CORPUS_END
    return timestamps


def _filler(rng):
    sender = rng.choice(PARTICIPANTS)
    text = rng.choice(FILLER_MESSAGES)
    if rng.random() < 0.12 and text not in {"k", "done", "hmm"}:
        text += rng.choice([" yaar", " lol", "...", " bhai", " pls"])
    return {"sender": sender, "text": text, "search_context": "casual Hinglish group chat"}


def generate_corpus(message_count=MESSAGE_COUNT, seed=SEED):
    rng = random.Random(seed)
    messages = []
    while len(messages) < message_count:
        thread = SPECIAL_THREADS.get(len(messages))
        if thread:
            if len(messages) + len(thread) > message_count:
                raise ValueError("A special thread would exceed the requested message count")
            messages.extend(thread)
        else:
            messages.append(_filler(rng))

    timestamps = _timestamps(len(messages), rng)
    corpus = []
    for message_id, (message, timestamp) in enumerate(zip(messages, timestamps)):
        search_context = DECISION_CONTEXT.get(message_id, message["search_context"])
        corpus.append({
            "id": message_id,
            "sender": message["sender"],
            "timestamp": timestamp.isoformat(),
            "text": message["text"],
            "search_context": search_context,
            "prev_id": message_id - 1 if message_id else None,
            "next_id": message_id + 1 if message_id < len(messages) - 1 else None,
        })
    return corpus


def main():
    corpus = generate_corpus()
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CORPUS_PATH.open("w", encoding="utf-8") as file:
        json.dump(corpus, file, ensure_ascii=False, indent=2)
    print(f"Generated {len(corpus)} fictional messages at {CORPUS_PATH}")
    print(f"Seed: {SEED}; range: {corpus[0]['timestamp']} to {corpus[-1]['timestamp']}")


if __name__ == "__main__":
    main()
