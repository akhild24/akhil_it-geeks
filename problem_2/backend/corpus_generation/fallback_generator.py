import random
from .participants import PARTICIPANTS

TEMPLATES = [
    # Template 1: Trip Decision (Zero Overlap for "decide trip")
    [
        {"role": "P1", "texts": ["14 ko free ho sab?", "14th weekend free?", "next long weekend kya plan hai?"]},
        {"role": "P2", "texts": ["haan mereko toh hai", "yup free", "haan bhai"]},
        {"role": "P3", "texts": ["train wali dekh li?", "tickets check ki?", "flight sasti hai kya?"]},
        {"role": "P1", "texts": ["haan wait", "checking", "ha ruk"]},
        {"role": "P2", "texts": ["okay fix hai phir", "done", "chalo set hai"]},
        {"role": "P3", "texts": ["booking mai krdu?", "mai book kar du?", "done book karta hu"]}
    ],
    # Template 2: Budget discussion
    [
        {"role": "P1", "texts": ["kitna kharcha aayega?", "budget kya socha hai?", "per head kitna?"]},
        {"role": "P2", "texts": ["around 5k", "5-6k max", "zyada nahi, 5k"]},
        {"role": "P3", "texts": ["sahi hai", "cool", "done"]},
        {"role": "P1", "texts": ["upi kar dena sab", "google pay krdo", "splitwise pe daal dena"]}
    ],
    # Template 3: Random Gossip
    [
        {"role": "P1", "texts": ["bhai kal kya hua pata hai?", "guys you won't believe", "omg kal ka gossip"]},
        {"role": "P2", "texts": ["kya hua?", "bol na", "kya?"]},
        {"role": "P3", "texts": ["jaldi bata", "suspense mat bana", "bata na bhai"]},
        {"role": "P1", "texts": ["wo ramesh ne job chhod di 😂", "riya aur uska break up ho gaya", "manager ne daant diya usko lol"]},
        {"role": "P2", "texts": ["lol 😂", "wtf", "sahi me? 😲"]},
        {"role": "P3", "texts": ["crazy yr", "omg", "bhai pagal hai wo"]}
    ],
    # Template 4: Late night food craving
    [
        {"role": "P1", "texts": ["bhook lagi hai yr", "zomato se order kare?", "maggi banau?"]},
        {"role": "P2", "texts": ["bhai raat ke 2 baj rahe hai", "ab kaun deliver karega", "so ja chup chap"]},
        {"role": "P1", "texts": ["pizza mangwa raha hu", "shawarma khayega?", "mac d khula hai"]},
        {"role": "P3", "texts": ["mere liye bhi 1", "haan order kar de", "i am in"]},
        {"role": "P2", "texts": ["chal theek hai 1 cheese burst", "ok done", "hmm mangwa le"]}
    ],
    # Template 5: Work/College Deadlines
    [
        {"role": "P1", "texts": ["assignment ho gaya?", "kal ka presentation ready?", "report submit ki?"]},
        {"role": "P2", "texts": ["nhi yr", "kuch nahi kiya", "abhi start karunga"]},
        {"role": "P3", "texts": ["bhai fail pakka", "ded hai hum", "lol same"]},
        {"role": "P1", "texts": ["copy bhej de koi", "help kardo guys", "plz share kardo na"]},
        {"role": "P2", "texts": ["karta hu wait", "sending in 5", "ruk laptop khol raha hu"]}
    ],
    # Template 6: Movie Plan
    [
        {"role": "P1", "texts": ["movie chale?", "deadpool dekhne chale?", "koi nayi movie lagi hai?"]},
        {"role": "P2", "texts": ["kab?", "weekend pe?", "time bol"]},
        {"role": "P1", "texts": ["kal evening", "saturday raat", "friday show"]},
        {"role": "P3", "texts": ["mere paas bms coupon hai", "100 off mil jayega", "mai book karta hu"]},
        {"role": "P2", "texts": ["mast", "set", "perfect"]}
    ],
    # Template 7: Very short burst (One words)
    [
        {"role": "P1", "texts": ["kaha ho?", "kidhar?", "reach kiya?"]},
        {"role": "P2", "texts": ["metro", "rasta", "bas 5 min"]},
        {"role": "P1", "texts": ["k", "ok", "jaldi aa"]},
        {"role": "P2", "texts": ["haan", "👍", "aaya"]}
    ]
]

def generate_fallback_messages(num_messages=4000):
    messages = []
    
    while len(messages) < num_messages:
        template = random.choice(TEMPLATES)
        
        # Pick 3 random distinct participants for this thread
        thread_participants = random.sample(PARTICIPANTS, 3)
        role_map = {
            "P1": thread_participants[0],
            "P2": thread_participants[1],
            "P3": thread_participants[2]
        }
        
        for msg_def in template:
            sender = role_map[msg_def["role"]]
            text = random.choice(msg_def["texts"])
            
            # Optionally add a typo or quirk noise ~10% of the time
            if random.random() < 0.1:
                noise = random.choice([" yaar", " lol", " 😂", "...", " bhai"])
                text += noise
                
            messages.append({
                "sender": sender,
                "text": text
            })
            
            if len(messages) >= num_messages:
                break
                
    return messages
