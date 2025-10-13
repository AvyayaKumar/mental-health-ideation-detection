from typing import Dict, List

HIGH_TRIGGERS = [
    "suicide",
    "kill myself",
    "end my life",
    "want to die",
    "hang myself",
    "overdose",
    "take my life",
    "kill myself tonight",
]

MEDIUM_TRIGGERS = [
    "self-harm",
    "self harm",
    "cutting",
    "cut myself",
    "i can't go on",
    "i cant go on",
    "hopeless",
    "worthless",
    "i wish i wasn't here",
    "i wish i wasnt here",
    "i don't want to live",
    "i dont want to live",
    "life is pointless",
]


def score(text: str) -> Dict:
    t = text.lower()
    matched_high: List[str] = [p for p in HIGH_TRIGGERS if p in t]
    matched_medium: List[str] = [p for p in MEDIUM_TRIGGERS if p in t]

    if matched_high:
        risk_level = "HIGH"
        score_val = 0.95
        triggers = matched_high
        guidance = (
            "High-risk indicators detected. Alert appropriate staff immediately and follow your school's escalation protocol."
        )
    elif matched_medium:
        risk_level = "MEDIUM"
        score_val = 0.6
        triggers = matched_medium
        guidance = (
            "Concerning signals present. Review with care and consider a check-in with the student per policy."
        )
    else:
        risk_level = "LOW"
        score_val = 0.1
        triggers = []
        guidance = "No explicit self-harm language detected by the heuristic. Still review in context."

    return {
        "risk_level": risk_level,
        "score": score_val,
        "triggers": triggers,
        "guidance": guidance,
        "needs_human_review": True,
    }
