import datetime
from typing import Dict, List, Optional, Tuple

# -----------------------------
# Issue 1–4: Data Loading
# -----------------------------
class ContentSubmission:
    def _init_(self, content_id: str, creator_id: str, content_type: str, timestamp: datetime.datetime):
        self.content_id = content_id
        self.creator_id = creator_id
        self.content_type = content_type
        self.timestamp = timestamp

class PlatformActivity:
    def _init_(self, platform: str, time_slot: int, score: float):
        self.platform = platform
        self.time_slot = time_slot
        self.score = score

class HistoricalEngagement:
    def _init_(self, creator_id: str, platform: str, content_type: str, time_slot: int, score: float):
        self.creator_id = creator_id
        self.platform = platform
        self.content_type = content_type
        self.time_slot = time_slot
        self.score = score

class CreatorBaseEngagement:
    def _init_(self, creator_id: str, multiplier: float):
        self.creator_id = creator_id
        self.multiplier = multiplier

# -----------------------------
# Issue 5: Scoring Function
# -----------------------------
def recommendation_score(activity: float, historical: float, base: float, content_factor: float) -> float:
    # Weighted combination
    return (0.4 * activity) + (0.4 * historical) + (0.15 * base) + (0.05 * content_factor)

# -----------------------------
# Issue 6–8: Platform & Time Optimization
# -----------------------------
def optimize_platform_time(content: ContentSubmission,
                           activities: List[PlatformActivity],
                           engagements: List[HistoricalEngagement],
                           base_engagements: Dict[str, CreatorBaseEngagement]) -> Tuple[str, int, float]:
    best_platform, best_slot, best_score = None, None, -1
    base = base_engagements.get(content.creator_id, CreatorBaseEngagement(content.creator_id, 1.0)).multiplier

    for act in activities:
        hist_score = next((h.score for h in engagements if h.creator_id == content.creator_id
                           and h.platform == act.platform
                           and h.content_type == content.content_type
                           and h.time_slot == act.time_slot), 0.0)
        score = recommendation_score(act.score, hist_score, base, content_factor=1.0)
        if score > best_score:
            best_platform, best_slot, best_score = act.platform, act.time_slot, score

    return best_platform, best_slot, best_score

# -----------------------------
# Issue 9: Scheduling Decision
# -----------------------------
def scheduling_decision(content: ContentSubmission, recommended_slot: int) -> str:
    current_slot = content.timestamp.hour
    if current_slot == recommended_slot:
        return "post_now"
    elif (recommended_slot - current_slot) <= 2:
        return "schedule"
    else:
        return "post_now"

# -----------------------------
# Issue 10: Creator Adaptation
# -----------------------------
def adapt_for_creator(creator_id: str, engagements: List[HistoricalEngagement]) -> float:
    creator_scores = [e.score for e in engagements if e.creator_id == creator_id]
    return sum(creator_scores) / len(creator_scores) if creator_scores else 1.0

# -----------------------------
# Issue 11–12: Deterministic Output
# -----------------------------
def generate_recommendation(content: ContentSubmission,
                            activities: List[PlatformActivity],
                            engagements: List[HistoricalEngagement],
                            base_engagements: Dict[str, CreatorBaseEngagement]) -> Dict:
    platform, slot, score = optimize_platform_time(content, activities, engagements, base_engagements)
    decision = scheduling_decision(content, slot)
    return {
        "content_id": content.content_id,
        "platform": platform,
        "time_slot": slot,
        "decision": decision,
        "score": score
    }

# -----------------------------
# Issue 13–14: Burst Handling & Latency
# -----------------------------
def process_batch(contents: List[ContentSubmission],
                  activities: List[PlatformActivity],
                  engagements: List[HistoricalEngagement],
                  base_engagements: Dict[str, CreatorBaseEngagement]) -> List[Dict]:
    return [generate_recommendation(c, activities, engagements, base_engagements) for c in contents]

# -----------------------------
# Issue 15–17: Error Handling
# -----------------------------
def safe_score(value: Optional[float]) -> float:
    return value if value is not None else 0.0

# -----------------------------
# Issue 18: Evaluation Metrics
# -----------------------------
def evaluate(recommendations: List[Dict]) -> Dict:
    avg_score = sum(r["score"] for r in recommendations) / len(recommendations)
    return {"avg_engagement_score": avg_score, "count": len(recommendations)}

# -----------------------------
# Issue 19–20: Documentation & Testing
# -----------------------------
def test_pipeline():
    # Example test case
    content = ContentSubmission("c1", "creatorA", "video", datetime.datetime.now())
    activities = [PlatformActivity("YouTube", 14, 0.8), PlatformActivity("Instagram", 14, 0.6)]
    engagements = [HistoricalEngagement("creatorA", "YouTube", "video", 14, 0.9)]
    base_engagements = {"creatorA": CreatorBaseEngagement("creatorA", 1.2)}

    rec = generate_recommendation(content, activities, engagements, base_engagements)
    assert rec["platform"] == "YouTube"
    assert rec["decision"] in ["post_now", "schedule"]
    print("Test passed:", rec)

if _name_ == "_main_":
    test_pipeline()