"""
Creator Content Posting Optimization System
============================================
A deterministic recommendation engine that suggests optimal platform and time slot
for content submissions, maximizing creator engagement based on historical data,
platform activity, and creator-specific patterns.

Architecture:
  - DataLoader      : Loads and validates all input datasets (Issues 1-4, 16)
  - ScoringEngine   : Computes engagement scores for platform/time combinations (Issues 5, 17)
  - Recommender     : Jointly optimizes platform + time slot selection (Issues 6-11)
  - OutputFormatter : Produces structured recommendation output (Issue 12)
  - EvaluationEngine: Calculates evaluation metrics (Issue 18)
  - TestFramework   : Validates system correctness (Issue 20)

Design Decisions:
  - All operations are deterministic; tie-breaking uses sorted() on string keys.
  - Missing historical data falls back to platform-level averages, then global average.
  - Scoring formula: platform_activity * historical_engagement * creator_base * content_affinity
  - Joint platform+time optimization (Issue 8) avoids greedy two-step selection bias.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_PLATFORMS = {"instagram", "youtube", "tiktok", "twitter", "linkedin"}
VALID_CONTENT_TYPES = {"video", "image", "text", "story", "reel", "article"}
TIME_SLOTS = list(range(24))  # 0–23 (hour of day)

CONTENT_PLATFORM_AFFINITY: dict[str, dict[str, float]] = {
    "video":   {"youtube": 1.4, "tiktok": 1.3, "instagram": 1.1, "twitter": 0.8, "linkedin": 0.9},
    "reel":    {"tiktok": 1.5, "instagram": 1.4, "youtube": 1.0, "twitter": 0.7, "linkedin": 0.6},
    "image":   {"instagram": 1.5, "twitter": 1.1, "linkedin": 1.0, "tiktok": 0.9, "youtube": 0.7},
    "story":   {"instagram": 1.4, "tiktok": 1.2, "twitter": 0.9, "youtube": 0.6, "linkedin": 0.5},
    "text":    {"twitter": 1.4, "linkedin": 1.3, "instagram": 0.8, "youtube": 0.6, "tiktok": 0.7},
    "article": {"linkedin": 1.5, "twitter": 1.1, "youtube": 0.8, "instagram": 0.6, "tiktok": 0.5},
}

# Metric weights for evaluation (Issue 18)
METRIC_WEIGHTS = {
    "engagement_score": 0.40,
    "timing_effectiveness": 0.25,
    "platform_quality": 0.20,
    "efficiency_score": 0.15,
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class ContentSubmission:
    content_id: str
    creator_id: str
    content_type: str
    submitted_at: datetime   # timezone-aware UTC


@dataclass
class PlatformActivity:
    """Hourly activity scores per platform.  activity[platform][hour] -> float"""
    activity: dict[str, dict[int, float]] = field(default_factory=dict)

    def get(self, platform: str, hour: int) -> float:
        return self.activity.get(platform, {}).get(hour, 0.5)


@dataclass
class HistoricalEngagement:
    """engagement[creator_id][platform][content_type][hour] -> float"""
    data: dict[str, dict[str, dict[str, dict[int, float]]]] = field(default_factory=dict)

    def get(self, creator_id: str, platform: str,
            content_type: str, hour: int) -> float | None:
        return (self.data
                .get(creator_id, {})
                .get(platform, {})
                .get(content_type, {})
                .get(hour))

    def platform_average(self, platform: str, content_type: str) -> float:
        """Fallback: average across all creators for a given platform+content_type."""
        values: list[float] = []
        for creator_data in self.data.values():
            for ct, hours in creator_data.get(platform, {}).items():
                if ct == content_type:
                    values.extend(hours.values())
        return statistics.mean(values) if values else 1.0

    def global_average(self) -> float:
        values: list[float] = []
        for creator_data in self.data.values():
            for platform_data in creator_data.values():
                for ct_data in platform_data.values():
                    values.extend(ct_data.values())
        return statistics.mean(values) if values else 1.0


@dataclass
class CreatorProfile:
    """base_engagement[creator_id] -> float"""
    base_engagement: dict[str, float] = field(default_factory=dict)

    def get(self, creator_id: str) -> float:
        return self.base_engagement.get(creator_id, 1.0)


@dataclass
class Recommendation:
    content_id: str
    platform: str
    time_slot: int          # recommended hour (0–23)
    decision: str           # "post_now" | "schedule"
    score: float            # composite engagement score
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Issue 1–4 + 16 : Data Loading & Validation
# ---------------------------------------------------------------------------

class DataLoader:
    """
    Loads and validates all four datasets.

    Expected JSON schemas
    ---------------------
    content_submissions : list of
        {"content_id": str, "creator_id": str, "content_type": str,
         "submitted_at": ISO-8601 str}

    platform_activity   : list of
        {"platform": str, "hour": int, "activity_score": float}

    historical_engagement: list of
        {"creator_id": str, "platform": str, "content_type": str,
         "hour": int, "engagement": float}

    creator_profiles    : list of
        {"creator_id": str, "base_engagement": float}
    """

    # --- public entry points -------------------------------------------

    @staticmethod
    def load_content_submissions(raw: list[dict]) -> list[ContentSubmission]:
        """Issue 1"""
        submissions: list[ContentSubmission] = []
        for idx, record in enumerate(raw):
            try:
                DataLoader._validate_content_submission(record)
                dt = datetime.fromisoformat(record["submitted_at"])
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                submissions.append(ContentSubmission(
                    content_id=str(record["content_id"]),
                    creator_id=str(record["creator_id"]),
                    content_type=str(record["content_type"]).lower(),
                    submitted_at=dt,
                ))
            except (KeyError, ValueError, TypeError) as exc:
                print(f"[DataLoader] Skipping malformed content record #{idx}: {exc}")
        return submissions

    @staticmethod
    def load_platform_activity(raw: list[dict]) -> PlatformActivity:
        """Issue 2"""
        pa = PlatformActivity()
        for idx, record in enumerate(raw):
            try:
                platform = str(record["platform"]).lower()
                hour = int(record["hour"])
                score = float(record["activity_score"])
                if platform not in VALID_PLATFORMS:
                    raise ValueError(f"Unknown platform '{platform}'")
                if not (0 <= hour <= 23):
                    raise ValueError(f"hour {hour} out of range")
                if not math.isfinite(score):
                    raise ValueError("activity_score is not finite")
                pa.activity.setdefault(platform, {})[hour] = max(0.0, score)
            except (KeyError, ValueError, TypeError) as exc:
                print(f"[DataLoader] Skipping malformed activity record #{idx}: {exc}")
        return pa

    @staticmethod
    def load_historical_engagement(raw: list[dict]) -> HistoricalEngagement:
        """Issue 3 – large dataset; nested dicts for O(1) lookup."""
        he = HistoricalEngagement()
        for idx, record in enumerate(raw):
            try:
                cid = str(record["creator_id"])
                platform = str(record["platform"]).lower()
                ct = str(record["content_type"]).lower()
                hour = int(record["hour"])
                eng = float(record["engagement"])
                if not (0 <= hour <= 23):
                    raise ValueError(f"hour {hour} out of range")
                if not math.isfinite(eng) or eng < 0:
                    raise ValueError("engagement must be non-negative finite")
                (he.data
                   .setdefault(cid, {})
                   .setdefault(platform, {})
                   .setdefault(ct, {})[hour]) = eng
            except (KeyError, ValueError, TypeError) as exc:
                print(f"[DataLoader] Skipping malformed engagement record #{idx}: {exc}")
        return he

    @staticmethod
    def load_creator_profiles(raw: list[dict]) -> CreatorProfile:
        """Issue 4"""
        cp = CreatorProfile()
        for idx, record in enumerate(raw):
            try:
                cid = str(record["creator_id"])
                base = float(record["base_engagement"])
                if not math.isfinite(base) or base <= 0:
                    raise ValueError("base_engagement must be positive finite")
                cp.base_engagement[cid] = base
            except (KeyError, ValueError, TypeError) as exc:
                print(f"[DataLoader] Skipping malformed creator record #{idx}: {exc}")
        return cp

    # --- private helpers --------------------------------------------------

    @staticmethod
    def _validate_content_submission(record: dict) -> None:
        """Issue 16 – raises if record is invalid."""
        required = {"content_id", "creator_id", "content_type", "submitted_at"}
        missing = required - record.keys()
        if missing:
            raise KeyError(f"Missing fields: {missing}")
        ct = str(record["content_type"]).lower()
        if ct not in VALID_CONTENT_TYPES:
            raise ValueError(f"Invalid content_type '{ct}'")


# ---------------------------------------------------------------------------
# Issue 5, 17 : Scoring Engine
# ---------------------------------------------------------------------------

class ScoringEngine:
    """
    Computes a composite engagement score for a (platform, time_slot) pair.

    Formula (multiplicative so every factor is meaningful):
        score = platform_activity(p, t)
              * historical_engagement(creator, p, ct, t)   [with fallback]
              * creator_base(creator)
              * content_platform_affinity(ct, p)

    All factors are non-negative; score = 0 only when activity is 0.
    """

    def __init__(self,
                 platform_activity: PlatformActivity,
                 historical_engagement: HistoricalEngagement,
                 creator_profiles: CreatorProfile):
        self._pa = platform_activity
        self._he = historical_engagement
        self._cp = creator_profiles

        # Cache global average once (Issue 14 – avoid recomputation)
        self._global_avg: float | None = None

    def score(self,
              creator_id: str,
              content_type: str,
              platform: str,
              hour: int) -> float:
        """Issue 5 – single call returns a comparable float score."""
        activity = self._pa.get(platform, hour)

        # Historical engagement with two-level fallback (Issue 15)
        hist = self._he.get(creator_id, platform, content_type, hour)
        if hist is None:
            hist = self._he.platform_average(platform, content_type)
        if hist == 1.0:  # platform_average returned default → use global
            g = self._global_average()
            hist = g if g != 1.0 else 1.0

        base = self._cp.get(creator_id)
        affinity = CONTENT_PLATFORM_AFFINITY.get(
            content_type, {}).get(platform, 1.0)

        raw = activity * hist * base * affinity

        # Issue 17 – guard against NaN / Inf
        return raw if (math.isfinite(raw) and raw >= 0) else 0.0

    def _global_average(self) -> float:
        if self._global_avg is None:
            self._global_avg = self._he.global_average()
        return self._global_avg


# ---------------------------------------------------------------------------
# Issues 6–11 : Recommender
# ---------------------------------------------------------------------------

class Recommender:
    """
    Joint platform + time-slot optimization (Issue 8).

    For each content item the recommender evaluates all
    (platform × hour) combinations and picks the highest-scoring pair.
    Tie-breaking is deterministic: sort by (score DESC, platform ASC, hour ASC).
    """

    def __init__(self, scoring_engine: ScoringEngine):
        self._scorer = scoring_engine

    def recommend(self, submission: ContentSubmission) -> Recommendation:
        """Issues 6, 7, 8, 9, 10, 11."""
        best_platform, best_hour, best_score = self._joint_optimize(submission)

        # Issue 9 – schedule vs post now
        submitted_hour = submission.submitted_at.hour
        decision = self._scheduling_decision(submitted_hour, best_hour, best_score,
                                              submission.creator_id,
                                              submission.content_type,
                                              best_platform)

        return Recommendation(
            content_id=submission.content_id,
            platform=best_platform,
            time_slot=best_hour,
            decision=decision,
            score=round(best_score, 6),
            metadata={
                "submitted_hour": submitted_hour,
                "content_type": submission.content_type,
                "creator_id": submission.creator_id,
            },
        )

    # --- private ----------------------------------------------------------

    def _joint_optimize(self,
                        submission: ContentSubmission
                        ) -> tuple[str, int, float]:
        """Issue 8 – evaluate all (platform, hour) pairs jointly."""
        candidates: list[tuple[float, str, int]] = []

        for platform in sorted(VALID_PLATFORMS):          # sorted → determinism
            for hour in TIME_SLOTS:
                s = self._scorer.score(
                    submission.creator_id,
                    submission.content_type,
                    platform,
                    hour,
                )
                candidates.append((s, platform, hour))

        # Sort: highest score first; ties broken by platform (alpha) then hour
        # Issue 11 – fully deterministic
        candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
        best_score, best_platform, best_hour = candidates[0]
        return best_platform, best_hour, best_score

    def _scheduling_decision(self,
                              submitted_hour: int,
                              best_hour: int,
                              best_score: float,
                              creator_id: str,
                              content_type: str,
                              platform: str) -> str:
        """
        Issue 9 – compare expected engagement of posting now vs at best_hour.
        Post now if:
          - submitted_hour == best_hour  (already at peak)
          - score at submitted_hour >= 90 % of best_score  (near-peak tolerance)
          - best_hour is in the past relative to submitted_hour
            (next occurrence is tomorrow; schedule if benefit justifies wait)
        """
        current_score = self._scorer.score(
            creator_id, content_type, platform, submitted_hour)

        # If best slot is in the past today, it wraps to tomorrow – always schedule
        if best_hour < submitted_hour:
            threshold_met = current_score >= 0.90 * best_score
            return "post_now" if threshold_met else "schedule"

        if submitted_hour == best_hour:
            return "post_now"

        threshold_met = current_score >= 0.90 * best_score
        return "post_now" if threshold_met else "schedule"


# ---------------------------------------------------------------------------
# Issue 12 : Output Formatter
# ---------------------------------------------------------------------------

class OutputFormatter:
    """Converts internal Recommendation objects to the required output schema."""

    @staticmethod
    def format(rec: Recommendation) -> dict:
        return {
            "content_id": rec.content_id,
            "platform": rec.platform,
            "time_slot": rec.time_slot,
            "decision": rec.decision,
        }

    @staticmethod
    def format_all(recs: list[Recommendation]) -> list[dict]:
        return [OutputFormatter.format(r) for r in recs]

    @staticmethod
    def to_json(recs: list[Recommendation], indent: int = 2) -> str:
        return json.dumps(OutputFormatter.format_all(recs), indent=indent)


# ---------------------------------------------------------------------------
# Issue 18 : Evaluation Engine
# ---------------------------------------------------------------------------

class EvaluationEngine:
    """
    Calculates four metrics then combines them with METRIC_WEIGHTS.

    engagement_score    : normalised composite score of the chosen option
    timing_effectiveness: how close the chosen time is to the platform's peak hour
    platform_quality    : content–platform affinity normalised to [0,1]
    efficiency_score    : proportion of (platform, hour) pairs scored ≥ chosen score
                          (higher means the chosen option is easily reachable)
    """

    def __init__(self, scoring_engine: ScoringEngine,
                 platform_activity: PlatformActivity):
        self._scorer = scoring_engine
        self._pa = platform_activity

    def evaluate(self, rec: Recommendation,
                 submission: ContentSubmission) -> dict[str, float]:
        creator_id = submission.creator_id
        content_type = submission.content_type
        platform = rec.platform
        chosen_hour = rec.time_slot

        # 1. Engagement score – normalise by max across all slots on chosen platform
        scores_on_platform = [
            self._scorer.score(creator_id, content_type, platform, h)
            for h in TIME_SLOTS
        ]
        max_score = max(scores_on_platform) if scores_on_platform else 1.0
        engagement_score = min(rec.score / max_score, 1.0) if max_score > 0 else 0.0

        # 2. Timing effectiveness – 1.0 if chosen hour == peak hour, else decay
        peak_hour = max(TIME_SLOTS,
                        key=lambda h: self._pa.get(platform, h))
        hour_diff = min(abs(chosen_hour - peak_hour),
                        24 - abs(chosen_hour - peak_hour))
        timing_effectiveness = math.exp(-hour_diff / 6.0)   # half-life ~4 h

        # 3. Platform quality – affinity normalised to [0,1]
        affinities = list(CONTENT_PLATFORM_AFFINITY.get(content_type, {}).values())
        min_aff = min(affinities) if affinities else 0.5
        max_aff = max(affinities) if affinities else 1.5
        raw_aff = CONTENT_PLATFORM_AFFINITY.get(
            content_type, {}).get(platform, 1.0)
        span = max_aff - min_aff
        platform_quality = ((raw_aff - min_aff) / span) if span > 0 else 0.5

        # 4. Efficiency score – fraction of candidates that scored below chosen
        all_scores: list[float] = []
        for p in VALID_PLATFORMS:
            for h in TIME_SLOTS:
                all_scores.append(
                    self._scorer.score(creator_id, content_type, p, h))
        n_below = sum(1 for s in all_scores if s <= rec.score)
        efficiency_score = n_below / len(all_scores) if all_scores else 1.0

        metrics = {
            "engagement_score": round(engagement_score, 4),
            "timing_effectiveness": round(timing_effectiveness, 4),
            "platform_quality": round(platform_quality, 4),
            "efficiency_score": round(efficiency_score, 4),
        }
        composite = sum(
            metrics[k] * METRIC_WEIGHTS[k] for k in METRIC_WEIGHTS)
        metrics["composite"] = round(composite, 4)
        return metrics


# ---------------------------------------------------------------------------
# Issues 13, 14 : Pipeline (batch / burst processing)
# ---------------------------------------------------------------------------

class OptimizationPipeline:
    """
    High-level façade that wires all components together.
    Processes submissions in batch; each item is independent so the pipeline
    is trivially parallelisable if needed (Issues 13, 14).
    """

    def __init__(self,
                 platform_activity: PlatformActivity,
                 historical_engagement: HistoricalEngagement,
                 creator_profiles: CreatorProfile):
        self._scorer = ScoringEngine(
            platform_activity, historical_engagement, creator_profiles)
        self._recommender = Recommender(self._scorer)
        self._evaluator = EvaluationEngine(self._scorer, platform_activity)
        self._formatter = OutputFormatter()

    def run(self,
            submissions: list[ContentSubmission],
            include_metrics: bool = False
            ) -> list[dict]:
        """
        Process all submissions and return formatted output.
        Issue 13: each submission is processed independently – no cross-item
        bottlenecks.  Issue 14: scoring lookups are O(1) dict access.
        """
        results: list[dict] = []
        for submission in submissions:
            rec = self._recommender.recommend(submission)
            out = self._formatter.format(rec)
            if include_metrics:
                out["metrics"] = self._evaluator.evaluate(rec, submission)
            results.append(out)
        return results


# ---------------------------------------------------------------------------
# Issue 20 : Test & Validation Framework
# ---------------------------------------------------------------------------

class TestFramework:
    """
    Lightweight test suite; no external dependencies required.
    Run via:  python content_optimization_system.py --test
    """

    def __init__(self):
        self._passed = 0
        self._failed = 0

    # --- helpers ----------------------------------------------------------

    def _assert(self, condition: bool, name: str, detail: str = "") -> None:
        if condition:
            print(f"  ✓ {name}")
            self._passed += 1
        else:
            print(f"  ✗ {name}" + (f" — {detail}" if detail else ""))
            self._failed += 1

    # --- individual test cases -------------------------------------------

    def test_data_loading(self) -> None:
        print("\n[Test] Data Loading")
        raw_submissions = [
            {"content_id": "c1", "creator_id": "u1",
             "content_type": "video", "submitted_at": "2024-01-15T10:00:00+00:00"},
            {"content_id": "c2", "creator_id": "u2",
             "content_type": "image", "submitted_at": "2024-01-15T14:00:00+00:00"},
            # malformed – missing field
            {"content_id": "c3", "creator_id": "u3", "submitted_at": "2024-01-15T09:00:00"},
        ]
        subs = DataLoader.load_content_submissions(raw_submissions)
        self._assert(len(subs) == 2, "Malformed record skipped", f"got {len(subs)}")
        self._assert(subs[0].content_type == "video", "Content type normalised")
        self._assert(subs[0].submitted_at.tzinfo is not None, "Timezone attached")

    def test_platform_activity(self) -> None:
        print("\n[Test] Platform Activity")
        raw = [{"platform": "youtube", "hour": 20, "activity_score": 0.9},
               {"platform": "youtube", "hour": 8,  "activity_score": 0.4},
               {"platform": "INVALID", "hour": 5,  "activity_score": 0.3}]
        pa = DataLoader.load_platform_activity(raw)
        self._assert(pa.get("youtube", 20) == 0.9, "Activity score stored")
        self._assert(pa.get("youtube", 0) == 0.5,  "Missing hour returns default 0.5")
        self._assert("invalid" not in pa.activity,  "Invalid platform rejected")

    def test_scoring_determinism(self) -> None:
        print("\n[Test] Scoring Determinism (Issue 11)")
        pa = PlatformActivity(
            activity={"youtube": {h: 0.5 + 0.01 * h for h in TIME_SLOTS}})
        he = HistoricalEngagement(
            data={"u1": {"youtube": {"video": {h: 1.0 + 0.01 * h
                                               for h in TIME_SLOTS}}}})
        cp = CreatorProfile(base_engagement={"u1": 1.2})
        engine = ScoringEngine(pa, he, cp)

        s1 = engine.score("u1", "video", "youtube", 15)
        s2 = engine.score("u1", "video", "youtube", 15)
        self._assert(s1 == s2, "Same inputs → same score")

    def test_missing_data_fallback(self) -> None:
        print("\n[Test] Missing Data Fallback (Issue 15)")
        pa = PlatformActivity(
            activity={"instagram": {h: 0.7 for h in TIME_SLOTS}})
        he = HistoricalEngagement(data={})   # no historical data at all
        cp = CreatorProfile(base_engagement={})  # no profile
        engine = ScoringEngine(pa, he, cp)

        score = engine.score("unknown_creator", "image", "instagram", 10)
        self._assert(score > 0, "Score > 0 even with missing data",
                     f"got {score}")
        self._assert(math.isfinite(score), "Score is finite")

    def test_joint_optimization(self) -> None:
        print("\n[Test] Joint Optimisation (Issue 8)")
        # Make tiktok+reel+hour14 clearly best
        pa = PlatformActivity(activity={
            p: {h: (0.95 if (p == "tiktok" and h == 14) else 0.3)
                for h in TIME_SLOTS}
            for p in VALID_PLATFORMS
        })
        he = HistoricalEngagement(data={
            "u1": {"tiktok": {"reel": {14: 2.0}}}
        })
        cp = CreatorProfile(base_engagement={"u1": 1.0})
        scorer = ScoringEngine(pa, he, cp)
        rec_engine = Recommender(scorer)

        sub = ContentSubmission(
            content_id="x1", creator_id="u1", content_type="reel",
            submitted_at=datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc))
        rec = rec_engine.recommend(sub)
        self._assert(rec.platform == "tiktok", "Best platform selected",
                     f"got {rec.platform}")
        self._assert(rec.time_slot == 14, "Best hour selected",
                     f"got {rec.time_slot}")
        self._assert(rec.decision == "schedule", "Decision is schedule (not at peak yet)",
                     f"got {rec.decision}")

    def test_post_now_decision(self) -> None:
        print("\n[Test] Post-Now Decision (Issue 9)")
        pa = PlatformActivity(
            activity={"twitter": {h: 0.9 for h in TIME_SLOTS}})
        he = HistoricalEngagement(data={})
        cp = CreatorProfile(base_engagement={"u1": 1.0})
        scorer = ScoringEngine(pa, he, cp)
        rec_engine = Recommender(scorer)

        # Uniform activity → any hour is peak → decision should be post_now
        sub = ContentSubmission(
            content_id="x2", creator_id="u1", content_type="text",
            submitted_at=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc))
        rec = rec_engine.recommend(sub)
        self._assert(rec.decision == "post_now",
                     "Uniform activity → post_now",
                     f"got {rec.decision}, time_slot={rec.time_slot}")

    def test_output_format(self) -> None:
        print("\n[Test] Output Format (Issue 12)")
        rec = Recommendation("c99", "linkedin", 9, "schedule", 1.23)
        out = OutputFormatter.format(rec)
        for key in ("content_id", "platform", "time_slot", "decision"):
            self._assert(key in out, f"Key '{key}' present")
        self._assert(out["platform"] in VALID_PLATFORMS,
                     "Platform in valid set", f"got {out['platform']}")
        self._assert(out["decision"] in ("post_now", "schedule"),
                     "Decision valid", f"got {out['decision']}")

    def test_evaluation_metrics(self) -> None:
        print("\n[Test] Evaluation Metrics (Issue 18)")
        pa = PlatformActivity(
            activity={"instagram": {h: 0.5 + 0.02 * h for h in TIME_SLOTS}})
        he = HistoricalEngagement(
            data={"u1": {"instagram": {"image": {h: 1.0 for h in TIME_SLOTS}}}})
        cp = CreatorProfile(base_engagement={"u1": 1.0})
        scorer = ScoringEngine(pa, he, cp)
        evaluator = EvaluationEngine(scorer, pa)

        rec = Recommendation("c1", "instagram", 23, "schedule", 1.5)
        sub = ContentSubmission(
            "c1", "u1", "image",
            datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc))
        metrics = evaluator.evaluate(rec, sub)
        for key in ("engagement_score", "timing_effectiveness",
                    "platform_quality", "efficiency_score", "composite"):
            self._assert(key in metrics, f"Metric '{key}' present")
            self._assert(0.0 <= metrics[key] <= 1.0,
                         f"Metric '{key}' in [0,1]", f"got {metrics[key]}")

    def test_burst_processing(self) -> None:
        print("\n[Test] Burst Processing (Issue 13)")
        pa = PlatformActivity(
            activity={p: {h: 0.5 for h in TIME_SLOTS} for p in VALID_PLATFORMS})
        he = HistoricalEngagement(data={})
        cp = CreatorProfile(base_engagement={})
        pipeline = OptimizationPipeline(pa, he, cp)

        subs = [
            ContentSubmission(
                f"burst_{i}", "u_burst", "video",
                datetime(2024, 1, 15, i % 24, 0, tzinfo=timezone.utc))
            for i in range(50)
        ]
        results = pipeline.run(subs)
        self._assert(len(results) == 50,
                     "All 50 burst items processed", f"got {len(results)}")
        ids = [r["content_id"] for r in results]
        self._assert(len(set(ids)) == 50, "All content_ids unique")

    # --- runner -----------------------------------------------------------

    def run_all(self) -> None:
        print("=" * 60)
        print("  Creator Content Posting Optimization System – Tests")
        print("=" * 60)
        self.test_data_loading()
        self.test_platform_activity()
        self.test_scoring_determinism()
        self.test_missing_data_fallback()
        self.test_joint_optimization()
        self.test_post_now_decision()
        self.test_output_format()
        self.test_evaluation_metrics()
        self.test_burst_processing()

        total = self._passed + self._failed
        print(f"\n{'=' * 60}")
        print(f"  Results: {self._passed}/{total} passed"
              + ("  🎉" if self._failed == 0 else f"  ❌ {self._failed} failed"))
        print("=" * 60)


# ---------------------------------------------------------------------------
# Demo / Example Usage
# ---------------------------------------------------------------------------

def run_demo() -> None:
    """End-to-end demo with synthetic data."""
    print("=" * 60)
    print("  Creator Content Posting Optimization System – Demo")
    print("=" * 60)

    # --- synthetic datasets ----------------------------------------------
    import random
    rng = random.Random(42)   # fixed seed for reproducibility

    raw_activity: list[dict] = [
        {"platform": p, "hour": h,
         "activity_score": round(rng.uniform(0.2, 1.0), 2)}
        for p in VALID_PLATFORMS for h in TIME_SLOTS
    ]

    raw_engagement: list[dict] = [
        {"creator_id": f"creator_{c}", "platform": p,
         "content_type": ct, "hour": h,
         "engagement": round(rng.uniform(0.5, 3.0), 2)}
        for c in range(1, 6)
        for p in VALID_PLATFORMS
        for ct in VALID_CONTENT_TYPES
        for h in TIME_SLOTS
    ]

    raw_profiles: list[dict] = [
        {"creator_id": f"creator_{c}",
         "base_engagement": round(rng.uniform(0.8, 2.0), 2)}
        for c in range(1, 6)
    ]

    raw_submissions: list[dict] = [
        {"content_id": f"post_{i:03d}",
         "creator_id": f"creator_{rng.randint(1, 5)}",
         "content_type": rng.choice(list(VALID_CONTENT_TYPES)),
         "submitted_at": f"2024-06-10T{rng.randint(0,23):02d}:00:00+00:00"}
        for i in range(1, 11)
    ]

    # --- load data -------------------------------------------------------
    pa = DataLoader.load_platform_activity(raw_activity)
    he = DataLoader.load_historical_engagement(raw_engagement)
    cp = DataLoader.load_creator_profiles(raw_profiles)
    submissions = DataLoader.load_content_submissions(raw_submissions)

    # --- run pipeline ----------------------------------------------------
    pipeline = OptimizationPipeline(pa, he, cp)
    results = pipeline.run(submissions, include_metrics=True)

    # --- display ---------------------------------------------------------
    print(f"\nProcessed {len(results)} content submissions:\n")
    print(f"{'content_id':<12} {'platform':<12} {'time_slot':>10} "
          f"{'decision':<12} {'composite':>10}")
    print("-" * 60)
    for r in results:
        composite = r.get("metrics", {}).get("composite", "n/a")
        comp_str = f"{composite:.4f}" if isinstance(composite, float) else str(composite)
        print(f"{r['content_id']:<12} {r['platform']:<12} "
              f"{r['time_slot']:>10} {r['decision']:<12} {comp_str:>10}")

    print("\nFull JSON output (first 3 items):")
    print(json.dumps(results[:3], indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        TestFramework().run_all()
    else:
        run_demo()
        print("\nTip: run with --test flag to execute the full test suite.")