# ABOUTME: Trains the selector's model on your ratings and measures it honestly: every visit is
# ABOUTME: scored by a model that never saw it, next to what the built-in scoring picked.

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from selector.embed import BACKBONE
from selector.features import FEATURES, Featurizer
from selector.model import fit, rated_pairs
from selector.store import GOOD, Store, _write

# Fewer rated frames than this on either side of "good" and the result would mean nothing.
FEWEST_OF_EACH = 10
FEWEST_VISITS = 5
FOLDS = 5


class NotEnoughLabels(ValueError):
    pass


@dataclass(frozen=True)
class Labelled:
    features: np.ndarray
    ratings: np.ndarray
    # Index of the visit each frame belongs to, and whether it is the frame the built-in
    # scoring picked (the posted one, for visits rated before both picks were recorded).
    visit: np.ndarray
    posted: np.ndarray

    @property
    def good(self) -> np.ndarray:
        return self.ratings >= GOOD


def labelled_frames(store: Store, featurizer: Featurizer) -> Labelled:
    paths, ratings, visit, posted = [], [], [], []
    for number, (key, entry) in enumerate(sorted(store.labels()["visits"].items())):
        built_in = entry.get("heuristic") or entry.get("posted")
        for name, value in entry["frames"].items():
            kept = store.kept_frame(key, name)
            if kept.is_file():
                paths.append(kept)
                ratings.append(value)
                visit.append(number)
                posted.append(name == built_in)
    return Labelled(
        featurizer.features(featurizer.describe(paths)),
        np.array(ratings, int),
        np.array(visit, int),
        np.array(posted, bool),
    )


def auc(scores: np.ndarray, good: np.ndarray) -> Optional[float]:
    """Chance that a good photo outscores one that is not; 0.5 is guessing, 1.0 is perfect."""
    goods, rest = scores[good], scores[~good]
    if not len(goods) or not len(rest):
        return None
    wins = (goods[:, None] > rest[None, :]).sum() + 0.5 * (goods[:, None] == rest[None, :]).sum()
    return float(wins / (len(goods) * len(rest)))


def held_out_scores(data: Labelled, folds: int = FOLDS) -> np.ndarray:
    """A score for every rated frame from a model trained without that frame's visit. A frame
    is left as NaN when the visits it was held out from offer only one side of "good" to
    train on."""
    visits = np.unique(data.visit)
    order = np.random.default_rng(0).permutation(visits)
    scores = np.full(len(data.ratings), np.nan)
    for fold in range(folds):
        held = np.isin(data.visit, order[fold::folds])
        train = ~held
        if not held.any() or data.good[train].all() or not data.good[train].any():
            continue
        model = fit(data.features[train], data.ratings[train], data.visit[train])
        scores[held] = model.scores(data.features[held])
    return scores


def evaluate(data: Labelled, scores: np.ndarray) -> Dict[str, object]:
    """How each way of choosing does over the visits that offer a real choice: at least one
    frame rated good or best, and at least one rated lower than the visit's top rating.
    Frames without a score (NaN) are left out."""
    scored = ~np.isnan(scores)
    if not scored.all():
        data = Labelled(
            data.features[scored], data.ratings[scored], data.visit[scored], data.posted[scored]
        )
        scores = scores[scored]
    model_good: List[bool] = []
    model_top: List[bool] = []
    posted_good: List[bool] = []
    posted_top: List[bool] = []
    chance_good: List[float] = []
    chance_top: List[float] = []
    for number in np.unique(data.visit):
        frames = data.visit == number
        ratings = data.ratings[frames]
        if ratings.max() < GOOD or ratings.min() == ratings.max():
            continue
        pick = int(np.argmax(scores[frames]))
        model_good.append(bool(ratings[pick] >= GOOD))
        model_top.append(bool(ratings[pick] == ratings.max()))
        chance_good.append(float((ratings >= GOOD).mean()))
        chance_top.append(float((ratings == ratings.max()).mean()))
        if data.posted[frames].any():
            was_posted = int(np.argmax(data.posted[frames]))
            posted_good.append(bool(ratings[was_posted] >= GOOD))
            posted_top.append(bool(ratings[was_posted] == ratings.max()))
    higher, lower, _ = rated_pairs(data.ratings, data.visit)
    return {
        "auc": auc(scores, data.good),
        "pairs": int(len(higher)),
        "pairs_right": float((scores[higher] > scores[lower]).mean()) if len(higher) else None,
        "visits_with_a_choice": len(model_good),
        "model_picks_good": _share(model_good),
        "model_picks_top": _share(model_top),
        "heuristic_picks_good": _share(posted_good),
        "heuristic_picks_top": _share(posted_top),
        "heuristic_visits": len(posted_good),
        "random_picks_good": float(np.mean(chance_good)) if chance_good else None,
        "random_picks_top": float(np.mean(chance_top)) if chance_top else None,
    }


def _share(hits: List[bool]) -> Optional[float]:
    return float(np.mean(hits)) if hits else None


def train(store: Store, featurizer: Featurizer) -> dict:
    """Train on every rating, save the model as the current one, and return its record."""
    data = labelled_frames(store, featurizer)
    good, rest = int(data.good.sum()), int((~data.good).sum())
    visits = len(np.unique(data.visit))
    if good < FEWEST_OF_EACH or rest < FEWEST_OF_EACH or visits < FEWEST_VISITS:
        raise NotEnoughLabels(
            f"need at least {FEWEST_OF_EACH} frames rated good or best and {FEWEST_OF_EACH} "
            f"rated lower, from {FEWEST_VISITS} visits; have {good} and {rest} from {visits}"
        )
    metrics = evaluate(data, held_out_scores(data))
    name = time.strftime("%Y%m%d-%H%M%S")
    store.create()
    fit(data.features, data.ratings, data.visit).save(store.models_dir / f"{name}.npz")
    meta = {
        "name": name,
        "trained_at": time.time(),
        "backbone": "/".join(BACKBONE),
        "features": FEATURES,
        "ratings": [int((data.ratings == value).sum()) for value in range(4)],
        "good": good,
        "bad": rest,
        "visits": visits,
        "metrics": metrics,
    }
    _write(store.models_dir / f"{name}.json", meta)
    store.set_model(name)
    return meta


def describe(meta: dict) -> str:
    """The training record as a few lines of plain text."""
    metrics = meta["metrics"]

    def percent(value: Optional[float]) -> str:
        return "n/a" if value is None else f"{value:.0%}"

    return "\n".join(
        [
            f"model {meta['name']}: {meta['good']} frames rated good or best and "
            f"{meta['bad']} rated lower, from {meta['visits']} visits",
            f"  over {metrics['visits_with_a_choice']} visits with a real choice, the pick is "
            "one you rated good or best / is your top-rated frame:",
            f"    model      {percent(metrics['model_picks_good'])} / "
            f"{percent(metrics['model_picks_top'])}",
            f"    built-in   {percent(metrics['heuristic_picks_good'])} / "
            f"{percent(metrics['heuristic_picks_top'])} "
            f"(the built-in scoring's own pick, {metrics['heuristic_visits']} visits)",
            f"    at random  {percent(metrics['random_picks_good'])} / "
            f"{percent(metrics['random_picks_top'])}",
            f"  of {metrics['pairs']} same-visit pairs you rated differently, the model orders "
            f"{percent(metrics['pairs_right'])} your way",
        ]
    )
