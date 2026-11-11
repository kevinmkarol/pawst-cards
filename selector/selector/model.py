# ABOUTME: The selector's trained model: a linear scorer over a photo's features that learns both
# ABOUTME: which photos you rate as worth sending and which of two you rate higher.

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from selector.store import GOOD

# What the score should be for each rating: bad, okay, good, best.
TARGETS = np.array([0.0, 0.3, 1.0, 1.0])


@dataclass(frozen=True)
class Model:
    weights: np.ndarray
    bias: float
    mean: np.ndarray
    scale: np.ndarray

    def scores(self, features: np.ndarray) -> np.ndarray:
        """0 to 1 per photo; above a half means "you would call this good". A model trained
        before features gained their last columns reads the ones it knows."""
        known = features[:, : len(self.mean)]
        logits = standardised(known, self.mean, self.scale) @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-logits))

    def save(self, path: Path) -> None:
        np.savez(path, weights=self.weights, bias=self.bias, mean=self.mean, scale=self.scale)

    @staticmethod
    def load(path: Path) -> "Model":
        with np.load(path) as data:
            return Model(data["weights"], float(data["bias"]), data["mean"], data["scale"])


def standardised(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Features as steps of `scale` from `mean`. One that was not measured (NaN) counts as the
    mean, which moves the score neither way."""
    return np.nan_to_num((features - mean) / scale, nan=0.0)


def spread_of(features: np.ndarray) -> tuple:
    """The mean and scale of each column over the rows where it was measured. A column never
    measured gets a mean of 0, and with nothing to learn from it its weight stays 0."""
    measured = ~np.isnan(features)
    count = np.maximum(measured.sum(axis=0), 1)
    mean = np.where(measured, features, 0.0).sum(axis=0) / count
    variance = np.where(measured, (features - mean) ** 2, 0.0).sum(axis=0) / count
    return mean, np.sqrt(variance) + 1e-6


def rated_pairs(ratings: np.ndarray, visit: np.ndarray) -> tuple:
    """Every pair of frames from one visit that you rated differently: indexes of the higher
    rated, of the lower rated, and how many steps apart they are."""
    higher, lower, gap = [], [], []
    for number in np.unique(visit):
        frames = np.flatnonzero(visit == number)
        for first in frames:
            for second in frames:
                if ratings[first] > ratings[second]:
                    higher.append(first)
                    lower.append(second)
                    gap.append(ratings[first] - ratings[second])
    return np.array(higher, int), np.array(lower, int), np.array(gap, float)


def fit(
    features: np.ndarray,
    ratings: np.ndarray,
    visit: np.ndarray,
    l2: float = 5.0,
    pair_weight: float = 1.0,
) -> Model:
    """A linear score on standardised features, fitted to two things at once: each frame's
    score should match its rating (good and bad weighing the same in total however many there
    are of each), and within a visit a higher rated frame should outscore a lower rated one,
    the more so the further apart you rated them. `l2` holds the weights down so that a few
    hundred labels are not memorised. A feature that is NaN for a frame was not measured of
    it, and takes no part in that frame's score."""
    import torch

    mean, scale = spread_of(features)
    x = torch.tensor(standardised(features, mean, scale), dtype=torch.float64)
    target = torch.tensor(TARGETS[ratings], dtype=torch.float64)
    worth_sending = ratings >= GOOD
    share = float(worth_sending.mean())
    if 0 < share < 1:
        sample_weight = torch.tensor(
            np.where(worth_sending, 0.5 / share, 0.5 / (1 - share)), dtype=torch.float64
        )
    else:
        sample_weight = torch.ones(len(ratings), dtype=torch.float64)
    higher, lower, gap = rated_pairs(ratings, visit)
    gap_weight = torch.tensor(gap / gap.mean(), dtype=torch.float64) if len(gap) else None
    weights = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimiser = torch.optim.LBFGS([weights, bias], max_iter=300, line_search_fn="strong_wolfe")

    def loss() -> "torch.Tensor":
        optimiser.zero_grad()
        score = x @ weights + bias
        total = torch.nn.functional.binary_cross_entropy_with_logits(
            score, target, weight=sample_weight
        )
        if gap_weight is not None:
            margin = score[higher] - score[lower]
            total = (
                total + pair_weight * (gap_weight * torch.nn.functional.softplus(-margin)).mean()
            )
        total = total + l2 * (weights**2).sum() / (2 * len(ratings))
        total.backward()
        return total

    optimiser.step(loss)
    return Model(
        weights.detach().numpy().astype(np.float32),
        float(bias.detach()),
        mean.astype(np.float32),
        scale.astype(np.float32),
    )
