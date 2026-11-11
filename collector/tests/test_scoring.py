# ABOUTME: Tests photo scoring: pet prominence from sightings, the moments chosen to photograph,
# ABOUTME: and the ranking that picks the best frame of a visit.

from pathlib import Path

from collector.scoring import Candidate, Sighting, best_moments, pet_prominence, rank

CLEAR, BLURRED = 1.0, 0.4


def test_prominence_prefers_large_centered_pets() -> None:
    big_centered = Sighting(at=10.0, cx=320, cy=320, w=300, h=300, score=0.9)
    small_edge = Sighting(at=10.0, cx=30, cy=320, w=60, h=60, score=0.9)
    assert pet_prominence(big_centered, resolution=(640, 640)) > pet_prominence(
        small_edge, resolution=(640, 640)
    )


def test_prominence_is_zero_without_a_nearby_sighting() -> None:
    sightings = [Sighting(at=10.0, cx=320, cy=320, w=300, h=300, score=0.9)]
    assert pet_prominence(None, resolution=(640, 640)) == 0.0
    assert Candidate.nearest_sighting(at=15.0, sightings=sightings, window=1.0) is None
    assert Candidate.nearest_sighting(at=10.6, sightings=sightings, window=1.0) == sightings[0]


def test_rank_prefers_clear_frames_with_prominent_pets() -> None:
    sightings = [
        Sighting(at=100.0, cx=320, cy=320, w=300, h=300, score=0.9),
        Sighting(at=105.0, cx=40, cy=40, w=50, h=50, score=0.5),
    ]
    sharp_big_pet = Candidate(path=Path("a.jpg"), at=100.0)
    blurry_big_pet = Candidate(path=Path("b.jpg"), at=100.2)
    sharp_small_pet = Candidate(path=Path("c.jpg"), at=105.0)
    sharp_no_pet = Candidate(path=Path("d.jpg"), at=150.0)
    ranked = rank(
        [blurry_big_pet, sharp_no_pet, sharp_small_pet, sharp_big_pet],
        sightings,
        resolution=(640, 640),
        clarity={
            Path("a.jpg"): CLEAR,
            Path("b.jpg"): BLURRED,
            Path("c.jpg"): CLEAR,
            Path("d.jpg"): CLEAR,
        },
    )
    assert [c.path.name for c in ranked] == ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]


def test_rank_falls_back_to_clarity_without_sightings() -> None:
    sharp = Candidate(path=Path("sharp.jpg"), at=1.0)
    soft = Candidate(path=Path("soft.jpg"), at=2.0)
    ranked = rank(
        [soft, sharp],
        sightings=[],
        resolution=(640, 640),
        clarity={sharp.path: CLEAR, soft.path: BLURRED},
    )
    assert [c.path.name for c in ranked] == ["sharp.jpg", "soft.jpg"]


def test_rank_puts_a_blurry_small_pet_ahead_of_a_sharp_empty_frame() -> None:
    sightings = [Sighting(at=100.0, cx=500, cy=200, w=60, h=50, score=0.4)]
    blurry_small_pet = Candidate(path=Path("pet.jpg"), at=100.0)
    sharp_no_pet = Candidate(path=Path("empty.jpg"), at=150.0)
    ranked = rank(
        [sharp_no_pet, blurry_small_pet],
        sightings,
        resolution=(640, 640),
        clarity={blurry_small_pet.path: BLURRED, sharp_no_pet.path: CLEAR},
    )
    assert [c.path.name for c in ranked] == ["pet.jpg", "empty.jpg"]


def test_best_moments_picks_the_most_prominent_sightings_spread_apart() -> None:
    burst = [
        Sighting(at=10.0 + i / 10, cx=320, cy=320, w=200 + i, h=200, score=0.9) for i in range(10)
    ]
    later = Sighting(at=20.0, cx=320, cy=320, w=150, h=150, score=0.9)
    last = Sighting(at=30.0, cx=320, cy=320, w=80, h=80, score=0.9)
    moments = best_moments(
        [last] + burst + [later], resolution=(640, 640), limit=3, spacing_seconds=1.0
    )
    assert moments == [burst[-1], later, last]
    assert best_moments(burst, resolution=(640, 640), limit=3, spacing_seconds=1.0) == [burst[-1]]
    assert best_moments([], resolution=(640, 640), limit=3, spacing_seconds=1.0) == []


def test_best_moments_go_by_the_worth_they_are_given() -> None:
    large = Sighting(at=10.0, cx=320, cy=320, w=300, h=300, score=0.9)
    small = Sighting(at=20.0, cx=320, cy=320, w=100, h=100, score=0.9)

    def still_only(sighting: Sighting) -> float:
        return 1.0 if sighting is small else 0.0

    assert best_moments([large, small], (640, 640), limit=1, spacing_seconds=1.0) == [large]
    assert best_moments(
        [large, small], (640, 640), limit=1, spacing_seconds=1.0, worth=still_only
    ) == [small]
