# ABOUTME: Tests the selector's tools with a stand-in for the image network: labels are kept
# ABOUTME: with their frames, training learns and reports honestly, and the page's API works.

import http.client
import http.server
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pytest
from PIL import Image

from selector.__main__ import main
from selector.annotate import App, make_handler, near_copies, propose
from selector.detect import PetBox
from selector.features import EARLIER_FEATURES, Described, Featurizer, rule_of_thumb
from selector.measures import MEASURED, measures_of
from selector.model import Model, fit, rated_pairs
from selector.shots import (
    FAMILIAR,
    SAME_SHOT,
    Looker,
    Shot,
    alikeness,
    fresh_picks,
    pick_fresh,
    repeat_cost,
)
from selector.store import BAD, BEST, GOOD, KEPT_WIDTH, OKAY, Store
from selector.train import Labelled, NotEnoughLabels, auc, evaluate, held_out_scores, train
from selector.visits import all_visits, archived_visit

BRIGHT, MID, DARK = (230, 230, 230), (150, 150, 150), (20, 20, 20)


def brightness(path: Path) -> float:
    with Image.open(path) as image:
        return float(np.asarray(image.convert("L").resize((4, 4))).mean()) / 255.0


class BrightnessEmbedder:
    """Stands in for CLIP: a photo's embedding is its brightness plus fixed padding, so
    'brighter photos are better' is something a linear model can learn."""

    def embed(self, paths: Sequence[Path], regions=None) -> np.ndarray:
        rows = [
            (
                [0.0, 0.0, 0.0]
                if regions is not None and regions[i] is None
                else [brightness(path), 0.5, 0.25]
            )
            for i, path in enumerate(paths)
        ]
        rows = np.array(rows, dtype=np.float32).reshape(len(paths), 3)
        lengths = np.linalg.norm(rows, axis=1, keepdims=True)
        return rows / np.where(lengths > 0, lengths, 1.0)


class BrightDetector:
    """Stands in for the object detector: the pet is in every photo that is not dark."""

    def detect(self, paths: Sequence[Path]) -> List[Optional[PetBox]]:
        return [
            PetBox(0.3, 0.3, 0.7, 0.7, 0.9) if brightness(path) > 0.3 else None for path in paths
        ]


def featurizer() -> Featurizer:
    return Featurizer(BrightnessEmbedder(), BrightDetector())


def make_visit(
    archive: Path,
    key: str,
    colours: List[Tuple[int, int, int]],
    posted: int = 0,
    started_at: float = 1000.0,
    new_format: bool = False,
) -> None:
    frames = archive / key / "frames"
    frames.mkdir(parents=True)
    names = []
    for index, colour in enumerate(colours):
        names.append(f"clip_at{index:06.2f}s.jpg")
        Image.new("RGB", (2592, 1944), colour).save(frames / names[-1])
    result: Dict[str, object] = {
        "best_frame": str(frames / names[posted]),
        "best_frame_at": started_at,
        "frames_considered": len(names),
    }
    if new_format:
        result["heuristic_frame"] = result["best_frame"]
        result["candidates"] = [
            {"frame": str(frames / name), "at": started_at, "source": "spread", "model_score": None}
            for name in names
        ]
    record = {"visit": {"id": key, "started_at": started_at, "clock_offset": 5.0}, "result": result}
    (archive / key / "visit.json").write_text(json.dumps(record))


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    directory = tmp_path / "visits"
    # Twelve visits of one bright and two dark frames; the dark first frame is the posted one.
    for number in range(12):
        make_visit(directory, f"visit-{number:02d}", [DARK, BRIGHT, DARK], started_at=1000 + number)
    (directory / "batch-20261112").mkdir()
    (directory / "batch-20261112" / "visit.json").write_text(json.dumps({"visits": []}))
    return directory


def label_all(store: Store, archive: Path) -> None:
    for visit in all_visits(archive, store):
        labels = {
            frame.name: GOOD if index == 1 else BAD for index, frame in enumerate(visit.frames)
        }
        store.label_visit(visit.key, visit.started_at, visit.posted, visit.paths(), labels)


def test_the_archive_is_read_newest_first_and_batches_are_not_visits(archive, tmp_path) -> None:
    visits = all_visits(archive, Store(tmp_path / "selector"))
    assert [visit.key for visit in visits][:2] == ["visit-11", "visit-10"]
    assert len(visits) == 12
    first = visits[0]
    assert first.started_at == 1016.0
    assert first.posted == first.heuristic == "clip_at000.00s.jpg" and first.model is None
    assert len(first.frames) == 3


def test_a_visit_with_recorded_candidates_lists_them_with_their_source(tmp_path: Path) -> None:
    make_visit(tmp_path, "v", [DARK, BRIGHT], new_format=True)
    visit = archived_visit(tmp_path / "v")
    assert [frame.source for frame in visit.frames] == ["spread", "spread"]


def test_a_frame_recorded_as_a_candidate_several_times_is_listed_once(tmp_path: Path) -> None:
    make_visit(tmp_path, "v", [DARK, BRIGHT], new_format=True)
    record = json.loads((tmp_path / "v" / "visit.json").read_text())
    candidates = record["result"]["candidates"]
    record["result"]["candidates"] = [candidates[0], candidates[0], candidates[1], candidates[0]]
    (tmp_path / "v" / "visit.json").write_text(json.dumps(record))
    visit = archived_visit(tmp_path / "v")
    names = [frame.name for frame in visit.frames]
    assert len(names) == 2 and len(set(names)) == 2


def test_labels_keep_a_reduced_copy_and_outlive_the_archive(archive, tmp_path: Path) -> None:
    store = Store(tmp_path / "selector")
    visit = archived_visit(archive / "visit-03")
    names = [frame.name for frame in visit.frames]
    store.label_visit(
        visit.key,
        visit.started_at,
        visit.posted,
        visit.paths(),
        {names[1]: GOOD, names[0]: BAD},
    )
    with Image.open(store.kept_frame("visit-03", names[1])) as kept:
        assert kept.width == KEPT_WIDTH
    assert not store.kept_frame("visit-03", names[2]).exists()

    for path in (archive / "visit-03").rglob("*"):
        if path.is_file():
            path.unlink()
    survivor = next(v for v in all_visits(archive, store) if v.key == "visit-03")
    assert not survivor.in_archive
    assert sorted(frame.name for frame in survivor.frames) == names[:2]
    assert survivor.posted == names[0]


def test_the_animals_looked_for_can_be_changed_to_ones_the_detector_knows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store(tmp_path / "selector")
    store.create()
    monkeypatch.delenv("PETCAM_PETS", raising=False)
    assert store.pet_classes() == ("cat", "dog")
    # The settings file's animals, less the class numbers only the camera understands.
    monkeypatch.setenv("PETCAM_PETS", "bird 17 dog")
    assert store.pet_classes() == ("bird", "dog")
    store.set_pet_classes(["bird", "cat"])
    assert store.pet_classes() == ("bird", "cat")
    assert json.loads((tmp_path / "selector" / "state.json").read_text())["pet_classes"] == [
        "bird",
        "cat",
    ]
    with pytest.raises(ValueError, match="does not know axolotl"):
        store.set_pet_classes(["axolotl"])
    assert store.pet_classes() == ("bird", "cat")


def test_a_state_file_that_is_not_text_is_moved_aside(tmp_path: Path) -> None:
    store = Store(tmp_path / "selector")
    store.create()
    (store.dir / "state.json").write_bytes(b"\xff\xfe\x00 not text")
    assert store.state() == {"active": "heuristic"}
    assert (store.dir / "state.json.unreadable").read_bytes() == b"\xff\xfe\x00 not text"


def test_the_switch_refuses_the_model_until_one_is_trained(archive, tmp_path: Path) -> None:
    store = Store(tmp_path / "selector")
    assert store.state()["active"] == "heuristic"
    with pytest.raises(ValueError, match="no trained model"):
        store.use("model")
    label_all(store, archive)
    train(store, featurizer())
    store.use("model")
    assert store.state()["active"] == "model"
    assert json.loads((tmp_path / "selector" / "state.json").read_text())["model"]
    store.use("heuristic")
    assert store.state()["active"] == "heuristic" and store.current_model() is not None


def test_training_needs_enough_labels(archive, tmp_path: Path) -> None:
    store = Store(tmp_path / "selector")
    visit = archived_visit(archive / "visit-00")
    store.label_visit(visit.key, 0, None, visit.paths(), {visit.frames[1].name: GOOD})
    with pytest.raises(NotEnoughLabels, match="have 1 and 0 from 1"):
        train(store, featurizer())


def test_training_learns_the_labels_and_reports_against_the_posted_frame(
    archive, tmp_path: Path
) -> None:
    store = Store(tmp_path / "selector")
    label_all(store, archive)
    meta = train(store, featurizer())
    assert (meta["good"], meta["bad"], meta["visits"]) == (12, 24, 12)
    assert meta["ratings"] == [24, 0, 12, 0]
    metrics = meta["metrics"]
    assert metrics["visits_with_a_choice"] == 12
    assert metrics["model_picks_good"] == 1.0 and metrics["auc"] == 1.0
    assert metrics["pairs"] == 24 and metrics["pairs_right"] == 1.0
    # Every posted frame was a dark one, and one frame in three is good.
    assert metrics["heuristic_picks_good"] == 0.0 and metrics["heuristic_picks_top"] == 0.0
    assert metrics["random_picks_good"] == pytest.approx(1 / 3)

    model = Model.load(store.current_model())
    bright, dark = tmp_path / "bright.jpg", tmp_path / "dark.jpg"
    Image.new("RGB", (64, 48), BRIGHT).save(bright)
    Image.new("RGB", (64, 48), DARK).save(dark)
    make = featurizer()
    scores = model.scores(make.features(make.describe([bright, dark])))
    assert scores[0] > 0.9 and scores[1] < 0.1
    assert store.model_meta()["name"] == meta["name"]


def test_the_built_in_picker_is_judged_on_its_own_pick_not_on_the_posted_frame(
    archive, tmp_path: Path
) -> None:
    """Once the model posts, the posted frame is the model's; the built-in scoring's pick is
    recorded beside it and is the one its test row is about."""
    store = Store(tmp_path / "selector")
    for visit in all_visits(archive, store):
        names = [frame.name for frame in visit.frames]
        labels = {name: GOOD if index == 1 else BAD for index, name in enumerate(names)}
        # The bright frame was posted; the built-in scoring had picked a dark one.
        store.label_visit(
            visit.key, visit.started_at, names[1], visit.paths(), labels, heuristic=names[0]
        )
    metrics = train(store, featurizer())["metrics"]
    assert metrics["heuristic_visits"] == 12 and metrics["heuristic_picks_good"] == 0.0


def test_every_training_run_is_kept_oldest_first_with_the_current_one_marked(
    archive, tmp_path: Path, monkeypatch
) -> None:
    store = Store(tmp_path / "selector")
    label_all(store, archive)
    stamps = iter(["20261101-000001", "20261101-000002"])
    monkeypatch.setattr("selector.train.time.strftime", lambda *_: next(stamps))
    first, second = train(store, featurizer()), train(store, featurizer())
    history = store.history()
    assert [run["name"] for run in history] == [first["name"], second["name"]]
    assert [run["current"] for run in history] == [False, True]
    assert history[0]["metrics"]["model_picks_good"] == 1.0
    assert App(store, archive, featurizer()).status()["history"] == history


def test_training_learns_which_of_two_good_frames_you_rated_higher(tmp_path: Path) -> None:
    # Every visit has a best (bright), a good (mid) and a bad (dark) frame; the mid one was posted.
    archive, store = tmp_path / "visits", Store(tmp_path / "selector")
    for number in range(12):
        make_visit(archive, f"visit-{number:02d}", [MID, BRIGHT, DARK])
    for visit in all_visits(archive, store):
        names = [frame.name for frame in visit.frames]
        store.label_visit(
            visit.key, 0, visit.posted, visit.paths(), dict(zip(names, [GOOD, BEST, BAD]))
        )
    metrics = train(store, featurizer())["metrics"]
    assert metrics["pairs"] == 36 and metrics["pairs_right"] == 1.0
    assert metrics["model_picks_top"] == 1.0
    # The posted frame was always good, never the best.
    assert metrics["heuristic_picks_good"] == 1.0 and metrics["heuristic_picks_top"] == 0.0


def test_labels_saved_as_words_by_the_first_version_read_as_ratings(tmp_path: Path) -> None:
    data = tmp_path / "selector"
    data.mkdir()
    (data / "labels.json").write_text(
        json.dumps({"visits": {"v": {"frames": {"a.jpg": "good", "b.jpg": "bad"}}}})
    )
    store = Store(data)
    assert store.labels()["visits"]["v"]["frames"] == {"a.jpg": GOOD, "b.jpg": BAD}
    assert store.rating_counts() == [1, 0, 1, 0]


def test_a_model_from_an_earlier_version_is_not_used(archive, tmp_path: Path) -> None:
    store = Store(tmp_path / "selector")
    label_all(store, archive)
    meta = train(store, featurizer())
    record = store.models_dir / f"{meta['name']}.json"
    record.write_text(json.dumps({**meta, "features": "something older"}))
    assert store.current_model() is None and store.outdated_model()
    with pytest.raises(ValueError, match="no trained model"):
        store.use("model")


def test_a_model_trained_before_the_last_features_were_added_is_still_used(
    archive, tmp_path: Path
) -> None:
    store = Store(tmp_path / "selector")
    label_all(store, archive)
    meta = train(store, featurizer())
    path = store.models_dir / f"{meta['name']}.npz"
    make = featurizer()
    visit = archived_visit(archive / "visit-00")
    features = make.features(make.describe([frame.path for frame in visit.frames]))
    # The same model as an earlier version would have saved it: without the measures' columns.
    whole = Model.load(path)
    known = len(whole.weights) - len(MEASURED)
    Model(whole.weights[:known], whole.bias, whole.mean[:known], whole.scale[:known]).save(path)
    (store.models_dir / f"{meta['name']}.json").write_text(
        json.dumps({**meta, "features": EARLIER_FEATURES[0]})
    )
    assert store.current_model() == path and not store.outdated_model()
    assert Model.load(path).scores(features) == pytest.approx(whole.scores(features))


def measured_archive(tmp_path: Path) -> Path:
    """One visit of a dark and a bright frame, of which the collector measured the bright one
    in full and the dark one only for sharpness."""
    archive = tmp_path / "visits"
    make_visit(archive, "v", [DARK, BRIGHT], new_format=True)
    (archive / "v" / "frames" / "measures.json").write_text(
        json.dumps(
            {
                "clip_at000.00s.jpg": {"sharpness": 40.0},
                "clip_at001.00s.jpg": {
                    "sharpness": 300.0,
                    "frame_change": 0.01,
                    "gimbal_turn": 0.0,
                    "box_speed": 0.02,
                },
            }
        )
    )
    return archive


def test_what_the_collector_measured_of_a_frame_becomes_its_last_features(tmp_path: Path) -> None:
    visit = archived_visit(measured_archive(tmp_path) / "v")
    make = featurizer()
    dark, bright = make.describe([frame.path for frame in visit.frames])
    assert dark.measures == {"sharpness": 40.0}
    assert set(bright.measures) == set(MEASURED)
    measured = make.features([dark, bright])[:, -len(MEASURED) :]
    assert measured[1] == pytest.approx(np.log1p([300.0, 0.01, 0.0, 0.02]))
    assert measured[0, 0] == pytest.approx(np.log1p(40.0)) and np.isnan(measured[0, 1:]).all()
    # A frame from before the collector measured any has none.
    Image.new("RGB", (64, 48), BRIGHT).save(tmp_path / "old.jpg")
    assert measures_of(tmp_path / "old.jpg") == {}
    assert np.isnan(make.features(make.describe([tmp_path / "old.jpg"]))[0, -len(MEASURED) :]).all()


def test_measures_are_kept_with_a_rated_frame_and_outlive_the_archive(tmp_path: Path) -> None:
    archive = measured_archive(tmp_path)
    store = Store(tmp_path / "selector")
    visit = archived_visit(archive / "v")
    labels = {visit.frames[0].name: BAD, visit.frames[1].name: GOOD}
    store.label_visit(visit.key, visit.started_at, visit.posted, visit.paths(), labels)
    before = {frame.name: measures_of(frame.path) for frame in visit.frames}
    for frame in visit.frames:
        frame.path.unlink()
    (archive / "v" / "frames" / "measures.json").unlink()
    kept = all_visits(archive, store)[0]
    assert {frame.name: measures_of(frame.path) for frame in kept.frames} == before
    # Rating it again from the kept copies loses nothing.
    store.label_visit(kept.key, kept.started_at, kept.posted, kept.paths(), labels)
    assert {frame.name: measures_of(frame.path) for frame in kept.frames} == before


def test_the_model_learns_from_a_measure_where_it_was_taken_and_ignores_it_elsewhere() -> None:
    rng = np.random.default_rng(0)
    count = 80
    visit = np.repeat(np.arange(count // 4), 4)
    ratings = np.tile([BAD, BAD, GOOD, GOOD], count // 4)
    noise = rng.normal(size=(count, 2))
    # Good frames barely changed from their neighbours; half the frames were never measured.
    change = np.where(ratings == GOOD, 0.01, 0.2) + rng.normal(scale=0.005, size=count)
    change[visit % 2 == 0] = np.nan
    never = np.full(count, np.nan)
    model = fit(np.column_stack([noise, change, never]), ratings, visit)
    assert model.weights[2] < 0 and model.weights[3] == 0.0
    still, moving, unmeasured, measured_now = model.scores(
        np.array(
            [
                [0.0, 0.0, 0.01, np.nan],
                [0.0, 0.0, 0.2, np.nan],
                [0.0, 0.0, np.nan, np.nan],
                [0.0, 0.0, np.nan, 5.0],
            ]
        )
    )
    assert still > 0.8 and moving < 0.2
    # Unmeasured, the frame is scored as if the measure were ordinary; a measure the model
    # never saw taken says nothing either.
    assert moving < unmeasured < still and measured_now == unmeasured


def photo(tmp_path: Path, name: str, box: Optional[PetBox], sharp: float = 100.0) -> Described:
    return Described(tmp_path / name, box, sharp)


def test_the_rule_of_thumb_prefers_a_large_central_whole_sharp_pet(tmp_path: Path) -> None:
    central = PetBox(0.3, 0.3, 0.7, 0.7, 0.95)
    described = [
        photo(tmp_path, "none", None),
        photo(tmp_path, "unsure", PetBox(0.3, 0.3, 0.7, 0.7, 0.4)),
        photo(tmp_path, "central", central),
        photo(tmp_path, "small", PetBox(0.45, 0.45, 0.55, 0.55, 0.95)),
        photo(tmp_path, "cut", PetBox(0.0, 0.3, 0.4, 0.7, 0.95)),
        photo(tmp_path, "blurred", central, sharp=10.0),
    ]
    none, unsure, best, small, cut, blurred = rule_of_thumb(described)
    assert none == 0.0 and unsure == 0.0
    assert best > max(small, cut, blurred) > 0.0


def test_near_copies_are_alike_pictures_with_the_pet_in_the_same_place(tmp_path: Path) -> None:
    here, there = PetBox(0.1, 0.1, 0.4, 0.4, 0.9), PetBox(0.6, 0.6, 0.9, 0.9, 0.9)
    described = [
        photo(tmp_path, "a", here),
        photo(tmp_path, "b", here),
        photo(tmp_path, "moved", there),
        photo(tmp_path, "empty", None),
        photo(tmp_path, "other scene", here),
    ]
    same = [1.0, 0.0]
    embeddings = np.array([same, same, same, same, [0.0, 1.0]])
    assert near_copies(described, embeddings, order=[0, 1, 2, 3, 4]) == {1: 0}


def test_proposals_start_from_the_rule_then_from_the_model(tmp_path: Path) -> None:
    box = PetBox(0.3, 0.3, 0.7, 0.7, 0.9)
    described = [photo(tmp_path, str(i), box) for i in range(4)] + [photo(tmp_path, "empty", None)]
    scores = np.array([0.9, 0.8, 0.3, 0.1, 0.95])
    order = [4, 0, 1, 2, 3]
    assert propose(described, scores, order, {1: 0}, trained=False) == {
        0: GOOD, 1: BAD, 2: BAD, 3: BAD, 4: BAD,
    }  # fmt: skip
    assert propose(described, scores, order, {}, trained=True) == {
        0: BEST, 1: GOOD, 2: OKAY, 3: BAD, 4: BAD,
    }  # fmt: skip
    assert set(propose(described[4:], scores[4:], [0], {}, trained=True).values()) == {BAD}


def test_auc_counts_how_often_good_outscores_bad() -> None:
    good = np.array([True, True, False, False])
    assert auc(np.array([0.9, 0.8, 0.2, 0.1]), good) == 1.0
    assert auc(np.array([0.9, 0.1, 0.8, 0.2]), good) == 0.5
    assert auc(np.array([0.5, 0.5]), np.array([True, True])) is None


def test_fit_weighs_a_rare_class_as_much_as_a_common_one() -> None:
    rng = np.random.default_rng(1)
    bad = rng.normal(0.0, 0.3, size=(200, 4))
    good = rng.normal(1.0, 0.3, size=(10, 4))
    ratings = np.array([BAD] * 200 + [GOOD] * 10)
    model = fit(np.vstack([bad, good]), ratings, visit=np.arange(210))
    assert model.scores(np.full((1, 4), 1.0))[0] > 0.8
    assert model.scores(np.zeros((1, 4)))[0] < 0.2


def test_rated_pairs_are_within_a_visit_and_carry_the_rating_gap() -> None:
    higher, lower, gap = rated_pairs(np.array([3, 0, 2, 2, 1]), np.array([0, 0, 0, 1, 1]))
    assert sorted(zip(higher, lower, gap)) == [(0, 1, 3.0), (0, 2, 1.0), (2, 1, 2.0), (3, 4, 1.0)]


def test_the_command_line_switches_and_reports(archive, tmp_path: Path, capsys) -> None:
    data = tmp_path / "selector"
    assert main(["--data-dir", str(data), "use", "model"]) == 1
    assert "no trained model" in capsys.readouterr().err
    store = Store(data)
    label_all(store, archive)
    train(store, featurizer())
    assert main(["--data-dir", str(data), "use", "model"]) == 0
    assert main(["--data-dir", str(data), "status"]) == 0
    out = capsys.readouterr().out
    assert "active selector: model" in out and "0 best, 12 good, 0 okay, 24 bad, 12 visits" in out


@pytest.fixture
def page(archive, tmp_path: Path) -> Iterator[Tuple[str, Store]]:
    store = Store(tmp_path / "selector")
    store.create()
    app = App(store, archive, featurizer())
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", store
    finally:
        server.shutdown()


def call(url: str, body: object = None) -> object:
    data = None if body is None else json.dumps(body).encode()
    with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=30) as response:
        payload = response.read()
    return (
        json.loads(payload) if response.headers["Content-Type"] == "application/json" else payload
    )


def test_the_page_labels_trains_compares_and_switches(page, archive) -> None:
    url, store = page
    assert b"<title>Pawst Cards</title>" in call(url + "/")
    visits = call(url + "/api/visits")
    assert len(visits) == 12 and not visits[0]["labelled"]

    first = call(f"{url}/api/visit?key={visits[0]['key']}")
    assert first["scored_by"] == "rule" and first["pets"] == 1
    # The bright frame is the only one with a pet: it comes first, boxed, proposed as good.
    top = first["frames"][0]
    assert top["name"] == "clip_at001.00s.jpg" and top["proposed"] == GOOD
    assert top["box"] == [0.3, 0.3, 0.7, 0.7] and top["rating"] is None
    assert [f["proposed"] for f in first["frames"][1:]] == [BAD, BAD]
    assert [f["copy_of"] for f in first["frames"]] == [None, None, first["frames"][1]["name"]]

    for visit in visits:
        detail = call(f"{url}/api/visit?key={visit['key']}")
        assert detail["posted"] == "clip_at000.00s.jpg"
        labels = {f["name"]: f["proposed"] for f in detail["frames"] if not f["copy_of"]}
        status = call(url + "/api/label", {"key": visit["key"], "labels": labels})
    assert status["ratings"] == [12, 0, 12, 0] and status["visits"] == 12
    again = call(f"{url}/api/visit?key=visit-00")
    assert again["labelled"] and again["frames"][0]["rating"] == GOOD
    assert [f["rating"] for f in again["frames"][1:]].count(None) == 1

    for visit in visits:
        detail = call(f"{url}/api/visit?key={visit['key']}")
        labels = {f["name"]: GOOD if f["box"] else BAD for f in detail["frames"]}
        status = call(url + "/api/label", {"key": visit["key"], "labels": labels})
    assert status["ratings"] == [24, 0, 12, 0]

    assert call(url + "/api/train", {})["training"] is True
    for _ in range(200):
        status = call(url + "/api/status")
        if not status["training"]:
            break
        threading.Event().wait(0.05)
    assert status["training_error"] is None
    assert status["model"]["metrics"]["model_picks_good"] == 1.0

    compared = call(url + "/api/compare")
    assert compared["looked_at"] == 12 and compared["same"] == 0
    entry = compared["entries"][0]
    assert entry["model"] == "clip_at001.00s.jpg" and entry["heuristic"] == "clip_at000.00s.jpg"
    assert entry["model_score"] > entry["heuristic_score"]
    status = call(url + "/api/vote", {"key": entry["key"], "vote": "model"})
    assert status["votes"] == {"model": 1, "heuristic": 0, "same": 0}
    # A vote is on one model's pick: it stops counting when another model is the current one.
    current = store.state()["model"]
    for kind in ("npz", "json"):
        (store.models_dir / f"another.{kind}").write_bytes(
            (store.models_dir / f"{current}.{kind}").read_bytes()
        )
    store.set_model("another")
    assert store.votes() == {}
    store.set_model(current)
    assert store.votes() == {entry["key"]: "model"}

    # A visit where you rated both picks lower than good is counted, not shown.
    detail = call(f"{url}/api/visit?key={entry['key']}")
    call(
        url + "/api/label",
        {"key": entry["key"], "labels": {f["name"]: BAD for f in detail["frames"]}},
    )
    again = call(url + "/api/compare")
    assert again["neither"] == 1 and len(again["entries"]) == len(compared["entries"]) - 1
    assert entry["key"] not in [e["key"] for e in again["entries"]]
    assert again["same"] == compared["same"] and again["looked_at"] == compared["looked_at"]

    assert call(url + "/api/use", {"selector": "model"})["active"] == "model"
    assert store.state()["active"] == "model"
    trained = call(f"{url}/api/visit?key=visit-00")
    assert trained["scored_by"] == "model" and trained["frames"][0]["score"] > 0.9
    with pytest.raises(urllib.error.HTTPError) as refused:
        call(url + "/api/label", {"key": "visit-00", "labels": {"clip_at000.00s.jpg": 7}})
    assert refused.value.code == 400


def test_the_page_serves_reduced_frames_and_refuses_paths_outside_the_archive(page) -> None:
    url, _ = page
    thumb = call(f"{url}/img?key=visit-00&name=clip_at000.00s.jpg&w=320")
    assert thumb[:2] == b"\xff\xd8" and len(thumb) < 20000
    for query in ("key=..&name=visit-00", "key=visit-00&name=../visit.json", "key=nope&name=x.jpg"):
        with pytest.raises(urllib.error.HTTPError) as refused:
            call(f"{url}/img?{query}")
        assert refused.value.code == 404
    with pytest.raises(urllib.error.HTTPError) as refused:
        call(url + "/api/use", {"selector": "model"})
    assert refused.value.code == 400


def test_an_older_visit_gets_extra_frames_cut_from_its_clips_once(tmp_path: Path) -> None:
    import subprocess

    from selector.frames import add_spread_frames

    make_visit(tmp_path, "old", [DARK])
    clips = tmp_path / "old" / "clips"
    clips.mkdir()
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10",
         "-t", "3", "-pix_fmt", "yuv420p", str(clips / "20261101_000000.mp4")],
        check=True,
    )  # fmt: skip
    assert add_spread_frames(tmp_path / "old", mirror=True, count=4) == 4
    assert add_spread_frames(tmp_path / "old", mirror=True, count=4) == 0
    visit = archived_visit(tmp_path / "old")
    assert [frame.source for frame in visit.frames].count("spread") == 4
    assert len(visit.frames) == 5 and visit.posted == "clip_at000.00s.jpg"


def test_a_visit_whose_clips_are_gone_keeps_the_frames_it_has(tmp_path: Path) -> None:
    from selector.frames import add_spread_frames

    make_visit(tmp_path, "old", [DARK, BRIGHT])
    assert add_spread_frames(tmp_path / "old", mirror=True) == 0
    assert len(archived_visit(tmp_path / "old").frames) == 2


def test_a_visit_the_collector_kept_back_is_marked_as_withheld(tmp_path: Path) -> None:
    make_visit(tmp_path, "kept-back", [DARK, BRIGHT], new_format=True)
    record_path = tmp_path / "kept-back" / "visit.json"
    record = json.loads(record_path.read_text())
    record["result"]["posted"] = False
    record_path.write_text(json.dumps(record))
    make_visit(tmp_path, "sent", [DARK, BRIGHT], new_format=True)
    assert archived_visit(tmp_path / "kept-back").withheld
    assert not archived_visit(tmp_path / "sent").withheld


# Photos that repeat earlier ones.


def shot(*numbers: float) -> Shot:
    whole = np.array(numbers, dtype=np.float32)
    return Shot(whole / np.linalg.norm(whole), None)


POT, DOOR, POST = shot(1, 0, 0), shot(0, 1, 0), shot(0, 0, 1)


def test_alikeness_weighs_the_pet_as_much_as_the_frame() -> None:
    a = Shot(POT.whole, DOOR.whole)
    b = Shot(POT.whole, POST.whole)
    assert alikeness(a, Shot(POT.whole, DOOR.whole)) == pytest.approx(1.0)
    assert alikeness(a, b) == pytest.approx(0.5)
    assert alikeness(a, POT) == pytest.approx(1.0)


def test_repeat_cost_is_nothing_for_familiar_views_and_everything_for_the_same_shot() -> None:
    assert repeat_cost(FAMILIAR - 0.1) == 0.0
    assert repeat_cost(SAME_SHOT) == pytest.approx(1.0)
    assert 0.0 < repeat_cost((FAMILIAR + SAME_SHOT) / 2) < 1.0
    assert repeat_cost(1.0, strictness=0.0) == 0.0


def test_pick_fresh_posts_a_visit_unlike_anything_earlier_as_it_is() -> None:
    picks = pick_fresh([[0.9, 0.6]], [[DOOR, POST]], [POT], post_at_least=0.5)
    assert picks[0].frame == 0 and picks[0].worth == pytest.approx(0.9)


def test_pick_fresh_takes_another_frame_or_holds_the_visit_back() -> None:
    swapped, held = pick_fresh([[0.9, 0.6], [0.95]], [[POT, DOOR], [POT]], [POT], post_at_least=0.5)
    assert swapped.frame == 1 and swapped.worth == pytest.approx(0.6)
    assert held.frame is None and held.alike == pytest.approx(1.0)


def test_pick_fresh_keeps_the_best_of_waiting_visits_that_look_alike() -> None:
    worse, better, other = pick_fresh(
        [[0.7], [0.9], [0.6]], [[POT], [POT], [DOOR]], [], post_at_least=0.5
    )
    assert worse.frame is None
    assert better.frame == 0
    assert other.frame == 0


def test_pick_fresh_never_chooses_a_frame_below_the_posting_score() -> None:
    (pick,) = pick_fresh([[0.9, 0.4]], [[POT, DOOR]], [POT], post_at_least=0.5)
    assert pick.frame is None


def test_fresh_picks_answers_the_collector_by_path(tmp_path: Path) -> None:
    """With the stand-in network, bright photos look alike and so do dark ones."""

    class Looks:
        def shots(self, paths: Sequence[Path]) -> List[Shot]:
            return [POT if brightness(path) > 0.5 else DOOR for path in paths]

    bright, bright_too, dark = (tmp_path / name for name in ("a.jpg", "b.jpg", "c.jpg"))
    Image.new("RGB", (8, 8), BRIGHT).save(bright)
    Image.new("RGB", (8, 8), BRIGHT).save(bright_too)
    Image.new("RGB", (8, 8), DARK).save(dark)
    answer = fresh_picks(
        {
            "visits": [
                [{"frame": str(bright_too), "score": 0.9}, {"frame": str(dark), "score": 0.7}],
                [{"frame": str(bright_too), "score": 0.9}],
                [{"frame": str(tmp_path / "gone.jpg"), "score": 0.9}],
            ],
            "earlier": [str(bright), str(tmp_path / "pruned.jpg")],
            "post_at_least": 0.5,
        },
        Looks(),
    )
    first, second, third = answer["picks"]
    assert first["frame"] == str(dark) and first["worth"] == pytest.approx(0.7)
    assert second["frame"] is None and second["alike"] == pytest.approx(1.0)
    assert third == {"frame": None, "worth": None, "alike": 0.0}
    json.dumps(answer)


def test_looker_embeds_the_pet_only_where_there_is_one(tmp_path: Path) -> None:
    bright, dark = tmp_path / "bright.jpg", tmp_path / "dark.jpg"
    Image.new("RGB", (64, 48), BRIGHT).save(bright)
    Image.new("RGB", (64, 48), DARK).save(dark)
    with_pet, without = Looker(BrightnessEmbedder(), BrightDetector()).shots([bright, dark])
    assert with_pet.pet is not None and without.pet is None
    assert Looker(BrightnessEmbedder(), BrightDetector()).shots([]) == []


def test_views_settings_are_read_with_defaults_and_changed_one_at_a_time(tmp_path: Path) -> None:
    store = Store(tmp_path / "data")
    app = App(store, tmp_path / "visits", featurizer())
    assert app.views() == {
        "map": None,
        "pose": None,
        "settings": {"mode": "fixed", "explore": 0.3, "since": None},
        "sweep": {
            "stops": 8,
            "yaw_low": 0.0,
            "yaw_high": 345.0,
            "pitch_low": 70.0,
            "pitch_high": 140.0,
        },
    }
    app.set_views({"mode": "learned"})
    app.set_views({"explore": 0.8})
    assert app.views()["settings"] == {"mode": "learned", "explore": 0.8, "since": None}
    app.set_views({"moved": True})
    assert app.views()["settings"]["since"] == pytest.approx(time.time(), abs=5)
    app.set_views({"moved": False})
    assert app.views()["settings"]["since"] is None
    for bad in ({"mode": "wild"}, {"explore": 2}, {"explore": "lots"}):
        with pytest.raises(ValueError):
            app.set_views(bad)
    assert app.views()["settings"]["mode"] == "learned"

    app.set_views({"stops": 4, "yaw_low": 30, "yaw_high": 135})
    settings = app.views()["settings"]
    assert (settings["stops"], settings["yaw_low"], settings["yaw_high"]) == (4, 30.0, 135.0)
    for bad in ({"stops": 40}, {"stops": 2.5}, {"yaw_low": 200}, {"pitch_high": 500}):
        with pytest.raises(ValueError):
            app.set_views(bad)
    assert app.views()["settings"]["stops"] == 4

    (store.dir / "views" / "map.json").write_text('{"cells": []}')
    (store.dir / "views" / "pose.json").write_text("not json")
    assert app.views()["map"] == {"cells": []}
    assert app.views()["pose"] is None


def test_the_page_refuses_changes_sent_by_another_web_page(page) -> None:
    # Any site open in the same browser can make it POST to this page's address. Browsers
    # name the sending site in Origin; only the page itself may change things.
    url, store = page
    body = json.dumps({"mode": "learned"}).encode()

    def post(origin: str) -> int:
        request = urllib.request.Request(url + "/api/views", data=body, headers={"Origin": origin})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status
        except urllib.error.HTTPError as error:
            return error.code

    assert post("http://elsewhere.example") == 403
    assert not (store.dir / "views" / "settings.json").exists()
    assert post(url) == 200
    assert json.loads((store.dir / "views" / "settings.json").read_text())["mode"] == "learned"


def test_the_page_refuses_a_body_too_long_to_be_one_of_its_own(page) -> None:
    url, _ = page
    connection = http.client.HTTPConnection(url[len("http://") :], timeout=5)
    try:
        connection.putrequest("POST", "/api/views")
        connection.putheader("Content-Length", "50000000")
        connection.endheaders()
        assert connection.getresponse().status == 400
    finally:
        connection.close()


def test_the_page_serves_nothing_from_beside_the_frames(page) -> None:
    url, store = page
    (store.dir / "labels.json").write_text("{}")
    for query in (
        "key=..&name=labels.json&w=0",
        "key=.&name=labels.json&w=0",
        "key=visit-00&name=..&w=0",
    ):
        with pytest.raises(urllib.error.HTTPError) as refused:
            call(f"{url}/img?{query}")
        assert refused.value.code == 404


def test_the_page_answers_only_to_an_address_it_was_given(page) -> None:
    # A page elsewhere can point a name of its own at this computer and so reach the selector
    # as if it were the same site; such a request carries that name as its Host.
    url, _ = page
    for method, path in (("GET", "/api/status"), ("POST", "/api/views")):
        connection = http.client.HTTPConnection(url[len("http://") :], timeout=5)
        try:
            connection.putrequest(method, path, skip_host=True)
            connection.putheader("Host", "rebound.example:8798")
            connection.putheader("Content-Length", "0")
            connection.endheaders()
            assert connection.getresponse().status == 403
        finally:
            connection.close()


def test_ratings_that_are_not_a_list_by_frame_are_refused(page) -> None:
    url, _ = page
    with pytest.raises(urllib.error.HTTPError) as refused:
        call(url + "/api/label", {"key": "visit-00", "labels": []})
    assert refused.value.code == 400


def test_a_rated_visit_gone_from_the_archive_keeps_the_built_in_pick(archive, tmp_path) -> None:
    store = Store(tmp_path / "selector")
    store.create()
    visit = archived_visit(archive / "visit-00")
    names = list(visit.paths())
    store.label_visit(
        "visit-00",
        visit.started_at,
        names[0],
        visit.paths(),
        {n: GOOD for n in names},
        heuristic=names[1],
    )
    [kept] = [v for v in all_visits(tmp_path / "no-archive", store) if v.key == "visit-00"]
    assert (kept.posted, kept.heuristic) == (names[0], names[1])


def test_frames_no_model_could_score_are_left_out_of_the_measurement() -> None:
    # Visit 0 holds every good frame, so the model that leaves it out has nothing good to learn
    # from and its frames get no score.
    data = Labelled(
        features=np.arange(8, dtype=float).reshape(8, 1),
        ratings=np.array([GOOD, BAD, BAD, BAD, BAD, BAD, BAD, BAD]),
        visit=np.array([0, 0, 1, 1, 2, 2, 3, 3]),
        posted=np.zeros(8, bool),
    )
    scores = held_out_scores(data, folds=4)
    assert np.isnan(scores[:2]).all()
    metrics = evaluate(data, scores)
    assert metrics["visits_with_a_choice"] == 0 and metrics["model_picks_good"] is None


def test_home_reports_what_has_been_collected_delivered_and_heard(page, archive) -> None:
    url, store = page
    home = call(url + "/api/home")
    # The directory of a picture of several visits is not a visit.
    assert home["collected"] == len(list(archive.glob("visit-*/visit.json")))
    assert home["collected_at"] is not None
    assert home["delivered"] == 0 and home["delivered_at"] is None
    assert home["camera_heard_at"] is None

    first, second = sorted(archive.glob("visit-*"))[:2]
    (first / "delivered").write_text("2026-11-01T06:25:00\n")
    (second / "delivered").write_text("2026-11-01T07:00:00 not posted\n")
    (archive / "batch-20261112" / "delivered").write_text("2026-11-12T07:00:00\n")
    (store.dir / "views").mkdir(exist_ok=True)
    (store.dir / "views" / "pose.json").write_text(json.dumps({"at": 1234.5, "pose": None}))
    home = call(url + "/api/home")
    assert home["delivered"] == 1 and home["delivered_at"] is not None
    assert home["camera_heard_at"] == 1234.5


def test_settings_are_saved_for_the_collector_and_nothing_is_saved_from_a_bad_request(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path / "data")
    store.create()
    app = App(store, tmp_path / "visits", featurizer())
    start = app.settings()
    assert start["saved"] == {"deliver_url_host": None}
    assert start["collector"] is None
    assert start["posting"] == {
        "post_at_least": 0.5,
        "repeat_strictness": 1.0,
        "repeat_hours": 24.0,
    }
    assert start["pets"] == ["cat", "dog"] and "bird" in start["known_pets"]

    app.set_settings(
        {
            "deliver": "webhook",
            "deliver_url": " https://hooks.example/T0/secret ",
            "deliver_every_minutes": 30,
            "subject": " Biscuit ",
            "timezone": "Europe/Paris",
            "keep_visits_days": 14,
            "post_at_least": 0.7,
            "repeat_strictness": 0,
        }
    )
    path = store.dir / "collector" / "settings.json"
    saved = json.loads(path.read_text())
    assert saved["deliver_url"] == "https://hooks.example/T0/secret"
    assert saved["subject"] == "Biscuit" and saved["deliver_every_minutes"] == 30.0
    assert path.stat().st_mode & 0o077 == 0
    shown = app.settings()
    assert shown["saved"]["deliver_url_host"] == "hooks.example"
    assert "secret" not in json.dumps(shown)
    assert shown["posting"]["post_at_least"] == 0.7
    assert store.state()["repeat_strictness"] == 0.0

    app.set_settings({"deliver_every_minutes": None, "keep_visits_days": None})
    saved = json.loads(path.read_text())
    assert saved["deliver_every_minutes"] is None and saved["keep_visits_days"] is None
    assert saved["deliver"] == "webhook"

    for bad in (
        {"subject": "Rex", "deliver": "pigeon"},
        {"subject": "Rex", "deliver_url": "file:///etc/passwd"},
        {"subject": "Rex", "timezone": "Mars/Olympus"},
        {"subject": "Rex", "keep_visits_days": 0},
        {"subject": "Rex", "post_at_least": 3},
        {"subject": "Rex", "pets": ["dragon"]},
        {"subject": ""},
    ):
        with pytest.raises(ValueError):
            app.set_settings(bad)
    assert json.loads(path.read_text())["subject"] == "Biscuit"


def test_changing_the_animals_swaps_the_detector_and_a_test_postcard_is_asked_for(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path / "data")
    store.create()
    made: List[Sequence[str]] = []

    def detector_for(classes: Sequence[str]) -> BrightDetector:
        made.append(list(classes))
        return BrightDetector()

    app = App(store, tmp_path / "visits", featurizer(), detector_for=detector_for)
    app.set_settings({"pets": ["cat", "dog"]})
    assert made == []
    app.set_settings({"pets": ["bird"]})
    assert made == [["bird"]] and store.pet_classes() == ("bird",)

    assert app.settings()["test_at"] is None
    app.set_settings({"test": True})
    assert app.settings()["test_at"] == pytest.approx(time.time(), abs=5)
    (store.dir / "collector" / "status.json").write_text(json.dumps({"problem": None}))
    assert app.settings()["collector"] == {"problem": None}


def test_with_a_passphrase_the_page_changes_nothing_without_it(archive, tmp_path: Path) -> None:
    store = Store(tmp_path / "selector")
    store.create()
    app = App(store, archive, featurizer())
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app, passphrase="tuna & biscuits"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"

    def post(header: Optional[str]) -> int:
        request = urllib.request.Request(
            url + "/api/settings",
            data=json.dumps({"subject": "Rex"}).encode(),
            headers={} if header is None else {"X-Pawst-Passphrase": header},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status
        except urllib.error.HTTPError as error:
            return error.code

    try:
        assert call(url + "/api/settings")["passphrase"] is True
        assert post(None) == 401
        assert post("tuna") == 401
        assert not (store.dir / "collector" / "settings.json").exists()
        assert post("tuna%20%26%20biscuits") == 200
        assert app.settings()["saved"]["subject"] == "Rex"
    finally:
        server.shutdown()


class FakeCameraHandler(http.server.BaseHTTPRequestHandler):
    """Answers as the camera's agent does, and keeps what it was sent."""

    state: dict = {}

    def _reply(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _patrol(self) -> dict:
        return {"plan": self.state["plan"], "default": self.state["plan"], "learned": True}

    def do_GET(self) -> None:
        if self.path == "/tuning":
            self._reply(200, {"values": self.state.get("tuning", {"gone_after": 60})})
        elif self.path == "/status":
            self._reply(
                200,
                {
                    "pose": self.state["pose"],
                    "tracking_enabled": True,
                    **self.state.get("status", {}),
                },
            )
        else:
            self._reply(200, self._patrol())

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.state["sent"].append((self.path, body))
        if self.path == "/tuning":
            if body.get("gone_after") == 5:
                self._reply(400, {"error": "gone_after must be from 10 to 600"})
                return
            self.state["tuning"] = body
            self._reply(200, {"values": body})
            return
        if self.path == "/patrol":
            if not body.get("stops") and self.state.get("refuse_empty"):
                self._reply(400, {"error": "no stops, no service"})
                return
            if "forget_missed" not in body:
                self.state["plan"] = body
        self._reply(200, self._patrol())

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def fake_camera() -> Iterator[Tuple[str, dict]]:
    FakeCameraHandler.state = {
        "pose": {"yaw": 123.44, "pitch": 98.76},
        "plan": {
            "home": {"yaw": 180.0, "pitch": 100.0},
            "stops": [{"yaw": 90.0, "pitch": 100.0}],
            "patrol_after": 600.0,
        },
        "sent": [],
    }
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeCameraHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", FakeCameraHandler.state
    finally:
        server.shutdown()


def test_the_page_turns_the_camera_and_keeps_where_it_points_as_home_or_a_stop(
    tmp_path: Path, fake_camera
) -> None:
    url, camera = fake_camera
    store = Store(tmp_path / "data")
    store.create()
    # The page learns where the camera is from the collector's status.
    (store.dir / "collector").mkdir()
    (store.dir / "collector" / "status.json").write_text(json.dumps({"camera": url}))
    app = App(store, tmp_path / "visits", featurizer())

    aim = app.aim()
    assert aim["problem"] is None and aim["tracking"] is True
    assert aim["pose"] == {"yaw": 123.44, "pitch": 98.76}

    app.set_aim({"move": {"yaw": -10, "pitch": 0}})
    app.set_aim({"tracking": True})
    assert camera["sent"] == [
        ("/move", {"yaw": -10.0, "pitch": 0.0}),
        ("/tracking", {"enabled": True}),
    ]
    assert app.views()["settings"]["mode"] == "fixed"

    app.set_aim({"keep": "home"})
    assert camera["plan"]["home"] == {"yaw": 123.4, "pitch": 98.8}
    assert app.views()["settings"]["mode"] == "manual"
    app.set_aim({"keep": "stop"})
    assert len(camera["plan"]["stops"]) == 2
    app.set_aim({"drop_stop": 0})
    assert camera["plan"]["stops"] == [{"yaw": 123.4, "pitch": 98.8}]

    for bad in ({"move": {"yaw": 400}}, {"tracking": "yes"}, {"keep": "sofa"}, {"drop_stop": 7}):
        with pytest.raises(ValueError):
            app.set_aim(bad)
    camera["refuse_empty"] = True
    with pytest.raises(ValueError, match="no stops, no service"):
        app.set_aim({"drop_stop": 0})


def test_aiming_says_why_when_the_camera_cannot_be_asked(tmp_path: Path) -> None:
    store = Store(tmp_path / "data")
    store.create()
    assert (
        "does not know where the camera is"
        in App(store, tmp_path / "visits", featurizer()).aim()["problem"]
    )
    app = App(store, tmp_path / "visits", featurizer(), camera_url="http://127.0.0.1:9")
    assert "did not answer" in app.aim()["problem"]
    with pytest.raises(ValueError, match="did not answer"):
        app.set_aim({"tracking": True})


def test_home_shows_what_waits_and_the_page_can_pause_and_send_now(tmp_path: Path) -> None:
    store = Store(tmp_path / "data")
    store.create()
    archive = tmp_path / "visits"
    (archive / "v1").mkdir(parents=True)
    (archive / "v1" / "clip.mp4").write_bytes(b"x" * 5000)
    app = App(store, archive, featurizer())
    assert app.home()["collector"] is None

    directory = store.dir / "collector"
    directory.mkdir()
    (directory / "status.json").write_text(
        json.dumps(
            {
                "at": 10.0,
                "waiting": 3,
                "next_delivery_at": 99.0,
                "settings": {"paused": False, "keep_visits_days": 3.0},
            }
        )
    )
    (directory / "log.json").write_text(
        json.dumps([{"at": 1.0, "level": "INFO", "text": f"line {n}"} for n in range(50)])
    )
    home = app.home()
    assert home["collector"]["waiting"] == 3 and home["collector"]["paused"] is False
    assert home["collector"]["keep_visits_days"] == 3.0
    assert len(home["log"]) == 40 and home["log"][-1]["text"] == "line 49"
    assert home["archive_bytes"] == 5000
    assert home["disk_free_bytes"] > 0

    app.set_settings({"paused": True, "send_now": True})
    saved = json.loads((directory / "settings.json").read_text())
    assert saved["paused"] is True
    assert saved["send_now_at"] == pytest.approx(time.time(), abs=5)
    with pytest.raises(ValueError):
        app.set_settings({"paused": "yes"})


def test_the_page_passes_the_camera_timings_on_and_says_what_the_camera_refused(
    tmp_path: Path, fake_camera
) -> None:
    url, camera = fake_camera
    store = Store(tmp_path / "data")
    store.create()
    app = App(store, tmp_path / "visits", featurizer(), camera_url=url)
    assert app.camera_timings() == {"problem": None, "values": {"gone_after": 60}}
    app.set_camera_timings({"gone_after": 45})
    assert camera["tuning"] == {"gone_after": 45}
    with pytest.raises(ValueError, match="from 10 to 600"):
        app.set_camera_timings({"gone_after": 5})
    unreachable = App(store, tmp_path / "visits", featurizer(), camera_url="http://127.0.0.1:9")
    assert "did not answer" in unreachable.camera_timings()["problem"]


def test_the_page_shows_what_the_camera_sees_from_its_stream(tmp_path: Path) -> None:
    store = Store(tmp_path / "data")
    store.create()
    asked = []

    def grab(stream: str, mirror: bool) -> bytes:
        asked.append((stream, mirror))
        return b"jpeg"

    stock = App(
        store, tmp_path / "visits", featurizer(), camera_url="http://camera.test:8765", grab=grab
    )
    assert stock.camera_picture() == b"jpeg"
    named = App(
        store,
        tmp_path / "visits",
        featurizer(),
        mirror=False,
        stream_url="rtsp://elsewhere.test/cam",
        grab=grab,
    )
    named.camera_picture()
    own = App(
        store,
        tmp_path / "visits",
        featurizer(),
        camera_url="http://camera.test:8765",
        grab=grab,
        stream_login=("pets", "p@ss/word"),
    )
    own.camera_picture()
    assert asked == [
        ("rtsp://admin:admin@camera.test:554/live", True),
        ("rtsp://elsewhere.test/cam", False),
        ("rtsp://pets:p%40ss%2Fword@camera.test:554/live", True),
    ]
    with pytest.raises(ValueError, match="does not know where the camera is"):
        App(store, tmp_path / "visits", featurizer(), grab=grab).camera_picture()


def test_a_still_that_cannot_be_cut_says_so_without_the_streams_address() -> None:
    from selector.annotate import grab_still

    with pytest.raises(ValueError) as refused:
        grab_still("rtsp://admin:secret@127.0.0.1:9/live", True)
    assert "video" in str(refused.value)
    assert "secret" not in str(refused.value)


def test_the_page_tells_whether_the_camera_runs_the_code_on_disk(
    tmp_path: Path, fake_camera
) -> None:
    import importlib.util

    from selector.annotate import AGENT_PACKAGE, code_fingerprint

    # The page's sum is the agent's own, for the agent in this repository.
    spec = importlib.util.spec_from_file_location("agent_version", AGENT_PACKAGE / "version.py")
    agents = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agents)
    assert code_fingerprint(AGENT_PACKAGE) == agents.fingerprint(AGENT_PACKAGE)

    url, camera = fake_camera
    store = Store(tmp_path / "data")
    store.create()
    package = tmp_path / "petcam"
    package.mkdir()
    (package / "agent.py").write_text("print('one')\n")
    app = App(store, tmp_path / "visits", featurizer(), camera_url=url, agent_package=package)
    here = code_fingerprint(package)
    # An agent from before the fingerprint does not report one.
    assert app.camera_code() == {"problem": None, "here": here, "camera": None}
    camera["status"] = {"code": here}
    assert app.camera_code()["camera"] == here
    (package / "agent.py").write_text("print('two')\n")
    assert app.camera_code()["here"] != here
    unreachable = App(store, tmp_path / "visits", featurizer(), camera_url="http://127.0.0.1:9")
    assert "did not answer" in unreachable.camera_code()["problem"]


def test_saving_the_sweep_ranges_has_the_camera_try_all_of_them_again(
    tmp_path: Path, fake_camera
) -> None:
    url, camera = fake_camera
    store = Store(tmp_path / "data")
    store.create()
    app = App(store, tmp_path / "visits", featurizer(), camera_url=url)
    app.set_views({"explore": 0.5})
    assert camera["sent"] == []
    app.set_views({"yaw_low": 30, "yaw_high": 300})
    assert camera["sent"] == [("/patrol", {"forget_missed": True})]

    unreachable = App(store, tmp_path / "visits", featurizer(), camera_url="http://127.0.0.1:9")
    unreachable.set_views({"yaw_low": 45})
    assert unreachable.views()["settings"]["yaw_low"] == 45


def test_the_camera_is_told_the_animals_and_the_page_says_when_it_could_not_be(
    tmp_path: Path, fake_camera
) -> None:
    url, camera = fake_camera
    store = Store(tmp_path / "data")
    store.create()
    app = App(store, tmp_path / "visits", featurizer(), camera_url=url)
    app.set_settings({"pets": ["bird", "dog"]})
    assert ("/pets", {"pets": ["bird", "dog"]}) in camera["sent"]
    assert app.settings()["camera_pets_problem"] is None

    unreachable = App(store, tmp_path / "visits", featurizer(), camera_url="http://127.0.0.1:9")
    unreachable.set_settings({"pets": ["cat"]})
    assert store.pet_classes() == ("cat",)
    assert "did not answer" in unreachable.settings()["camera_pets_problem"]


def test_a_host_name_allowed_in_capitals_is_matched_as_browsers_send_it() -> None:
    from selector.annotate import own_address

    assert own_address("mymac.example.net:8798", also=["MyMac.Example.net"])
    assert not own_address("other.example.net:8798", also=["MyMac.Example.net"])
