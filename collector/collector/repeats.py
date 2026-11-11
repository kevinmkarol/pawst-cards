# ABOUTME: Keeps a day's postcards varied: before waiting visits are delivered, the trained
# ABOUTME: selector swaps or holds back photos that look like ones posted in the hours before.

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from collector.deliver import Delivery
from collector.files import write_whole
from collector.selector import MODEL, ModelSelector

log = logging.getLogger("collector")

# Written in a visit's archive directory once its photo has been delivered, handed to the
# outbox or held back, saying which.
DELIVERED_FILE = "delivered"


@dataclass(frozen=True)
class Sifted:
    # The visits still to be delivered, in the order they were given.
    kept: List[Delivery]
    # The photo each kept visit shows, to count as posted once the delivery has gone out.
    shown: List[Path]


def _record(delivery: Delivery) -> Optional[dict]:
    try:
        record = json.loads(delivery.visit_json.read_text())
        return record if isinstance(record.get("result"), dict) else None
    except (OSError, ValueError, AttributeError):
        return None


def _frames(record: Optional[dict]) -> List[Tuple[Path, float, float]]:
    """The frames of a visit the trained model chose from, each with its score and its
    moment; none for a visit the built-in scoring picked, whose photo is posted as it is."""
    if record is None or record["result"].get("chosen_by") != MODEL:
        return []
    frames = []
    for entry in record["result"].get("candidates") or []:
        try:
            if entry["model_score"] is not None:
                frames.append((Path(entry["frame"]), float(entry["model_score"]), entry["at"]))
        except (KeyError, TypeError, ValueError):
            continue
    return frames


class RepeatSifter:
    """Sifts the visits about to be delivered. While the trained model is the active selector,
    each visit's photo is weighed against the photos posted in the hours before and against
    the other waiting visits: a visit whose chosen frame repeats them gets another of its
    frames when it has one good enough, and is otherwise held back, recorded as not posted.
    Visits the built-in scoring picked, and every visit when the selector fails, pass as
    they are."""

    def __init__(self, selector: ModelSelector) -> None:
        self._selector = selector

    def window_seconds(self) -> float:
        """How long a posted photo goes on counting as posted."""
        return self._selector.repeat_seconds()

    def sift(self, batch: Sequence[Delivery], earlier: Sequence[Path]) -> Sifted:
        records = [_record(delivery) for delivery in batch]
        shown = [
            Path(record["result"]["best_frame"]) if record is not None else delivery.photo
            for delivery, record in zip(batch, records)
        ]
        unchanged = Sifted(list(batch), shown)
        if self._selector.active() != MODEL or self._selector.repeat_strictness() <= 0:
            return unchanged
        frames = [_frames(record) for record in records]
        judged = [index for index, visit in enumerate(frames) if visit]
        if not judged:
            return unchanged
        # A photo that goes out whatever happens counts as posted for the others.
        fixed = [shown[index] for index in range(len(batch)) if index not in judged]
        picks = self._selector.fresh(
            [[(frame, score) for frame, score, _ in frames[index]] for index in judged],
            [*earlier, *fixed],
        )
        if picks is None:
            return unchanged
        kept: List[Delivery] = []
        kept_shown: List[Path] = []
        swapped = held = 0
        for index, delivery in enumerate(batch):
            pick = picks[judged.index(index)] if index in judged else None
            if pick is None:
                kept.append(delivery)
                kept_shown.append(shown[index])
            elif pick.get("frame") is None:
                held += 1
                self._hold_back(delivery, records[index], pick)
            else:
                frame = Path(pick["frame"])
                if frame != shown[index]:
                    swapped += 1
                    self._swap(delivery, records[index], frames[index], frame, pick)
                kept.append(delivery)
                kept_shown.append(frame)
        if swapped or held:
            log.info(
                "repeats: of %d waiting visits, %d got another frame and %d were held back",
                len(batch),
                swapped,
                held,
            )
        return Sifted(kept, kept_shown)

    def _swap(
        self,
        delivery: Delivery,
        record: dict,
        frames: Sequence[Tuple[Path, float, float]],
        frame: Path,
        pick: dict,
    ) -> None:
        score, at = next((score, at) for path, score, at in frames if path == frame)
        shutil.copyfile(frame, delivery.photo)
        result = record["result"]
        result["repeat"] = {"first_choice": result["best_frame"], "alike": pick.get("alike")}
        result.update(best_frame=str(frame), best_frame_at=at, score=score)
        write_whole(delivery.visit_json, json.dumps(record, indent=1))

    def _hold_back(self, delivery: Delivery, record: dict, pick: dict) -> None:
        result = record["result"]
        result["posted"] = False
        result["repeat"] = {"held_back": True, "alike": pick.get("alike")}
        write_whole(delivery.visit_json, json.dumps(record, indent=1))
        (delivery.visit_dir / DELIVERED_FILE).write_text(
            datetime.now().astimezone().isoformat() + " not posted, repeats an earlier photo\n"
        )
        delivery.photo.unlink(missing_ok=True)
