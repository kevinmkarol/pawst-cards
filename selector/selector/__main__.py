# ABOUTME: Command line for the selector: annotate (the web page), train, rank and fresh (used by
# ABOUTME: the collector), use (the model or the built-in scoring), pets and status.

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from selector.store import SELECTORS, Store, default_data_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pawst-selector",
        description="Rate your pet's photos, train a model on the ratings, and choose what picks the "
        "photo that gets posted.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="labels, kept frames and models (default: $PETCAM_SELECTOR_DIR or ~/petcam/selector)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    annotate = commands.add_parser("annotate", help="serve the selector page")
    annotate.add_argument("--archive-dir", type=Path, default=Path.home() / "petcam" / "visits")
    annotate.add_argument(
        "--host",
        default="127.0.0.1",
        help="address to listen on, and a name the page may be reached by",
    )
    annotate.add_argument("--port", type=int, default=8798)
    annotate.add_argument(
        "--camera",
        default=os.environ.get("PETCAM_CAMERA_AGENT_URL"),
        help="camera agent base URL, for the Camera tab (default: $PETCAM_CAMERA_AGENT_URL, "
        "or what the running collector uses)",
    )
    annotate.add_argument(
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="flip the camera's still and the extra frames cut for older visits left to right, as "
        "the collector does",
    )

    commands.add_parser("train", help="train a model on the labels and make it the current one")

    rank = commands.add_parser("rank", help="score photos with the current model, as JSON")
    rank.add_argument("photos", nargs="+", type=Path)

    commands.add_parser(
        "fresh",
        help="pick, for visits waiting to be posted, frames unlike the photos posted earlier; "
        "reads a JSON request on standard input and prints the picks as JSON",
    )

    use = commands.add_parser("use", help="choose what picks the posted photo")
    use.add_argument("selector", choices=SELECTORS)

    pets = commands.add_parser(
        "pets",
        help="say which animals to look for in the photos (default: $PETCAM_PETS or cat dog)",
    )
    pets.add_argument("classes", nargs="+", help="COCO class names, such as: cat dog bird")

    commands.add_parser("status", help="show the active selector, the model and label counts")
    return parser


def status_text(store: Store) -> str:
    bad, okay, good, best = store.rating_counts()
    lines = [
        f"active selector: {store.state()['active']}",
        f"looking for: {', '.join(store.pet_classes())}",
        f"ratings: {best} best, {good} good, {okay} okay, {bad} bad, "
        f"{len(store.labels()['visits'])} visits",
    ]
    meta = store.model_meta()
    if store.outdated_model():
        lines.append(
            "model: made by an earlier version, or its file is gone; train again to use it"
        )
    elif meta is None:
        lines.append("model: none trained yet")
    else:
        from selector.train import describe

        lines.append(describe(meta))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    store = Store(args.data_dir)
    if args.command == "status":
        print(status_text(store))
        return 0
    if args.command == "use":
        try:
            store.use(args.selector)
        except ValueError as error:
            print(error, file=sys.stderr)
            return 1
        print(f"the collector now posts the {args.selector}'s pick, starting with the next visit")
        return 0
    if args.command == "pets":
        try:
            store.set_pet_classes(args.classes)
        except ValueError as error:
            print(error, file=sys.stderr)
            return 1
        print(
            f"looking for: {', '.join(store.pet_classes())}; train again if a model exists, "
            "and restart the page if it is running"
        )
        return 0
    if args.command == "rank":
        # The collector must not wait on, or fail for want of, a network: the backbone's
        # weights are already in the local cache from training.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from selector.detect import PetDetector
        from selector.embed import ClipEmbedder
        from selector.features import Featurizer
        from selector.model import Model

        path = store.current_model()
        if path is None:
            print("no trained model", file=sys.stderr)
            return 1
        # Where the pet is in each frame is kept: `fresh` asks about the same frames later.
        featurizer = Featurizer(
            ClipEmbedder(None), PetDetector(store.dir / "detections", store.pet_classes())
        )
        scores = Model.load(path).scores(featurizer.features(featurizer.describe(args.photos)))
        print(json.dumps({"model": path.stem, "scores": [round(float(s), 4) for s in scores]}))
        return 0

    if args.command == "fresh":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from selector.detect import PetDetector
        from selector.shots import Looker, ShotEmbedder, fresh_picks

        looker = Looker(
            ShotEmbedder(store.dir / "dreamsim", store.dir / "shots"),
            PetDetector(store.dir / "detections", store.pet_classes()),
        )
        print(json.dumps(fresh_picks(json.load(sys.stdin), looker)))
        return 0

    from selector.detect import PetDetector
    from selector.embed import ClipEmbedder
    from selector.features import Featurizer

    store.create()
    featurizer = Featurizer(
        ClipEmbedder(store.dir / "embeddings"),
        PetDetector(store.dir / "detections", store.pet_classes()),
    )
    if args.command == "train":
        from selector.train import NotEnoughLabels, describe, train

        try:
            print(describe(train(store, featurizer)))
        except NotEnoughLabels as error:
            print(error, file=sys.stderr)
            return 1
        return 0
    from selector.annotate import FACTORY_STREAM_LOGIN, serve

    serve(
        store,
        args.archive_dir,
        featurizer,
        args.host,
        args.port,
        args.mirror,
        detector_for=lambda classes: PetDetector(store.dir / "detections", classes),
        # From the environment, not a flag: command lines can be read by other users.
        passphrase=os.environ.get("PETCAM_PAGE_PASSPHRASE") or None,
        camera_url=args.camera or None,
        # From the environment for the same reason: the address may hold a password.
        stream_url=os.environ.get("PETCAM_CAMERA_STREAM_URL") or None,
        stream_login=(
            os.environ.get("PETCAM_RTSP_USER") or FACTORY_STREAM_LOGIN[0],
            os.environ.get("PETCAM_RTSP_PASSWORD") or FACTORY_STREAM_LOGIN[1],
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
