# ABOUTME: Which animals the agent follows, changeable while it runs through the API and kept
# ABOUTME: across restarts. The ones the agent was started with are what it goes back to.

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

from petcam.controller import Controller
from petcam.detections import COCO_NAMES, class_ids, class_names
from petcam.files import remove, save_whole


class Pets:
    """Reads and changes the animals a running controller follows. Ones set through the API
    replace the ones the agent was started with until the API is told to go back, and are kept
    in `state_file`."""

    def __init__(self, controller: Controller, state_file: Optional[Path] = None) -> None:
        self._controller = controller
        self._state_file = state_file
        self.default: Tuple[int, ...] = controller.pets
        self.changed = False
        saved = self._load()
        if saved:
            controller.pets = saved
            self.changed = True

    def status(self) -> dict:
        return {
            "pets": class_names(self._controller.pets),
            "default": class_names(self.default),
            "known": list(COCO_NAMES),
            "changed": self.changed,
        }

    def set(self, pets: Any) -> None:
        """Follow `pets`, a list of names or class numbers; ValueError, with nothing changed,
        when it is not one the camera's model knows."""
        if not isinstance(pets, list):
            raise ValueError("pets must be a list of animal names")
        self._controller.pets = class_ids(pets)
        self.changed = True
        self._save(class_names(self._controller.pets))

    def reset(self) -> None:
        """Back to the animals the agent was started with."""
        self._controller.pets = self.default
        self.changed = False
        if self._state_file is not None:
            remove(self._state_file)

    def _load(self) -> Tuple[int, ...]:
        if self._state_file is None:
            return ()
        try:
            return class_ids(json.loads(self._state_file.read_text()))
        except (OSError, ValueError, TypeError):
            return ()

    def _save(self, names: List[str]) -> None:
        if self._state_file is not None:
            save_whole(self._state_file, json.dumps(names))
