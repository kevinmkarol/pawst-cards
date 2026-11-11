# ABOUTME: Entry point for the on-camera agent: creates the detection and recorder nodes, feeds
# ABOUTME: detections to the controller, drives the gimbal over CAN, and serves the control API.

import argparse
import logging
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Optional, Sequence

from petcam.api import ControlApi
from petcam.capture import CaptureWatchdog, Remedy
from petcam.clock import Clock, restore_saved_time, save_time
from petcam.controller import Controller
from petcam.detections import DEFAULT_PETS, class_ids, parse_invoke
from petcam.gaze import GazeLog
from petcam.gimbal import PITCH_ID, YAW_ID, Gimbal
from petcam.mqtt import MqttClient
from petcam.patrol import Patrol, Plans, Pose
from petcam.pets import Pets
from petcam.tuning import Tuning
from petcam.recorder import SaveNodeRecorder
from petcam.sscma import (
    CAMERA_NODE_ID,
    MODEL_NODE_ID,
    SAVE_NODE_ID,
    create_model_message,
    create_save_message,
    created_by_request,
    in_topic,
    out_topic,
)

log = logging.getLogger("petcam")

# How often the current time is saved for the next boot, in seconds.
SAVE_TIME_EVERY_SECONDS = 300.0
# Longest the recovery command may run before it is abandoned.
RECOVERY_TIMEOUT_SECONDS = 300
# Seconds between questions to the motors about where the camera points.
POSE_EVERY_SECONDS = 0.2
# The kernel's name for the current boot.
BOOT_ID_FILE = Path("/proc/sys/kernel/random/boot_id")
# Restarts the capture service and builds the camera and stream nodes again.
DEFAULT_RECOVER_COMMAND = "/etc/init.d/S91sscma-node restart && /usr/local/bin/start_pipeline.sh"


def pose(text: str) -> Pose:
    """A pose given on the command line as yaw,pitch in degrees."""
    try:
        yaw, pitch = (float(part) for part in text.split(","))
        return Pose.from_dict({"yaw": yaw, "pitch": pitch})
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected yaw,pitch in degrees, yaw 0 to 360 and pitch 0 to 180, got {text!r}"
        )


# Where the camera rests, and the stops it sweeps when nothing has happened for a while: the
# middle of the gimbal's turn, tilted a little down, and up to a quarter turn to either side. They
# know nothing about the room; --home and --patrol say where to look in yours.
DEFAULT_HOME = Pose(yaw=180, pitch=100)
DEFAULT_PATROL = [
    Pose(yaw=90, pitch=100),
    Pose(yaw=135, pitch=100),
    Pose(yaw=225, pitch=100),
    Pose(yaw=270, pitch=100),
]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="reCamera pet cam agent")
    parser.add_argument("--broker", default="127.0.0.1", help="MQTT broker host of sscma-node")
    parser.add_argument("--can", default="can0", help="SocketCAN interface for the gimbal")
    parser.add_argument(
        "--api-host",
        default="0.0.0.0",
        help="address the control API listens on; it has no authentication, so bind it to a "
        "trusted network",
    )
    parser.add_argument("--api-port", type=int, default=8765)
    parser.add_argument(
        "--api-allow-host",
        nargs="*",
        default=[],
        metavar="NAME",
        help="host names the camera is reached by, beyond numeric addresses, names without "
        "dots and names ending .local, .lan, .home.arpa, .internal or .ts.net; requests "
        "under any other name are refused",
    )
    parser.add_argument(
        "--gone-after",
        type=float,
        default=60.0,
        help="seconds without a pet that end a visit",
    )
    parser.add_argument(
        "--max-visit",
        type=float,
        default=600.0,
        help="longest visit in seconds; a pet that stays longer is posted once and then ignored "
        "until it moves",
    )
    parser.add_argument(
        "--score",
        type=float,
        default=0.45,
        help="detection score threshold; a change takes effect when the camera restarts",
    )
    parser.add_argument(
        "--confirm-sightings",
        type=int,
        default=3,
        help="detections within 2 s needed before a visit starts (filters one-frame false "
        "positives)",
    )
    parser.add_argument(
        "--pets",
        nargs="+",
        default=list(DEFAULT_PETS),
        help="animals to follow and record, each a COCO name the camera's model knows (cat, "
        "dog, bird, horse, sheep, cow, elephant, bear, zebra, giraffe) or a class number",
    )
    parser.add_argument(
        "--slice",
        type=int,
        default=10,
        help="seconds per recorded clip file; a change takes effect when the camera restarts",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("/userdata/petcam/visits.json"),
        help="where finished, uncollected visits are remembered across restarts",
    )
    parser.add_argument(
        "--home",
        type=pose,
        default=DEFAULT_HOME,
        help="yaw,pitch the camera returns to after a visit (default 180,100)",
    )
    parser.add_argument(
        "--patrol",
        type=pose,
        nargs="*",
        default=DEFAULT_PATROL,
        help="yaw,pitch stops swept when idle; give none to disable",
    )
    parser.add_argument(
        "--home-after", type=float, default=90.0, help="quiet seconds before returning home"
    )
    parser.add_argument(
        "--patrol-after", type=float, default=600.0, help="idle seconds at home before a sweep"
    )
    parser.add_argument("--dwell", type=float, default=25.0, help="seconds at each patrol stop")
    parser.add_argument(
        "--plan-file",
        type=Path,
        default=Path("/userdata/petcam/plan.json"),
        help="where a home and patrol set through the API are kept; they replace --home, "
        "--patrol and --patrol-after until the API is told to go back to those",
    )
    parser.add_argument(
        "--pets-file",
        type=Path,
        default=Path("/userdata/petcam/pets.json"),
        help="where animals chosen through the API are kept; they replace --pets until the "
        "API is told to go back to those",
    )
    parser.add_argument(
        "--tuning-file",
        type=Path,
        default=Path("/userdata/petcam/tuning.json"),
        help="where timings changed through the API are kept; they replace --gone-after, "
        "--max-visit, --confirm-sightings, --home-after and --dwell until the API is told to "
        "go back to those",
    )
    parser.add_argument(
        "--gaze-file",
        type=Path,
        default=Path("/userdata/petcam/gaze.json"),
        help="where the record of which way the camera has looked is kept",
    )
    parser.add_argument(
        "--time-file",
        type=Path,
        default=Path("/userdata/petcam/last_time"),
        help="where the time is saved for the next boot; the camera has no battery clock",
    )
    parser.add_argument(
        "--stall-after",
        type=float,
        default=90.0,
        help="seconds without a detection frame before the capture service is restarted",
    )
    parser.add_argument(
        "--recover-command",
        default=DEFAULT_RECOVER_COMMAND,
        help="shell command that restarts capture when frames stop or the camera node is "
        "replaced; --recover-command= to never restart it",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    try:
        args.pets = class_ids(args.pets)
    except ValueError as error:
        parser.error(str(error))
    return args


def boot_id() -> Optional[str]:
    try:
        return BOOT_ID_FILE.read_text().strip() or None
    except OSError:
        return None


def recover_capture(command: str) -> None:
    """Run the recovery command and log how it went."""
    log.warning("restarting capture: %s", command)
    try:
        result = subprocess.run(
            ["sh", "-c", command],
            capture_output=True,
            text=True,
            timeout=RECOVERY_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        log.error("capture recovery did not finish: %s", error)
        return
    output = (result.stdout + result.stderr).strip()[-500:]
    log.log(
        logging.INFO if result.returncode == 0 else logging.ERROR,
        "capture recovery exited with status %d: %s",
        result.returncode,
        output,
    )


def stop_on_sigterm() -> None:
    """Make `kill` end the agent the way Ctrl-C does, so it stops a recording in progress and
    saves the visit on the way out instead of leaving the recorder running."""

    def stop(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    # UTC, like the camera's clip names and visit ids, whatever time zone the camera is set to.
    logging.Formatter.converter = time.gmtime
    if restore_saved_time(args.time_file):
        log.info("clock was behind the time saved before this boot; set to the saved time")
    stop_on_sigterm()

    clock = Clock()
    gimbal = Gimbal(args.can)
    mqtt = MqttClient(args.broker)
    recorder = SaveNodeRecorder(publish=mqtt.publish)
    gaze = GazeLog(args.gaze_file)
    controller = Controller(
        gimbal=gimbal,
        recorder=recorder,
        gone_after_seconds=args.gone_after,
        move_interval_seconds=0.3,
        pets=args.pets,
        state_file=args.state_file,
        confirm_sightings=args.confirm_sightings,
        max_visit_seconds=args.max_visit,
        patrol=Patrol(
            home=args.home,
            stops=args.patrol,
            home_after_seconds=args.home_after,
            patrol_after_seconds=args.patrol_after,
            dwell_seconds=args.dwell,
        ),
        boot_id=boot_id(),
        gaze=gaze,
    )
    plans = Plans(controller.patrol, args.plan_file)
    if plans.learned:
        log.info("following the saved plan: %s", plans.status()["plan"])
    pets = Pets(controller, args.pets_file)
    if pets.changed:
        log.info("following the saved choice of animals: %s", pets.status()["pets"])
    tuning = Tuning(controller, args.tuning_file)
    if tuning.changed:
        log.info("using the saved timings: %s", tuning.status()["values"])
    watchdog = CaptureWatchdog(now=time.monotonic(), restart_after_seconds=args.stall_after)
    lock = threading.Lock()
    # Set while the agent itself is rebuilding the pipeline.
    recovering = threading.Event()

    def on_message(topic: str, payload: bytes) -> None:
        if topic == out_topic(CAMERA_NODE_ID):
            # The camera node exists before the agent starts, so its creation now is someone
            # else's doing: the factory Node-RED flow builds its own, at 1080p,
            # whenever sscma-node restarts. Rebuilding the pipeline puts ours back.
            if created_by_request(payload) and not recovering.is_set():
                log.warning("camera node was created by something else: %s", payload[:120])
                with lock:
                    watchdog.camera_replaced()
            return
        if topic != out_topic(MODEL_NODE_ID):
            return
        frame = parse_invoke(payload.decode(errors="replace"))
        if frame is None:
            log.info("model node: %s", payload[:200].decode(errors="replace"))
            return
        with lock:
            if stopping.is_set():
                # The recorder has been stopped for the last time; a frame must not start it.
                return
            if watchdog.frame(time.monotonic()):
                # Frames are arriving for the first time or again after a gap. The recorder
                # may have been restarted meanwhile, or left running by an agent that was
                # killed, so it is told again what it should be doing.
                log.info("detection frames arriving")
                recorder.set_recording(controller.presence.in_visit)
            was_in_visit = controller.presence.in_visit
            controller.observe(frame, now=clock.now())
            if controller.presence.in_visit != was_in_visit:
                log.info("visit %s", "started" if controller.presence.in_visit else "ended")

    def create_nodes() -> None:
        # Creating a node that already exists is answered with "Node already exists" and is
        # harmless. sscma-node drops requests that reach it before it is ready, so the watchdog
        # has these sent again for as long as no frames arrive.
        mqtt.publish(in_topic(SAVE_NODE_ID), create_save_message(slice_seconds=args.slice).encode())
        mqtt.publish(in_topic(MODEL_NODE_ID), create_model_message(args.score, 0.25).encode())

    stopping = threading.Event()

    def watch_pose() -> None:
        # The only reader of the CAN socket, so the motors' answers reach no one else.
        while not stopping.is_set():
            try:
                yaw = gimbal.read_angle(YAW_ID)
                pitch = gimbal.read_angle(PITCH_ID) if yaw is not None else None
            except OSError:
                yaw = pitch = None
            with lock:
                controller.pointing(
                    Pose(yaw=yaw, pitch=pitch) if yaw is not None and pitch is not None else None,
                    clock.now(),
                )
            stopping.wait(POSE_EVERY_SECONDS)

    def health() -> dict:
        with lock:
            return {"capture": watchdog.status(time.monotonic())}

    mqtt.on_message = on_message
    try:
        mqtt.connect()
    except OSError as error:
        # The boot script starts the agent again in a few seconds.
        log.error("broker %s is not answering: %s", args.broker, error)
        gimbal.close()
        return 1
    mqtt.subscribe(out_topic(MODEL_NODE_ID))
    mqtt.subscribe(out_topic(CAMERA_NODE_ID))
    create_nodes()

    try:
        api = ControlApi(
            controller=controller,
            gimbal=gimbal,
            host=args.api_host,
            port=args.api_port,
            slice_seconds=args.slice,
            lock=lock,
            health=health,
            plans=plans,
            names=args.api_allow_host,
            tuning=tuning,
            pets=pets,
        )
        api.start()
    except OSError as error:
        log.error("the API cannot listen on %s:%d: %s", args.api_host, args.api_port, error)
        mqtt.close()
        gimbal.close()
        return 1
    threading.Thread(target=watch_pose, name="pose", daemon=True).start()
    log.info("agent running: api on :%d, broker %s, can %s", api.port, args.broker, args.can)

    time_saved_at = time.monotonic()
    try:
        while not mqtt.lost.is_set():
            time.sleep(1.0)
            with lock:
                before = clock.now()
                stepped = clock.resync()
                if stepped:
                    log.warning("system clock stepped by %+.0f s", stepped)
                    controller.clock_stepped(stepped, now=before)
                was_in_visit = controller.presence.in_visit
                controller.tick(now=clock.now())
                if was_in_visit and not controller.presence.in_visit:
                    log.info("visit ended")
                remedy = watchdog.check(time.monotonic())
            if remedy == Remedy.RESTART and args.recover_command:
                recovering.set()
                try:
                    recover_capture(args.recover_command)
                finally:
                    recovering.clear()
            if remedy is not None:
                create_nodes()
            if time.monotonic() - time_saved_at >= SAVE_TIME_EVERY_SECONDS:
                time_saved_at = time.monotonic()
                save_time(args.time_file, time.time())
                with lock:
                    gaze.save()
        # Without the broker there are no frames and no way to drive the recorder. Exiting
        # lets the boot script start a fresh agent, which connects again.
        log.error("broker connection lost; exiting to be started again")
        return 1
    except KeyboardInterrupt:
        return 0
    except OSError as error:
        # A message that could not be sent because the broker had just gone.
        log.error("broker connection lost: %s; exiting to be started again", error)
        return 1
    finally:
        stopping.set()
        api.stop()
        with lock:
            gaze.save()
            try:
                controller.shutdown(clock.now())
            except OSError as error:
                log.warning("could not stop the recorder on the way out: %s", error)
        save_time(args.time_file, time.time())
        mqtt.close()
        gimbal.close()


if __name__ == "__main__":
    sys.exit(main())
