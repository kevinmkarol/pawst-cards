# Development

How Pawst Cards works inside, for building on it or changing it. Setup and everyday use are in
the [README](README.md); every setting is explained in `petcam.env.example`.

## How it works

```text
reCamera Gimbal                               always-on computer              command
+---------------------------+                 +--------------------------+    or webhook
| sscma-node: camera, RTSP, |                 | collector (every 15 s)   |
|   YOLO model, recorder    |  HTTP :8765     |  fetch finished visits   |
| petcam agent: visits,     | <-------------- |  download clips          |     photo +
|   tracking, patrol, API   | --------------> |  frames at best moments  | --> caption
+---------------------------+   clips, JSON   |  rank, archive, deliver  |
                                              +--------------------------+
```

- A **visit** starts when the camera's detections confirm a pet and ends after a quiet period,
  or at a cap so a sleeping pet is posted once. The gimbal follows the largest one while
  recording, returns to a home pose afterwards, and patrols a list of stops when idle.
- The **collector** polls the camera; the camera never calls out. For each finished visit it
  downloads the clips, cuts candidate frames, ranks them, delivers the winner, archives the
  visit and deletes it from the camera.
- The **selector** (optional, on the collector computer) is the web page, the trained photo
  model and the tools around them.

| | On the camera | On the collector computer |
| --- | --- | --- |
| Code | `camera/pipeline/` (capture and RTSP at boot), `camera/agent/` (visits, tracking, API), `camera/tailscale/` (optional) | `collector/`, `collector/macos/` and `collector/linux/` (service installs), `selector/`, `streaming/` (live view) |
| Installed to | `/userdata/petcam`, `/usr/local/bin/start_pipeline.sh`, `/etc/init.d/S9[56]petcam_*`, `/etc/petcam_rtsp.conf` | Wherever you run it; `collector/macos/deploy.sh` puts a copy in `~/petcam/repo` and the `PETCAM_*` values in a launchd plist |
| Keeps | Uncollected clips in `/userdata/Videos`. In `/userdata/petcam`: their visits (`visits.json`), what the page or API set (`plan.json`, `tuning.json`, `pets.json`), where the camera has looked (`gaze.json`), the time last saved (`last_time`), the flags (`agent.args`) and the log (`agent.log`) | Every visit in `~/petcam/visits/<visit id>/`, logs in `~/petcam/logs/`, the selector's data in `~/petcam/selector` |
| Deletes | A visit's clips once collected. The recorder deletes its oldest clips if storage fills, and the agent forgets its oldest visits once about 2,000 wait uncollected | Visits older than `PETCAM_KEEP_VISITS_DAYS`, once a day, if set. A visit is roughly 100 MB |
| Sends out | Nothing | The chosen photo and caption, through your command or webhook |

The camera keeps UTC (clip names, visit ids, agent log). Captions and photo names use
`PETCAM_TIMEZONE`.

## The agent

Standard-library Python 3.9, beside the camera's stock software. Flags go in
`PETCAM_AGENT_ARGS` (rerun `camera/agent/install_service.sh` after changing them);
`python3 -m petcam.agent --help` lists them all. The ones that matter: `--gone-after` (quiet
seconds that end a visit, 60), `--max-visit` (600), `--home-after` (90), `--patrol-after`
(600), `--dwell` (25), `--score` (0.45), `--pets` (COCO names or class numbers, `cat dog`),
`--home` and `--patrol` (poses as `yaw,pitch`). A plan, timings or animals set through the page
or the API are kept in `/userdata/petcam` and win over the flags until `{"default": true}` is
posted. A new `--score` or `--slice` takes effect when the camera restarts.

Its API on port 8765 has no login:

| Request | Effect |
| --- | --- |
| `GET /status` | Tracking state, visit in progress, last detection, `pose` (the angles the motors report), patrol position, and `capture`: whether detection frames are arriving and how often a restart of capture was called for. |
| `GET /visits` | Finished visits, oldest first: start, end, sightings (time, box, score), `gaze` (where the camera pointed), clips, and `total`. `?limit=N` returns the first N. |
| `GET /clips/<name>` | Download a clip. `Range: bytes=N-` resumes one. |
| `DELETE /visits/<id>` | Forget a collected visit and delete its clips, except those another visit needs. |
| `POST /tracking {"enabled": bool}` | Gimbal tracking on or off. Patrol pauses while off. |
| `POST /move {"yaw": deg, "pitch": deg}` | Turn by this many degrees from where it is (positive yaw is the camera's own left) and switch tracking off. |
| `GET /patrol` | The home and stops in use (`plan`), the ones from the flags (`default`), whether the plan was set through the API (`learned`), and the poses the gimbal was sent to but stopped short of (`missed`, each with `asked` and `reached`). |
| `POST /patrol {"home": {"yaw", "pitch"}, "stops": [...], "patrol_after": seconds}` | Rest and sweep by this plan; kept across restarts, but a running collector puts its own back unless the **Camera** tab says *Set by hand*. `{"default": true}` goes back to the flags; `{"forget_missed": true}` empties `missed`. |
| `GET /tuning` | Timings that can change while the agent runs (`values`), the flags' (`default`), their `limits`, and whether any was set through the API (`changed`). |
| `POST /tuning {"gone_after", "max_visit", "confirm_sightings", "home_after", "dwell"}` | Change any of them; kept across restarts. `{"default": true}` goes back to the flags. |
| `GET /pets` | The animals followed (`pets`), the ones from `--pets` (`default`), whether they were set through the API (`changed`) and every name the model knows (`known`). |
| `POST /pets {"pets": ["cat", "bird"]}` | Follow these from now on, the largest in view first; kept across restarts. `{"default": true}` goes back to the flag. |
| `GET /gaze?since=YYYYMMDDHH` | Seconds the camera faced each direction while waiting, by hour (UTC) and by direction rounded to 5 degrees. |

`/status` also reports a fingerprint of `camera/agent/petcam/*.py`; the page's Home tab compares
it with its own copy of the repository to say whether the camera needs a deploy.

## The collector

**Settings.** `PETCAM_*` values are starting values. The page saves changes to
`~/petcam/selector/collector/settings.json` (posting thresholds to
`~/petcam/selector/state.json`), which the
collector reads at every poll and which win over the environment; delete a key to go back. The
delivery command and stage directory are never changed from the page, because a command is a
program run on the collector's computer.

**The archive.** Each visit lands in `~/petcam/visits/<visit id>/` with `clips/`, `frames/`
(with `measures.json`), the chosen photo and `visit.json`: the visit, which frame won and why.
`started_at` there is the camera's own time; add `clock_offset` for the real one.

**Delivery.**

- A command gets `{photo}`, `{caption}` and `{visit_dir}` as one argument each, with no shell.
- A webhook gets a multipart POST with `caption`, `visit` (the JSON record) and `photo`, plus
  each clip as `clip` when clips are on.
- In code, `collect_visit` in `collector/collector/process.py` calls a `Sender`, a function of
  one `Delivery`; `collector/collector/deliver.py` holds the two built-in ones.
- With `PETCAM_DELIVER_EVERY_MINUTES` set, `outbox.py` holds visits in
  `~/petcam/visits/outbox.json` and sends several as one `Delivery`: the photo is a grid of up
  to 12, `{visit_dir}` is a `batch-<time>` directory, and the `visit` record is
  `{"visits": [...]}`. A lone waiting visit is delivered unchanged.

**Which animals.** One list says what counts as the pet. `PETCAM_PETS` in `petcam.env` is the
starting value: `camera/agent/install_service.sh` gives it to the agent as `--pets`, and the
selector's detector reads it from its environment. The page's Settings tab then changes both,
the detector's in `state.json` (`pet_classes`) and the camera's through `POST /pets`; the same
goes for `./pawst-selector pets bird`, except that the camera is not told. The agent follows
only what the stock YOLO11n model reports, which is the COCO classes; a class number in the
list (`0` is a person) reaches the camera but means nothing to the selector. Ratings are kept
when the list changes; train again afterwards. `PETCAM_SUBJECT` is only the word in captions.

## How a photo is picked

**Built-in picker** (`collector/collector/scoring.py`): the frame where the detector saw the
pet largest and nearest the centre, marked down for blur. A following camera is turning much of
the time, so three things keep smeared frames out, each a module in `collector/collector/`:

- **Still moments** (`motion.py`). A moment is worth its prominence times its stillness: half
  as much for a degree of camera turn within half a second, half again for a box crossing 6% of
  the picture's width a second.
- **Bursts** (`burst.py`). Every frame within half a second of a moment is cut, about fifteen,
  and the clearest kept; one half a second off has to be a third clearer. This also absorbs the
  slack between the detector's clock and the clip's.
- **Blur measured where it matters** (`measures.py`). Sharpness is edge contrast inside the
  pet's box at half the 5 MP frame. Movement is how much that region differs from the
  neighbouring frames, as a share of its contrast. Both are judged against fixed marks, so a
  visit with nothing sharp has nothing that passes for sharp.

The marks they judge by (`AMPLE_SHARPNESS`, `CHANGE_HALVES_AT`, `TURN_HALVES_AT_DEGREES`,
`SPEED_HALVES_AT_WIDTHS`, `FAR_FRAME_COST`) are constants at the top of those modules, set by eye
on one camera in one room.

**Trained model** (`selector/`). Once the selector's tools have run, the collector also takes
12 frames at even steps through each visit, and once a model exists it records both picks in
`visit.json` (`heuristic_frame`, `model_frame`, each candidate's `model_score`).

- **Rating.** A local detector (Faster R-CNN, about 170 MB) boxes the pet; near-copies are
  hidden. Rated frames are copied to `~/petcam/selector/frames`, so ratings outlive the keep
  time.
- **Training.** Each frame becomes a CLIP embedding of the frame and of the pet's crop (CLIP is
  about 350 MB), measurements of the box, and the blur measures above. A linear model is fitted
  so that good and best frames score high and, within a visit, higher-rated frames outscore
  lower-rated ones. The report is for held-out visits, against the built-in picker and chance.
- **Switching.** `use model` or `use heuristic` writes `~/petcam/selector/state.json`, read for
  every visit. If the model fails, the built-in picker takes that visit and the log says why.
- **Posting bar.** While the model picks, a visit whose best frame scores below
  `post_at_least` (0.5) is archived, not delivered, and marked "not posted". 0 posts everything.
- **Repeats** (`collector/collector/repeats.py`, `selector/selector/shots.py`). With batched
  delivery, waiting visits are weighed against photos delivered in the last `repeat_hours` (24)
  and each other, by DreamSim likeness (about 1.9 GB) of the frame and the crop. A frame loses
  nothing below 0.62 alike and a whole point at 0.86, scaled by `repeat_strictness` (1; 0 turns
  it off). A visit whose first choice repeats posts another frame still worth 0.5, or is held
  back with the reason under `repeat` in `visit.json`. If the check fails, everything is sent.

## How it chooses where to look

The agent reads the motors' pose five times a second. Between visits it totals the time spent
facing each direction (`GET /gaze`); during a visit the direction goes into the visit's record.

Every ten minutes the collector redraws a map in cells 10 degrees on a side: the hours each
cell was in view while the camera waited, and the worth of the visits that began there. A visit
is worth how far its best photo scored above 0.5, and a week-old visit counts half.
Worth divided by hours is what a direction pays for being watched.

Home is the pose whose view pays best, counting only what is fairly certain, and moves only on
clear evidence. Stops are chosen one after another to add most to what home already sees, with
a bonus for cells little is known about; the page's slider sets that bonus and how soon an idle
camera sets off (every twenty minutes at one end, every two at the other). A change of the
settings reaches the camera through `POST /patrol` at the next poll; otherwise a plan is replaced
at most every three hours, so each sweep is watched long enough to learn from. Settings are in
`~/petcam/selector/views/settings.json`; sweeps default to 8 stops within yaw 0 to 345 and
pitch 70 to 140.

The motors stop silently at the end of their travel. On leaving a pose it rested at, the agent
compares it with the pose the motors report; one more than 5 degrees off is listed as `missed`
until it is reached, the agent restarts or the sweep limits are saved again. The collector
draws the limits in to where the gimbal stopped, sends a new plan at its next redrawing, and
marks on the map the cells no pose within the limits can see; the page draws those grey and
leaves them out of "Worth watching next".

## Recovery

| Event | What happens |
| --- | --- |
| Camera loses power | Everything starts again at boot. A visit in progress is lost; finished ones are kept. |
| Camera boots with no internet | The agent sets the clock to the time it last saved (within 5 minutes of the cut). When NTP answers, visits since boot are marked with the correction. |
| Camera or collector offline for days | Visits wait on the camera. The collector fetches the backlog a few at a time and logs `camera unreachable` once, then hourly. |
| Capture stops | The agent asks for its nodes again every 15 s; after 90 s it restarts `sscma-node` and rebuilds the pipeline, then after 5 min, 10 min, up to hourly. |
| Factory Node-RED flow takes the camera node | It asks for 1080p whenever `sscma-node` restarts. The agent rebuilds the pipeline at the resolution in `start_pipeline.sh`. |
| MQTT broker restarts | The agent exits and is started again, pipeline first. |
| Agent is killed | It stops the recorder and saves the visit in progress, and is started again 10 s later. |
| Download interrupted | The collector keeps what arrived and asks for the rest next time. |
| Delivery fails | The visit stays on the camera and is retried after 30 s, 1 min, 2 min, up to every 10 min. With batched delivery it waits in the archive, safe from the keep time, retried every 5 min. |
| Delivered but the camera could not be told | The visit is cleared on the next pass without sending again. |
| Clips gone or unreadable | Without clips the camera drops the visit; unreadable ones are retried every 10 min. |
| Power cut or full disk while saving state | State files are replaced whole, so the previous version survives. What could not be saved is kept in memory and logged. |
| `tailscaled` exits | Its boot script starts it again within seconds. |

Not handled: an expired Tailscale key, and a `mosquitto` that does not come back. The factory
Node-RED flow runs a second detection model; moving `/etc/init.d/S92node-red` into
`/etc/init.d/disabled/` frees that load if you never use the factory dashboard.

## Security model

The trust model and what counts as a vulnerability are in [SECURITY.md](SECURITY.md).

- The agent and the page refuse changes a browser sends on behalf of another site, and any
  request under a host name that is not an address, a name without dots, or one ending
  `.local`, `.lan`, `.home.arpa`, `.internal` or `.ts.net`. Allow another with
  `--api-allow-host <name>` in `PETCAM_AGENT_ARGS`, or as the page's `--host`.
- `--api-host <address>` limits the agent's API to one of the camera's addresses.
  `install_service.sh` checks the agent at `127.0.0.1:8765` and reports no answer otherwise.
- The page never sends the webhook's address back to the browser, only its host name.
- The RTSP login is kept on the camera in `/etc/petcam_rtsp.conf`, readable by root only.
- tailscaled runs in userspace mode on the camera: inbound connections work, but the camera
  cannot start tailnet connections. They arrive as if from the camera itself, so what listens
  only on 127.0.0.1 there, the MQTT broker included, is open to the tailnet; the pipeline
  sends the RTSP login through that broker at every start. `camera/tailscale/install.sh` pins the version and SHA-256.

## Live view

`streaming/live.sh` runs mediamtx on `streaming/mediamtx.yml` with the stream's login from
`petcam.env` filled in. `streaming/tunnel.sh` forwards local port 8555 to the camera's RTSP port,
which sidesteps macOS Local Network privacy blocking Homebrew binaries. The `live` path flips
the picture and encodes audio to Opus for the browser; `cam` is the raw stream. The sensor's
picture is mirrored and its mirror option does nothing, which is also why the collector flips
frames (`--no-mirror` turns that off).

## Tests and tooling

```sh
(cd camera/agent && uv sync && uv run pytest)
(cd collector && uv sync && uv run pytest)
(cd selector && uv sync && uv run pytest)
uvx pre-commit run --all-files
```

- Tests run against real code: the collector's talk to an in-process copy of the agent's API,
  and the agent's MQTT tests talk to the camera when `PETCAM_BROKER` is set.
- Run the collector's tests in a shell that has not loaded `petcam.env`.
- Changes under `selector/selector/*.py` need the page restarted; `page.html` is read on every
  request.
- Nothing of Seeed's is in this repository. `camera/factory_images/` and `camera/stock/` are
  ignored by git as places for your own copies of their images and factory flow. Upgrading the
  OS is `upgrade.sh start <ota.zip>` on the camera; it is A/B, so the previous version stays.

Conventions for changes are in [CONTRIBUTING.md](CONTRIBUTING.md).
