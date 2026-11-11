# Pawst Cards

Postcards from your pet. A camera on a gimbal follows your pet, records each visit, and when the
visit is over a collector picks the frame where it looked best and sends it, with a caption,
wherever you like. You never watch a feed; the pet sends word.

Video is handled on a [Seeed reCamera Gimbal](https://www.seeedstudio.com/reCamera-Gimbal-2002w-64GB-p-6403.html)
with its stock OS and YOLO model. A small collector on any always-on computer fetches the
finished visits. Clips stay in the house; only the chosen photo leaves.

The camera's model sees cats, dogs, birds, horses, sheep, cows, elephants, bears, zebras and
giraffes. Other pets, such as rabbits, hamsters and reptiles, are not seen and cannot be
followed.

Setup means editing one settings file and running scripts in a terminal; there is no installer
or app. Allow an evening. Read [Security](#security) before putting the camera on a network you
share. How it all works inside is in [DEVELOPMENT.md](DEVELOPMENT.md).

## What you need

| Part | Notes |
| --- | --- |
| Seeed reCamera Gimbal | reCamera OS 0.2.5. No reflash: the agent runs beside the stock software. |
| 12 V barrel supply | On USB-C alone the gimbal motors do not move. |
| An always-on computer | Runs the collector: Python 3.12, [uv](https://docs.astral.sh/uv/) and `ffmpeg`. Install scripts are included for macOS; a systemd unit for Linux is included but untried. |
| Somewhere to post | Any command line tool that takes a file and a message, or a URL that accepts a multipart POST. |
| Tailscale (optional) | Only if the camera and the collector are on different networks. |
| About 2.5 GB of disk | For the selector page and its photo-rating model, downloaded the first time the page starts. Built on an Apple-silicon Mac; untried elsewhere. |

## Setup

1. **Configure.** Copy `petcam.env.example` to `petcam.env` and fill it in. Every setting is
   explained in the file, and git ignores your copy. The ones you must set:
   - `PETCAM_CAMERA_HOST` and `PETCAM_CAMERA_AGENT_URL`: the camera's address. On the same
     network use `recamera.local` or an address reserved on your router.
   - `PETCAM_DELIVER` with either `PETCAM_DELIVER_COMMAND` or `PETCAM_DELIVER_URL`: where
     photos go. Leave `PETCAM_STAGE_DIR` empty unless your command needs it.
   - `PETCAM_PETS`: the animals to follow, if not `cat dog`.
   - `PETCAM_RTSP_PASSWORD`: your own password for the camera's video stream, in place of the
     factory's `admin`.
   - `PETCAM_COLLECTOR_HOST` and `PETCAM_COLLECTOR_HOME`, if the collector will run on another
     Mac.

2. **Camera access.** The scripts reach the camera over SSH and use `sudo` there without
   being asked for a password. Give the camera your SSH key and allow that, once (the password
   is the one you set in the camera's dashboard, or `recamera` if you never did). If you
   have no SSH key yet, make one first with `ssh-keygen -t ed25519`.

   ```sh
   ssh-copy-id recamera@recamera.local
   ssh -t recamera@recamera.local "echo 'recamera ALL=(ALL) NOPASSWD: ALL' | sudo tee /etc/sudoers.d/zz_recamera_nopasswd"
   ```

3. **Camera pipeline.** Run `camera/pipeline/install.sh`. The camera now starts its video
   stream at every boot. The password from step 1 takes effect when the camera is next
   restarted, as does a later change, after running the script again.

4. **Agent.** Run `camera/agent/deploy.sh`, then `camera/agent/install_service.sh`.

5. **Aim it.** The built-in home and patrol poses know nothing about your room. The easy way
   is the selector page's **Camera** tab, once step 6 is done
   ([Everyday use](#everyday-use)). By hand:

   ```sh
   curl -d '{"yaw": 20, "pitch": -5}' http://recamera.local:8765/move
   curl http://recamera.local:8765/status
   curl -d '{"enabled": true}' http://recamera.local:8765/tracking
   ```

   The first turns it from where it is, the second shows where it points as `pose`, and the
   third has it follow the pet again.

   Write the angles as `--home` and `--patrol` in `PETCAM_AGENT_ARGS` and run
   `install_service.sh` again. A home or stops set on the **Camera** tab win over these
   flags. The motors stop silently at their limits: if `pose` does not show the angle you
   asked for, the gimbal cannot reach it.

6. **Collector.** On the collector computer, install [uv](https://docs.astral.sh/uv/) and
   `ffmpeg` (`brew install uv ffmpeg` on a Mac). Try one pass that collects the waiting visits
   but sends nothing:

   ```sh
   (cd collector && uv sync && set -a && . ../petcam.env && set +a &&
       uv run python -m collector.service --once --dry-run)
   ```

   Those visits are archived in `~/petcam/visits` and cleared from the camera, so they are
   never posted; open a visit's folder to see the photo it would have sent. Then keep the
   collector running:
   - **Mac:** `collector/macos/deploy.sh --local` for the Mac you are sitting at, or
     `collector/macos/deploy.sh` for the one named by `PETCAM_COLLECTOR_HOST`, which needs
     `uv` and `ffmpeg` too.
   - **Linux:** `collector/linux/petcam-collector.service`; the commands are at the top of the
     file.
   - **Anywhere else:** run `uv run python -m collector.service` with the values of
     `petcam.env` in its environment, under whatever keeps services alive there.

7. **Check it works.** Run `./pawst-selector annotate` on the collector computer and open
   `http://127.0.0.1:8798/`. The **Home** tab says whether the camera is answering, visits are
   being collected and photos are being sent, and what to do about each; visits from the dry
   run count as sent there. Press *Send a test postcard* on **Settings** to try your delivery.
   Then get the pet in front of the camera: about a minute after it leaves, the photo arrives.

   If the collector runs on another Mac, set `PETCAM_PAGE_HOST` in `petcam.env` to an address
   of that Mac, run `collector/macos/deploy.sh` again and open the page at that address.

**Different networks?** Run `camera/tailscale/install.sh`, which downloads Tailscale for the
camera, verifies its checksum and installs it. Enrol the camera with the command it prints,
**disable key expiry** for the camera in the Tailscale admin console, and put the camera's
tailnet address in both camera settings in `petcam.env`. `camera/tailscale/check.sh` verifies
it. The Tailscale account and Tailscale on the collector computer are yours to set up.

## Everyday use

Everything after setup happens on the selector page: `./pawst-selector annotate`, then
`http://127.0.0.1:8798/`. To keep the page running on a Mac, set `PETCAM_PAGE_HOST` in
`petcam.env` and run `collector/macos/deploy.sh` again.

| Tab | What it is for |
| --- | --- |
| **Home** | Health of the camera, collector and deliveries. Visits waiting to be sent, *Send them now*, *Pause postcards*, disk use, the collector's latest log lines. |
| **Photos** | Rate photos and train a model on your ratings (below). |
| **Camera** | Turn the camera with a still of what it sees, set its home and patrol stops, and change when a visit ends and how long the camera waits before moving. |
| **Settings** | Where postcards go and how often, with *Send a test postcard*. The animals to follow, the pet's name, the time zone, how long visits are kept, and which photos are posted. |

Changes take effect within 15 seconds, with no reinstall, and win over the values in
`petcam.env`. The delivery command and its stage directory are the exception: they change only
in `petcam.env`, followed by the collector's install script.

### Teach it your taste (optional)

Out of the box the collector picks the frame where the pet is largest, most central and least
blurred. To have it pick the ones you would:

1. **Rate.** On **Photos**, mark frames bad, okay, good or best. The page proposes a rating for
   each; correct the ones it gets wrong.
2. **Train.** Press *Train now*. It needs at least 10 frames rated good or best and 10 rated lower,
   from 5 visits; a few hundred is where it starts to mean something. The report compares the
   model with the built-in picker on visits it was not trained on. If the collector was
   installed with `collector/macos/deploy.sh`, run that once more now, so the installed copy
   has what the model needs.
3. **Switch.** In **Settings**, choose the trained model. From then on a visit with nothing
   you would call good is archived but not posted, and with batched delivery a photo that
   looks like one sent in the last day is held back. Both are adjustable in **Settings**.

Everything runs on the collector computer; nothing is uploaded. The same actions exist on the
command line: `./pawst-selector train`, `use model`, `use heuristic`, `status`.

### Let it choose where to look (optional)

The collector keeps a map of which directions yield good photos. On the **Camera** tab set
*Home and sweep* to *Learned* and the camera picks its own home and patrol stops from that map.

- It needs several visits with good photos before the home moves, so expect a few days.
- Set *Sweep limits* to what your gimbal can reach. If a sweep sends it further than it can
  turn, the limits are drawn in by themselves and the page says so.
- Press *I moved the camera* after moving it, to start the map again.
- With `PETCAM_KEEP_VISITS_DAYS` set, it learns only from that many days; a week or more is best.

### Live view (optional)

For checking the picture in a browser. On a Mac, `brew install mediamtx ffmpeg`, then:

```sh
streaming/tunnel.sh &
streaming/live.sh
```

The first is an SSH tunnel to the camera's stream; with both running, open
`http://127.0.0.1:8889/live/`.

## Security

Nothing here has a login by default. It is built for a home network or a tailnet you trust.

- **Change the camera's passwords.** SSH (user `recamera`) with `passwd` on the camera; the
  video stream with `PETCAM_RTSP_PASSWORD` (setup step 1). The camera's stock dashboard
  is open to the same network. On a tailnet so is its MQTT broker, which carries the stream
  password each time the camera starts: that password keeps out only what cannot reach the
  tailnet.
- **The agent's API** (port 8765) lets anyone who can reach it move the gimbal, download clips
  and delete visits. Keep the camera on a network you trust.
- **The selector page** serves only its own computer unless you give it a host
  (`PETCAM_PAGE_HOST`, or `--host` when starting it by hand). If you do, choose an address
  only your devices reach and set `PETCAM_PAGE_PASSPHRASE`, which is then asked for before
  any change. Looking needs no passphrase: the photos, the camera's still
  and the collector's log are open to whoever reaches the page.
- **Traffic is plain HTTP.** Across networks use Tailscale, which encrypts it.
- **`petcam.env` holds your secrets.** Git ignores it; never share it or a launchd plist made
  from it.
- **Reaching the camera or the page by an unusual host name** is refused, as protection
  against other web pages. Addresses, names without dots and names ending `.local`, `.lan`,
  `.home.arpa`, `.internal` or `.ts.net` work; [DEVELOPMENT.md](DEVELOPMENT.md#security-model)
  says how to allow another.

To report a vulnerability see [SECURITY.md](SECURITY.md).

## When things go wrong

Both programs restart themselves and pick up where they left off after power cuts, network
drops and failed deliveries; visits wait on the camera until they can be collected. Start with
the page's **Home** tab, then the logs: `~/petcam/logs/collector.log` on a collector Mac (the journal on Linux),
`/userdata/petcam/agent.log` on the camera.

| Symptom | Likely cause and remedy |
| --- | --- |
| Gimbal answers but does not move | It is powered over USB-C alone. Use the 12 V supply. |
| Log says `camera unreachable` | The camera's address changed: reserve one on your router or use `recamera.local`, correct `petcam.env`, and run the collector install again. On a tailnet, the camera's key expired or it was enrolled again. |
| No photo arrives | The collector's log names the failure. *Send a test postcard* in **Settings** tries your delivery on its own. |
| Visits are collected but "not posted" | The trained model scored them below the posting bar, or as repeats. Adjust both in **Settings**. |
| SSH refuses the camera after a reset | `ssh-keygen -R <camera>`, then set up again from step 2. |
| `/status` shows `"flowing": false` | Capture stopped. The agent restarts it by itself, at first after 90 seconds and then with growing pauses. |

To restart by hand: `sudo /etc/init.d/S96petcam_agent restart` on the camera;
`launchctl kickstart -k gui/$(id -u)/com.petcam.collector` on a Mac.

## Uninstalling

- **Camera:** over SSH, `sudo /etc/init.d/S96petcam_agent stop`, then remove
  `/etc/init.d/S96petcam_agent`, `/etc/init.d/S95petcam_pipeline`,
  `/usr/local/bin/start_pipeline.sh`, `/etc/petcam_rtsp.conf`,
  `/etc/sudoers.d/zz_recamera_nopasswd` and `/userdata/petcam`, and
  restart the camera. For Tailscale, also remove `/etc/init.d/S60tailscale` and
  `/userdata/tailscale`, and the camera from your tailnet.
- **Collector on a Mac:** `launchctl bootout gui/$(id -u)/com.petcam.collector`, then delete
  `~/Library/LaunchAgents/com.petcam.collector.plist`; the same for `com.petcam.page` if you
  installed the page. `~/petcam` holds the installed copy, logs, ratings and every archived
  visit; delete it when you no longer want them.

## More

- [DEVELOPMENT.md](DEVELOPMENT.md): how it works, the agent's API and flags, what is kept
  where, building on it, and running the tests.
- [CONTRIBUTING.md](CONTRIBUTING.md): bug reports and pull requests.
- [MIT License](LICENSE).
