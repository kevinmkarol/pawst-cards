# Security

## Reporting a problem

Please report security problems privately, not in a public issue: on the repository's
**Security** tab choose **Report a vulnerability**
([direct link](https://github.com/kevinmkarol/pawst-cards/security/advisories/new)). Say what
you did, what happened and which commit you ran. This is a spare-time project with no promised
response time, but reports are read and answered.

Only the current `main` is maintained. There are no releases to patch.

## What is and is not a vulnerability here

Pawst Cards is built for a home network or a tailnet you trust, and nothing in it has a login
by default. The [Security section of the README](README.md#security) lists what follows from
that and what to do about it. These are known and documented, not vulnerabilities:

- Anyone who can reach the camera agent's API (port 8765) can move the gimbal, download clips
  and delete visits.
- Anyone who can reach the selector page, once you serve it beyond `127.0.0.1`, can see the
  photos and, without `PETCAM_PAGE_PASSPHRASE`, change the settings.
- Everything between camera, collector and browser travels as plain HTTP.
- The reCamera's SSH and RTSP passwords are the factory's until you change them, and its
  stock dashboard and MQTT broker are Seeed's and stay as you leave them.

Reports are wanted for anything that goes beyond that, for example:

- A web page from elsewhere driving the agent or the selector page through your browser
  (cross-site requests, DNS rebinding).
- A request reading or writing files outside the clips, the archive or the selector's data.
- The camera, or anyone answering in its place, making the collector run a command or write
  outside its archive.
- The passphrase being bypassed, the delivery command being read or changed from the page, or
  the webhook address being read from it.
- A secret from `petcam.env` ending up in a log, a world-readable file or a delivery.
