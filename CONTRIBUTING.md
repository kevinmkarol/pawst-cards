# Contributing

Pawst Cards is one person's project, built around one camera in one room, and published in case
it is useful to someone else. Issues and pull requests are welcome; answers may take a while.

## Asking for help or reporting a bug

Open an issue and include:

- which part it concerns: the camera agent, the collector, or the selector page;
- the camera's reCamera OS version and what the collector runs on (macOS or Linux, Python
  version);
- the lines from the log around the failure: `/userdata/petcam/agent.log` on the camera,
  `~/petcam/logs/collector.log` on a collector Mac (the journal on Linux);
- what `curl http://<camera>:8765/status` answers, if the camera is involved.

Before posting, take your own addresses, webhook URLs, tokens and channel ids out of whatever
you paste. Never attach `petcam.env` or a rendered launchd plist. Photos and clips show the
inside of your home; attach them only if you are happy for them to be public.

Security problems go through [SECURITY.md](SECURITY.md), not a public issue.

## Changing the code

[DEVELOPMENT.md](DEVELOPMENT.md) describes how the parts work and, under
[Tests and tooling](DEVELOPMENT.md#tests-and-tooling), the checks to run.

- **The agent runs on the camera's Python 3.9 with no packages.** Keep `camera/agent` to the
  standard library and to syntax 3.9 accepts.
- **Tests run against real code.** The collector's tests talk to an in-process copy of the
  agent's API and cut frames from real clips with ffmpeg; add to them in the same way rather
  than with mocks.
- **Formatting** is black and ruff at the versions pinned in `.pre-commit-config.yaml`, with
  lines up to 100 characters.
- **Every source file opens with one or two `ABOUTME:` comment lines** saying what the file is for.
  Comments describe what the code does and why, not how it came to be that way.
- **Keep the documentation true.** The README covers setup and use, `petcam.env.example` every
  setting, and DEVELOPMENT.md the API, flags and what is kept on disk; a change to any of those
  belongs there in the same pull request.
- **Nothing of your own setup in a commit:** no addresses, host names, tokens, photos or clips
  from your home. GitHub's push protection will refuse known kinds of token, but not the rest.

Small pull requests that do one thing are the easiest to accept. If you plan something larger,
such as support for another camera, open an issue first to talk it through.

By contributing you agree that your contribution is released under the [MIT License](LICENSE).
