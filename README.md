# linearrr

Linear MIDI sequencer with locators and p-locks. Sister app to [rrresponseq](https://github.com/) — full-song linear composition in the style of the Yamaha QX3 / MMT-8 / MPC step recording.

- Step input and real-time recording
- Locators navigable by resolution (1/1 to 1/128, dotted and triplets)
- Per-event p-locks (hold a note + turn a knob to edit)
- MIDI and Sampler tracks (internal audio engine with pitch shift, reverse, attack/release, gate)
- Tracker-style operations over selection or bars: reverse, stretch, shift, vel_ramp, humanize, scale_quantize, thin, double_oct, legato, gate_scale, chop, sample_start, attack, release
- JJOS-style modal for every operation
- Footer layout inspired by the QX3 JOB COMMAND TABLE

## Stack

Python + Flask + SSE + inline HTML. No frontend dependencies.

## Requirements

- Python 3.10–3.12 (**do not use 3.13+** — `python-rtmidi` currently crashes on newer versions)
- macOS (tested), Linux (untested)
- Optional: MIDI controller (Launchkey, nanoKONTROL, etc.)

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install flask python-rtmidi sounddevice soundfile numpy
```

## Run

```bash
./run.sh           # default browser
./run_app.sh       # Chrome in --app mode (no chrome UI)
```

Then open `http://localhost:5002`.

## MIDI configuration

Edit `config.py` with your MIDI port names (`MIDI_OUT_PORT`, `MIDI_KB_PORT`, `NK_IN_PORT`).

## License

[PolyForm Noncommercial 1.0.0](LICENSE.md) — use, modification and distribution are allowed **for noncommercial purposes only**. For commercial use, please contact the author.
