# linearrr

Secuenciador MIDI lineal con locators y p-locks. App hermana de [rrresponseq](https://github.com/) — composición lineal de temas enteros estilo Yamaha QX3 / MMT-8 / MPC step recording.

- Step input y real-time recording
- Locators navegables por resolución (1/1 a 1/128, dotted y triplets)
- P-locks por evento (mantén nota + gira knob para editar)
- Tracks MIDI + Sampler (audio engine interno con pitch shift, reverse, attack/release, gate)
- Operaciones tracker-style sobre selección o bars: reverse, stretch, shift, vel_ramp, humanize, scale_quantize, thin, double_oct, legato, gate_scale, chop, sample_start, attack, release
- Modal JJOS-style para todas las operaciones
- Layout footer estilo QX3 JOB COMMAND TABLE

## Stack

Python + Flask + SSE + HTML inline. Sin dependencias frontend.

## Requisitos

- Python 3.10+
- macOS (probado), Linux (sin probar)
- Opcional: controlador MIDI (Launchkey, nanoKONTROL, etc.)

## Instalación

```bash
python3 -m venv .venv
.venv/bin/pip install flask python-rtmidi sounddevice soundfile numpy
```

## Lanzar

```bash
./run.sh           # navegador por defecto
./run_app.sh       # Chrome en modo --app (sin barras)
```

Abre `http://localhost:5002`.

## Configuración MIDI

Edita `config.py` con tus puertos MIDI (`MIDI_OUT_PORT`, `MIDI_KB_PORT`, `NK_IN_PORT`).

## Licencia

[PolyForm Noncommercial 1.0.0](LICENSE.md) — uso, modificación y distribución permitidos **solo para fines no comerciales**. Para uso comercial, contactar al autor.
