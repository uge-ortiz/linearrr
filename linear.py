#!/usr/bin/env python3
"""rrrlinear — secuenciador lineal con locators y p-locks.
Fase 3: step input MIDI + copy/erase/duplicate evento.
"""
import os
import json
import time
import copy
import random
import threading
from flask import Flask, Response, request

import config

# nanoKONTROL2 transport CCs (estándar Korg)
NK_PLAY = 41
NK_STOP = 42
NK_REW  = 43
NK_FF   = 44
NK_REC  = 45

try:
    import rtmidi
    _HAS_RTMIDI = True
except ImportError:
    _HAS_RTMIDI = False
    print("[rrrlinear] aviso: rtmidi no disponible — sin entrada MIDI")

try:
    import sounddevice as _sd
    import soundfile as _sf
    import numpy as _np
    _HAS_AUDIO = True
except ImportError:
    _HAS_AUDIO = False
    print("[rrrlinear] aviso: sin sounddevice/soundfile/numpy — sampler off")

# ── Paths ────────────────────────────────────────────────────────────────────
_APP_SUPPORT = os.path.expanduser('~/Library/Application Support/rrrlinear')
os.makedirs(_APP_SUPPORT, exist_ok=True)
SONGS_FILE = os.path.join(_APP_SUPPORT, 'songs.json')
SETTINGS_FILE = os.path.join(_APP_SUPPORT, 'settings.json')
_LEGACY_SONGS = os.path.expanduser('~/Library/Application Support/rrresponseq/songs.json')

# Override config.* con values del settings.json propio (no compartido)
try:
    with open(SETTINGS_FILE) as _f:
        _s = json.load(_f)
    for _k in ('MIDI_OUT_PORT','MIDI_OUT_PORT2','MIDI_KB_PORT',
               'NK_IN_PORT','NK_OUT_PORT','LAUNCHPAD_PORT'):
        if _k in _s and _s[_k]:
            setattr(config, _k, _s[_k])
            print(f"[CONFIG] {_k} = {_s[_k]} (de settings.json)")
except FileNotFoundError:
    pass
except Exception as _e:
    print(f"[CONFIG] error leyendo settings.json: {_e}")

# Migración: si existe un songs.json antiguo en el dir compartido y no hay uno propio aún
if not os.path.exists(SONGS_FILE) and os.path.exists(_LEGACY_SONGS):
    try:
        import shutil
        shutil.copy(_LEGACY_SONGS, SONGS_FILE)
        print(f"[MIGRATE] copiado {_LEGACY_SONGS} → {SONGS_FILE}")
    except Exception as _e:
        print(f"[MIGRATE] error: {_e}")

# ── Constantes musicales ────────────────────────────────────────────────────
TPQ = 192  # ticks por quarter — 192 permite 1/128 sin truncar
# int = recto, "NT" = triplete (2/3), "N." = puntillo (3/2)
RESOLUTIONS = [1, 2, '2.', 4, '4.', '4T', 8, '8.', '8T',
               16, '16.', '16T', 32, '32.', '32T', 64, '64.', '64T', 128]
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(n):
    return f"{NOTE_NAMES[n % 12]}{n // 12 - 2}"


def ticks_per_step(res):
    """Acepta int, 'NT' (tresillo, 2/3) o 'N.' (puntillo, 3/2)."""
    if isinstance(res, str):
        if res.endswith('T'):
            return (TPQ * 4 // int(res[:-1])) * 2 // 3
        if res.endswith('.'):
            return (TPQ * 4 // int(res[:-1])) * 3 // 2
    return TPQ * 4 // int(res)


def res_label(res):
    return f"1/{res}" if not (isinstance(res, str) and res.endswith('T')) else f"1/{res}"


def res_short(res):
    return str(res)


def steps_per_bar(res, ts_num=4):
    """Cuántos steps de res entran en 1 bar (depende del time signature)."""
    return (TPQ * ts_num) // ticks_per_step(res)


# Ribbon fields por tipo de evento (8 slots; '' = vacío)
RIBBON_FIELDS = {
    'note': [('NOTE','note'),('VEL','vel'),('LEN','len_steps'),
             ('PROB','prob'),('MICRO','micro'),('GATE','gate'),
             ('RTCH','ratchet'),('RDIV','rdiv'),
             ('PROG','program'),('BANK','bank')],
    'cc':   [('CC','cc'),('VAL','val'),('RAMP','ramp'),
             ('','_'),('','_'),('','_'),('','_'),('','_')],
    'prog': [('PROG','program'),('','_'),('','_'),('','_'),
             ('','_'),('','_'),('','_'),('','_')],
    'bank': [('BANK','msb'),('','_'),('','_'),('','_'),
             ('','_'),('','_'),('','_'),('','_')],
}


# ── Audio engine (simple sampler) ───────────────────────────────────────────
SAMPLE_RATE = 44100
BLOCK_SIZE  = 256


class SamplerEngine:
    """Mixer simple: hasta N voces de samples reproduciéndose simultáneamente."""
    def __init__(self):
        self.voices = []        # [(data, pos, rate, vel)]
        self.lock = threading.Lock()
        self.cache = {}         # {path: np.float32 mono}
        self.stream = None

    def load_sample(self, path):
        if path in self.cache:
            return self.cache[path]
        try:
            data, sr = _sf.read(path, dtype='float32', always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)  # downmix a mono
            if sr != SAMPLE_RATE:
                # Resample muy básico (lineal) al rate de salida
                ratio = SAMPLE_RATE / sr
                n_out = int(len(data) * ratio)
                idx = _np.arange(n_out) / ratio
                idx_floor = idx.astype(_np.int64).clip(0, len(data)-1)
                data = data[idx_floor].astype(_np.float32)
            self.cache[path] = data
            return data
        except Exception as e:
            print(f"[SAMPLE LOAD ERROR] {path}: {e}")
            return None

    def play(self, path, note=60, vel=100, dur_sec=None,
              reverse=False, start=0.0, attack_ms=0, release_ms=0):
        data = self.load_sample(path)
        if data is None:
            return
        # Reverse y start offset → mutamos una vista del data
        if reverse:
            data = data[::-1]
        if start > 0 and start < 1.0:
            skip = int(start * len(data))
            data = data[skip:]
        rate = 2.0 ** ((note - 60) / 12.0)
        gain = max(0.0, min(1.0, vel / 127.0))
        cut = None if dur_sec is None else int(dur_sec * SAMPLE_RATE * rate)
        fade_out = int(0.005 * SAMPLE_RATE * rate)
        if release_ms > 0:
            fade_out = max(fade_out, int(release_ms/1000 * SAMPLE_RATE * rate))
        fade_in = int(attack_ms/1000 * SAMPLE_RATE * rate) if attack_ms > 0 else 0
        with self.lock:
            self.voices.append([data, 0.0, rate, gain, cut, fade_out, fade_in])
            if len(self.voices) > 32:
                self.voices.pop(0)

    def _callback(self, outdata, frames, time_info, status):
        out = _np.zeros(frames, dtype=_np.float32)
        with self.lock:
            alive = []
            for v in self.voices:
                data, pos, rate, gain, cut, fade_out, fade_in = v
                idx = pos + _np.arange(frames) * rate
                last = len(data) - 1
                effective_last = min(last, cut) if cut is not None else last
                in_bounds = idx < effective_last
                if not in_bounds.any():
                    continue
                ii = idx.astype(_np.int64).clip(0, last)
                samp = data[ii] * gain
                samp[~in_bounds] = 0
                # Fade-in al inicio
                if fade_in > 0:
                    in_attack = (idx < fade_in) & in_bounds
                    if in_attack.any():
                        ramp = _np.clip(idx[in_attack] / fade_in, 0, 1).astype(_np.float32)
                        samp[in_attack] = samp[in_attack] * ramp
                # Fade-out cerca del corte
                if cut is not None and fade_out > 0:
                    dist = effective_last - idx
                    in_release = (dist < fade_out) & in_bounds
                    if in_release.any():
                        ramp = _np.clip(dist[in_release] / fade_out, 0, 1).astype(_np.float32)
                        samp[in_release] = samp[in_release] * ramp
                out += samp
                v[1] = pos + frames * rate
                if v[1] < effective_last:
                    alive.append(v)
            self.voices = alive
        # Stereo: duplicar mono
        outdata[:, 0] = out
        if outdata.shape[1] > 1:
            outdata[:, 1] = out

    def start(self):
        if not _HAS_AUDIO or self.stream is not None:
            return
        try:
            self.stream = _sd.OutputStream(
                samplerate=SAMPLE_RATE, channels=2, dtype='float32',
                blocksize=BLOCK_SIZE, callback=self._callback)
            self.stream.start()
            print(f"[AUDIO] stream iniciado @{SAMPLE_RATE}Hz")
        except Exception as e:
            print(f"[AUDIO ERROR] {e}")

    def stop(self):
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None


def _new_note(tick, res, note=60, vel=100):
    return {'type':'note', 'tick':tick, 'res':res,
            'note':note, 'vel':vel, 'len_steps':1,
            'prob':1.0, 'micro':0.0, 'gate':0.95,
            'ratchet':1, 'rdiv':1.0,
            # Cambios inline (None = no cambia):
            'program':None, 'bank':None}


def _new_cc(tick, res, cc=74, val=64, ramp=False):
    return {'type':'cc', 'tick':tick, 'res':res,
            'cc':cc, 'val':val, 'ramp':ramp}


def _new_prog(tick, program=0):
    return {'type':'prog', 'tick':tick, 'program':program}


def _new_bank(tick, msb=0):
    return {'type':'bank', 'tick':tick, 'msb':msb}


# ── Sequencer ───────────────────────────────────────────────────────────────
class Sequencer:
    def __init__(self):
        self.song = {
            'bpm': float(config.BPM),
            'time_sig': [4, 4],
            'resolution': 16,
            'cur_track': 0,
            'cur_event': 0,
            'cur_field': 0,
            'locator_tick': 0,    # cursor de inserción (tick abs)
            'sel': None,          # {'tick_start', 'tick_end'} o None
            'view': 'micro',      # 'micro' (1 pista detallada) | 'macro' (8 pistas zoom)
            'skin': 'modern',     # 'modern' | 'qx3'
            'tracks': [
                {'name': f'T{i+1}', 'channel': i, 'port': 0,
                 'mute': False, 'solo': False, 'events': [],
                 'type': 'midi', 'sample_dir': '', 'samples': [],
                 'current_program': 0}
                for i in range(8)
            ],
        }
        self.running = False
        self.recording = False
        self.loop_on = False
        self.dirty = False
        self._last_saved_ts = 0.0
        self.clipboard = None
        self._lock = threading.Lock()
        self._kb_in = None
        self._nk_in = None
        self._outs = []         # MidiOut por índice de port
        self._play_start_tick = 0
        self._play_start_time = 0.0
        self._last_emit_tick = [-1] * 8   # por pista
        self._held_notes = {}  # real-time: {(ch, note): {'ev':dict, 'tick_on':int}}
        # Estado per-pista vigente: port/program/bank (cambia con eventos inline)
        self._track_routing = [{'port':None, 'program':None, 'bank':None}
                                for _ in range(8)]
        # Sampler engine (mono shared)
        self._sampler = SamplerEngine()
        if _HAS_AUDIO:
            self._sampler.start()
        self._chord_held = set()       # step input: notas sostenidas
        self._chord_tick = 0           # tick donde se insertan
        self._chord_events = []        # refs a los eventos del acorde actual
        self._chord_t_start = 0.0      # perf_counter del primer note_on
        self._last_cc_ts = {}          # thinning de grabación CC: {cc_num: t}
        self._flash_label = ''         # mini-tooltip al cambiar samples/PC/Bank
        self._flash_ts = 0
        self._modal_capture_armed = False
        self._modal_captured_note = None
        self._modal_captured_ts = 0.0
        self._undo_stack = []          # snapshots para undo
        self._redo_stack = []
        self.UNDO_MAX = 50

    # ── MIDI ────────────────────────────────────────────────────────────────
    # ── Persistencia ────────────────────────────────────────────────────────
    def save(self):
        with self._lock:
            try:
                with open(SONGS_FILE, 'w') as f:
                    json.dump(self.song, f, indent=2)
                self.dirty = False
                self._last_saved_ts = time.time()
                print(f"[SAVE] {SONGS_FILE}")
            except Exception as e:
                print(f"[SAVE ERROR] {e}")

    def load(self):
        if not os.path.exists(SONGS_FILE):
            return
        try:
            with open(SONGS_FILE) as f:
                self.load_from_dict(json.load(f))
            print(f"[LOAD] {SONGS_FILE}")
        except Exception as e:
            print(f"[LOAD ERROR] {e}")

    def load_from_dict(self, data):
        with self._lock:
            for k, v in data.items():
                self.song[k] = v
            self.dirty = False

    def push_undo(self):
        """Guarda snapshot del estado actual para undo."""
        with self._lock:
            snap = copy.deepcopy(self.song)
            self._undo_stack.append(snap)
            if len(self._undo_stack) > self.UNDO_MAX:
                self._undo_stack.pop(0)
            self._redo_stack = []

    def undo(self):
        with self._lock:
            if not self._undo_stack:
                return
            self._redo_stack.append(copy.deepcopy(self.song))
            self.song = self._undo_stack.pop()
            self.dirty = True

    def redo(self):
        with self._lock:
            if not self._redo_stack:
                return
            self._undo_stack.append(copy.deepcopy(self.song))
            self.song = self._redo_stack.pop()
            self.dirty = True

    def new_song(self):
        with self._lock:
            for t in self.song['tracks']:
                t['events'] = []
                t['mute'] = False
                t['solo'] = False
            self.song['locator_tick'] = 0
            self.song['cur_event'] = -1
            self.song['cur_field'] = 0
            self.song['cur_track'] = 0
            self.song['sel'] = None
            self.song['resolution'] = 16
            self.song['bpm'] = float(config.BPM)
            self.clipboard = None
            self.dirty = False

    def toggle_loop(self):
        with self._lock:
            self.loop_on = not self.loop_on

    def set_loop_playback(self, from_bar, to_bar):
        """Activa loop entre from_bar..to_bar (TO exclusivo). Setea sel + loop_on."""
        with self._lock:
            bt = self._bar_ticks()
            lo = max(0, int(from_bar)-1) * bt
            hi = max(int(from_bar), int(to_bar)-1) * bt
            if hi <= lo:
                return
            self.song['sel'] = {'tick_start': lo, 'tick_end': hi, 'anchor': lo}
            self.loop_on = True
            self.song['locator_tick'] = lo
            self._sync_event_from_locator()

    def open_midi(self):
        if not _HAS_RTMIDI:
            return
        # Outs
        for name, tag in [(config.MIDI_OUT_PORT, 'OUT1'),
                          (config.MIDI_OUT_PORT2, 'OUT2')]:
            self._outs.append(self._open_output(name, tag))
        # Teclado
        self._kb_in = self._open_input(config.MIDI_KB_PORT, self._on_keyboard, 'KB')
        # nanoKONTROL
        self._nk_in = self._open_input(config.NK_IN_PORT, self._on_nano, 'NANO')

    def _open_output(self, name, tag):
        out = rtmidi.MidiOut()
        ports = out.get_ports()
        for i, p in enumerate(ports):
            if name in p:
                try:
                    out.open_port(i)
                    print(f"[{tag}] abierto: {p}")
                    return out
                except Exception as e:
                    print(f"[{tag}] error: {e}")
        print(f"[{tag}] no encontrado: '{name}'")
        return None

    # ── Settings: cambio dinámico de puertos ────────────────────────────────
    PORT_SLOTS = [
        ('midi_out',  'MIDI_OUT_PORT',  'OUT 1'),
        ('midi_out2', 'MIDI_OUT_PORT2', 'OUT 2'),
        ('kb',        'MIDI_KB_PORT',   'KEYBOARD'),
        ('nano',      'NK_IN_PORT',     'CONTROLLER'),
    ]

    def list_midi_ports(self):
        if not _HAS_RTMIDI:
            return {'in': [], 'out': []}
        return {
            'in': rtmidi.MidiIn().get_ports(),
            'out': rtmidi.MidiOut().get_ports(),
        }

    def get_settings_view(self):
        rows = [
            {'slot': slot, 'label': label,
             'value': getattr(config, attr, '') or '—'}
            for slot, attr, label in self.PORT_SLOTS
        ]
        rows.append({'slot': 'skin', 'label': 'SKIN',
                     'value': self.song.get('skin', 'modern').upper()})
        return rows

    def cycle_port(self, slot):
        """Cicla al siguiente puerto disponible para ese slot."""
        ports_info = self.list_midi_ports()
        is_out = slot in ('midi_out', 'midi_out2')
        ports = ports_info['out'] if is_out else ports_info['in']
        if not ports:
            return
        attr = next(a for s, a, _ in self.PORT_SLOTS if s == slot)
        cur = getattr(config, attr, '')
        # Buscar índice actual
        idx = -1
        for i, p in enumerate(ports):
            if cur and cur in p:
                idx = i; break
        new_port = ports[(idx + 1) % len(ports)]
        setattr(config, attr, new_port)
        self._reopen_port(slot)
        self._persist_settings()

    def _reopen_port(self, slot):
        if slot == 'midi_out':
            if self._outs and self._outs[0]:
                try: self._outs[0].close_port()
                except: pass
            new = self._open_output(config.MIDI_OUT_PORT, 'OUT1')
            if not self._outs: self._outs = [new, None]
            else: self._outs[0] = new
        elif slot == 'midi_out2':
            if len(self._outs) > 1 and self._outs[1]:
                try: self._outs[1].close_port()
                except: pass
            new = self._open_output(config.MIDI_OUT_PORT2, 'OUT2')
            if len(self._outs) < 2: self._outs.append(new)
            else: self._outs[1] = new
        elif slot == 'kb':
            if self._kb_in:
                try: self._kb_in.cancel_callback(); self._kb_in.close_port()
                except: pass
            self._kb_in = self._open_input(config.MIDI_KB_PORT, self._on_keyboard, 'KB')
        elif slot == 'nano':
            if self._nk_in:
                try: self._nk_in.cancel_callback(); self._nk_in.close_port()
                except: pass
            self._nk_in = self._open_input(config.NK_IN_PORT, self._on_nano, 'NANO')

    def _persist_settings(self):
        try:
            data = {}
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE) as f:
                    data = json.load(f)
            for _, attr, _ in self.PORT_SLOTS:
                data[attr] = getattr(config, attr, '')
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[SETTINGS] error: {e}")

    def _open_input(self, name, cb, tag):
        inp = rtmidi.MidiIn()
        inp.ignore_types(sysex=True, timing=True, active_sense=True)
        ports = inp.get_ports()
        for i, p in enumerate(ports):
            if name in p:
                try:
                    inp.open_port(i)
                    inp.set_callback(cb)
                    print(f"[{tag}] abierto: {p}")
                    return inp
                except Exception as e:
                    print(f"[{tag}] error: {e}")
        print(f"[{tag}] no encontrado: '{name}'")
        print(f"  Disponibles: {ports}")
        return None

    def _on_keyboard(self, event, _):
        msg = event[0]
        if not msg or len(msg) < 3:
            return
        status = msg[0] & 0xF0
        # MIDI thru SIEMPRE → oír lo que tocas
        self._thru(msg)
        is_on  = (status == 0x90 and msg[2] > 0)
        is_off = (status == 0x80) or (status == 0x90 and msg[2] == 0)
        is_cc  = (status == 0xB0)
        # ── Captura para campos de tipo NOTE en modales ──
        if self._modal_capture_armed and is_on:
            self._modal_captured_note = msg[1]
            self._modal_captured_ts = time.time()
            self._modal_capture_armed = False
            _poke()
            return
        if not self.recording:
            return
        # ── CC events (modwheel, pedal, etc.) con thinning 50ms por CC# ──
        if is_cc:
            cc_num, val = msg[1], msg[2]
            now = time.perf_counter()
            if now - self._last_cc_ts.get(cc_num, 0) < 0.05:
                return
            self._last_cc_ts[cc_num] = now
            if self.running:
                tps = self._ticks_per_sec()
                elapsed = now - self._play_start_time
                cur_tick = self._play_start_tick + int(elapsed * tps)
                with self._lock:
                    ev = _new_cc(cur_tick, self.song['resolution'], cc_num, val)
                    self._insert_event(ev, advance=False)
            else:
                with self._lock:
                    ev = _new_cc(self.song['locator_tick'],
                                  self.song['resolution'], cc_num, val)
                    self._insert_event(ev, advance=False)
            _poke()
            return
        note, vel = msg[1], msg[2]
        ch = msg[0] & 0x0F
        if is_on:
            if self.running:
                if not self._held_notes:
                    self.push_undo()
                tps = self._ticks_per_sec()
                now_t = time.perf_counter()
                elapsed = now_t - self._play_start_time
                cur_tick = self._play_start_tick + int(elapsed * tps)
                # Detectar acorde: si han pasado <50ms desde la primera mantenida,
                # compartir tick para que aparezca como acorde en el mismo step
                if not self._held_notes:
                    self._chord_anchor_tick = cur_tick
                    self._chord_anchor_t = now_t
                elif now_t - getattr(self, '_chord_anchor_t', 0) < 0.05:
                    cur_tick = self._chord_anchor_tick
                self._held_notes[(ch, note)] = {
                    'note': note, 'vel': vel, 'tick_on': cur_tick}
            else:
                # Step input: acorde si hay notas sostenidas
                if not self._chord_held:
                    self.push_undo()
                with self._lock:
                    if not self._chord_held:
                        self._chord_tick = self.song['locator_tick']
                        self._chord_events = []
                        self._chord_t_start = time.perf_counter()
                    ev = _new_note(self._chord_tick, self.song['resolution'], note, vel)
                    self._apply_track_routing_to_note(ev, self.song['cur_track'])
                    self._insert_event(ev, advance=False)
                    self._chord_events.append(ev)
                self._chord_held.add(note)
            _poke()
        elif is_off:
            if self.running:
                held = self._held_notes.pop((ch, note), None)
                if held:
                    tps = self._ticks_per_sec()
                    elapsed = time.perf_counter() - self._play_start_time
                    cur_tick = self._play_start_tick + int(elapsed * tps)
                    dur_ticks = max(1, cur_tick - held['tick_on'])
                    with self._lock:
                        ev = _new_note(held['tick_on'], self.song['resolution'],
                                        held['note'], held['vel'])
                        ev['len_ticks'] = dur_ticks
                        self._apply_track_routing_to_note(ev, self.song['cur_track'])
                        self._insert_event(ev, advance=False)
                    _poke()
            else:
                # Step input: soltar tecla
                self._chord_held.discard(note)
                if not self._chord_held:
                    # Acorde terminado: calcular duración real y aplicar a todas
                    dur_sec = max(0.05, time.perf_counter() - self._chord_t_start)
                    tps = self._ticks_per_sec()
                    dur_ticks = int(dur_sec * tps)
                    with self._lock:
                        for ev in self._chord_events:
                            ev['len_ticks'] = dur_ticks
                        self.song['locator_tick'] += ticks_per_step(self.song['resolution'])
                        self._sync_event_from_locator()
                    self._chord_events = []
                    _poke()

    def _thru(self, msg):
        """Reenvía mensaje. Si pista es sampler: dispara sample local en vez de MIDI."""
        if not msg: return
        ct = self.song['cur_track']
        track = self.song['tracks'][ct]
        # Sampler: dispara sample en note_on, ignora resto (no enviar MIDI)
        if track.get('type') == 'sampler':
            if (msg[0] & 0xF0) == 0x90 and len(msg) >= 3 and msg[2] > 0:
                samples = track.get('samples', [])
                if samples:
                    cur = max(0, min(len(samples)-1, track.get('current_program', 0)))
                    sdir = track.get('sample_dir', '')
                    path = os.path.join(sdir, samples[cur])
                    self._sampler.play(path, note=msg[1], vel=msg[2])
            return
        # MIDI thru estándar
        port = track['port']
        out = self._outs[port] if port < len(self._outs) else None
        if not out: return
        status = msg[0] & 0xF0
        new_status = status | (track['channel'] & 0x0F)
        try:
            out.send_message([new_status] + list(msg[1:]))
        except Exception:
            pass

    def _on_nano(self, event, _):
        msg = event[0]
        if not msg or len(msg) < 3:
            return
        if msg[0] & 0xF0 != 0xB0:
            return
        cc, val = msg[1], msg[2]
        # Transport (botones del nano, val>0 = press)
        if val > 0:
            if cc == NK_PLAY: self.toggle_play(); _poke(); return
            if cc == NK_STOP: self.stop(); _poke(); return
            if cc == NK_REW:  self.rewind(); _poke(); return
            if cc == NK_FF:   self.fast_forward(); _poke(); return
        # BPM knob (CC 19)
        if cc == getattr(config, 'NK_BPM_CC', 19):
            with self._lock:
                self.song['bpm'] = round(30 + (val / 127.0) * 210, 1)
            _poke()
            return
        # Knobs (CC 11-18) → fields 0-7
        kbase = config.NK_KNOB_BASE
        if kbase <= cc < kbase + 8:
            knob_idx = cc - kbase
            with self._lock:
                self.song['cur_field'] = knob_idx
            self.edit_field(absolute=val)
            _poke()
            return
        # Faders 1-2 → current_program / current_bank de pista (con preview)
        fbase = config.NK_FADER_BASE
        if cc == fbase:
            self.set_track_current_program(val)
            _poke()
            return
        if cc == fbase + 1:
            self.set_track_current_bank(val)
            _poke()
            return
        # Faders 3-8 → fields 10-15 (libres por ahora)
        if fbase + 2 <= cc < fbase + 8:
            field_idx = 8 + (cc - fbase)
            with self._lock:
                self.song['cur_field'] = field_idx
            self.edit_field(absolute=val)
            _poke()

    # ── Navegación / cursor ─────────────────────────────────────────────────
    def select_track(self, t):
        with self._lock:
            if 0 <= t < 8:
                self.song['cur_track'] = t
                self._sync_event_from_locator()

    def set_track_port(self, port):
        with self._lock:
            self.song['tracks'][self.song['cur_track']]['port'] = max(0, min(1, int(port)))
            self.dirty = True

    def toggle_mute(self):
        with self._lock:
            t = self.song['tracks'][self.song['cur_track']]
            t['mute'] = not t.get('mute', False)
            self.dirty = True

    def toggle_solo(self):
        with self._lock:
            t = self.song['tracks'][self.song['cur_track']]
            t['solo'] = not t.get('solo', False)
            self.dirty = True

    def next_sample(self, delta=1):
        with self._lock:
            t = self.song['tracks'][self.song['cur_track']]
            if t.get('type') != 'sampler':
                return
            samples = t.get('samples', [])
            if not samples:
                return
            cur = t.get('current_program', 0)
            new_idx = max(0, min(len(samples)-1, cur + delta))
            if new_idx == cur:
                return
            t['current_program'] = new_idx
            self._flash_label = f"SMP {new_idx+1}/{len(samples)}  {samples[new_idx]}"
            self._flash_ts = time.time()
        sdir = t.get('sample_dir', '')
        path = os.path.join(sdir, samples[new_idx])
        self._sampler.play(path, note=60, vel=100)

    def set_track_current_program(self, val):
        """Setea current_program de la pista (0-127). Sampler → preview;
        MIDI → envía PC inmediato para audicionar."""
        with self._lock:
            t = self.song['tracks'][self.song['cur_track']]
            ttype = t.get('type', 'midi')
            if ttype == 'sampler':
                samples = t.get('samples', [])
                if not samples:
                    return
                idx = max(0, min(len(samples)-1, int(val * (len(samples)-1) / 127)))
                if idx == t.get('current_program', 0):
                    return
                t['current_program'] = idx
                self._flash_label = f"SMP {idx+1}/{len(samples)}  {samples[idx]}"
                self._flash_ts = time.time()
                sdir = t.get('sample_dir', '')
                path = os.path.join(sdir, samples[idx])
                self._sampler.play(path, note=60, vel=100)
            else:
                v = max(0, min(127, int(val)))
                t['current_program'] = v
                self._flash_label = f"PROG {v}"
                self._flash_ts = time.time()
                # Preview: enviar PC al synth
                ch = t['channel']
                port = t['port']
                out = self._outs[port] if port < len(self._outs) else None
                if out:
                    try: out.send_message([0xC0|ch, v])
                    except: pass

    def set_track_current_bank(self, val):
        """Setea current_bank de la pista (0-127). Envía Bank MSB inmediato."""
        with self._lock:
            t = self.song['tracks'][self.song['cur_track']]
            v = max(0, min(127, int(val)))
            t['current_bank'] = v
            self._flash_label = f"BANK {v}"
            self._flash_ts = time.time()
            if t.get('type', 'midi') == 'midi':
                ch = t['channel']
                port = t['port']
                out = self._outs[port] if port < len(self._outs) else None
                if out:
                    try: out.send_message([0xB0|ch, 0, v])
                    except: pass

    def set_track_type(self, track_type, sample_dir=None):
        """Cambia tipo de pista. Si sampler + sample_dir → escanea samples."""
        with self._lock:
            t = self.song['tracks'][self.song['cur_track']]
            t['type'] = 'sampler' if track_type == 'sampler' else 'midi'
            if sample_dir is not None:
                t['sample_dir'] = sample_dir
            # Escanear samples si es sampler
            if t['type'] == 'sampler' and t['sample_dir']:
                t['samples'] = self._scan_samples(t['sample_dir'])
            else:
                t['samples'] = []
            self.dirty = True

    def _scan_samples(self, dir_path):
        """Lista archivos .wav/.aif/.aiff/.flac. Filtra ocultos y AppleDouble."""
        try:
            exts = ('.wav', '.aif', '.aiff', '.flac')
            files = [f for f in os.listdir(dir_path)
                     if f.lower().endswith(exts)
                     and not f.startswith('.')]
            return sorted(files)
        except Exception as e:
            print(f"[SCAN ERROR] {e}")
            return []

    def _sync_locator_from_event(self):
        ct = self.song['cur_track']
        evs = self.song['tracks'][ct]['events']
        if evs and 0 <= self.song['cur_event'] < len(evs):
            self.song['locator_tick'] = evs[self.song['cur_event']]['tick']

    def _sync_event_from_locator(self):
        """Selecciona el primer evento que cae en el step contenedor del locator
        (tolera eventos con tick intermedio grabados en real-time)."""
        ct = self.song['cur_track']
        evs = self.song['tracks'][ct]['events']
        res = self.song['resolution']
        tps = ticks_per_step(res)
        loc_step = self.song['locator_tick'] // tps
        for i, ev in enumerate(evs):
            if ev['tick'] // tps == loc_step:
                if self.song['cur_event'] != i:
                    self.song['cur_event'] = i
                    self._clamp_cur_field()
                    self._preview_event(ev)
                return
        self.song['cur_event'] = -1

    def _clamp_cur_field(self):
        """Garantiza que cur_field apunta a un slot válido del ribbon actual."""
        ev = self._selected_event()
        if not ev:
            return
        fields = RIBBON_FIELDS.get(ev['type'], RIBBON_FIELDS['note'])
        valid = [i for i, (lbl, _) in enumerate(fields) if lbl]
        if valid and self.song['cur_field'] not in valid:
            self.song['cur_field'] = valid[0]

    def _sampler_play_note(self, track, ev):
        """Dispara sample(s) de la pista para el NOTE event. Soporta ratchet."""
        samples = track.get('samples', [])
        if not samples:
            return
        prog = ev.get('program') or 0
        prog = max(0, min(len(samples)-1, int(prog)))
        sdir = track.get('sample_dir', '')
        path = os.path.join(sdir, samples[prog])
        note = ev.get('note', 60)
        vel = ev.get('vel', 100)
        ratchet = max(1, int(ev.get('ratchet', 1)))
        rdiv = max(0.05, float(ev.get('rdiv', 1.0)))
        res = ev.get('res', 16)
        gate = max(0.05, min(0.99, ev.get('gate', 0.95)))
        if 'len_ticks' in ev and ev['len_ticks']:
            total_sec = ev['len_ticks'] / self._ticks_per_sec()
        else:
            len_ticks = ev.get('len_steps', 1) * ticks_per_step(res)
            total_sec = len_ticks / self._ticks_per_sec()
        hit_interval = (total_sec / ratchet) * rdiv
        # Cada hit suena gate% de su slot. Sin gate (=1.0), sample completo.
        hit_dur = (hit_interval * gate) if gate < 0.98 else None
        # Efectos sample por evento
        sk = dict(note=note, vel=vel, dur_sec=hit_dur,
                   reverse=bool(ev.get('sample_reverse', False)),
                   start=float(ev.get('sample_start', 0.0)),
                   attack_ms=int(ev.get('attack_ms', 0)),
                   release_ms=int(ev.get('release_ms', 0)))
        for r in range(ratchet):
            delay = r * hit_interval
            if delay == 0:
                self._sampler.play(path, **sk)
            else:
                threading.Timer(delay, self._sampler.play,
                                args=(path,), kwargs=dict(sk)).start()

    def _preview_event(self, ev):
        """Emite evento(s) para previsualizar (sin avance del playhead).
        Para NOTE: dispara todo el acorde del mismo step + aplica routing inline."""
        if not ev:
            return
        ct = self.song['cur_track']
        track = self.song['tracks'][ct]
        # Si la pista es sampler, dispara sample
        if track.get('type') == 'sampler' and ev.get('type') == 'note':
            tps = ticks_per_step(self.song['resolution'])
            step_t = ev['tick'] // tps
            chord = [e for e in track['events']
                     if e.get('type') == 'note' and e['tick'] // tps == step_t]
            for cev in chord:
                self._sampler_play_note(track, cev)
            return
        ch = track['channel']
        port = track['port']
        out = self._outs[port] if port < len(self._outs) else None
        if not out:
            return
        t = ev.get('type', 'note')
        try:
            if t == 'note':
                # Routing inline antes de la nota
                bk = ev.get('bank')
                if bk is not None:
                    out.send_message([0xB0 | ch, 0, bk])
                pg = ev.get('program')
                if pg is not None:
                    out.send_message([0xC0 | ch, pg])
                # Disparar TODAS las notas del mismo step (acorde)
                tps = ticks_per_step(self.song['resolution'])
                step_t = ev['tick'] // tps
                chord = [e for e in self.song['tracks'][ct]['events']
                         if e.get('type') == 'note' and e['tick'] // tps == step_t]
                for cev in chord:
                    out.send_message([0x90 | ch, cev['note'], cev['vel']])
                    threading.Timer(0.30, self._note_off,
                                    args=(out, ch, cev['note'])).start()
            elif t == 'cc':
                out.send_message([0xB0 | ch, ev['cc'], ev['val']])
            elif t == 'prog':
                out.send_message([0xC0 | ch, ev['program']])
            elif t == 'bank':
                out.send_message([0xB0 | ch, 0, ev['msb']])
        except Exception:
            pass

    def nav_event_in_step(self, direction):
        """Cicla cur_event entre eventos del mismo step (para editar uno u otro)."""
        with self._lock:
            ct = self.song['cur_track']
            evs = self.song['tracks'][ct]['events']
            if not evs or self.song['cur_event'] < 0:
                return
            tps = ticks_per_step(self.song['resolution'])
            cur = self.song['cur_event']
            cur_step = evs[cur]['tick'] // tps
            same = [i for i, ev in enumerate(evs)
                    if ev['tick'] // tps == cur_step]
            if len(same) <= 1:
                return
            idx = same.index(cur)
            d = 1 if direction == 'next' else -1
            self.song['cur_event'] = same[(idx + d) % len(same)]
            self._clamp_cur_field()
        self._preview_event(self._selected_event())

    def nav_event(self, direction):
        with self._lock:
            ct = self.song['cur_track']
            evs = self.song['tracks'][ct]['events']
            if not evs:
                return
            cur = self.song['cur_event']
            tick = self.song['locator_tick']
            if cur < 0:
                # Sin selección: encontrar evento más cercano al locator
                if direction == 'up':
                    # primer evento con tick < locator
                    cur = -1
                    for i in range(len(evs) - 1, -1, -1):
                        if evs[i]['tick'] < tick:
                            cur = i; break
                    if cur < 0:
                        cur = 0
                else:  # down
                    cur = len(evs) - 1
                    for i, ev in enumerate(evs):
                        if ev['tick'] > tick:
                            cur = i; break
            else:
                if direction == 'up':
                    cur = max(0, cur - 1)
                elif direction == 'down':
                    cur = min(len(evs) - 1, cur + 1)
            self.song['cur_event'] = cur
            self._clamp_cur_field()
            self._sync_locator_from_event()
        self._preview_event(self._selected_event())

    def nav_field(self, direction):
        with self._lock:
            d = -1 if direction == 'prev' else 1
            ev = self._selected_event()
            t = ev['type'] if ev else 'note'
            fields = RIBBON_FIELDS.get(t, RIBBON_FIELDS['note'])
            valid = [i for i, (lbl, _) in enumerate(fields) if lbl]
            if not valid:
                return
            cur = self.song['cur_field']
            if cur not in valid:
                self.song['cur_field'] = valid[0] if d > 0 else valid[-1]
                return
            idx = valid.index(cur)
            self.song['cur_field'] = valid[(idx + d) % len(valid)]

    def toggle_record(self):
        with self._lock:
            self.recording = not self.recording
            print(f"[REC] {'ON' if self.recording else 'OFF'}")

    def toggle_view(self):
        with self._lock:
            self.song['view'] = 'macro' if self.song.get('view', 'micro') == 'micro' else 'micro'

    SKINS = ['modern', 'qx3']
    def cycle_skin(self):
        with self._lock:
            cur = self.song.get('skin', 'modern')
            try: idx = self.SKINS.index(cur)
            except ValueError: idx = 0
            self.song['skin'] = self.SKINS[(idx + 1) % len(self.SKINS)]

    # ── Transport ───────────────────────────────────────────────────────────
    def play(self):
        with self._lock:
            if self.running:
                return
            self.running = True
            self._play_start_tick = self.song['locator_tick']
            self._play_start_time = time.perf_counter()
            self._last_emit_tick = [self._play_start_tick - 1
                                     for _ in self.song['tracks']]
        threading.Thread(target=self._play_loop, daemon=True).start()

    def toggle_play(self):
        if self.running:
            self.stop()
        else:
            self.play()

    def stop(self):
        with self._lock:
            self.running = False
        # All notes off en todos los canales y outs
        for out in self._outs:
            if not out: continue
            for ch in range(16):
                try:
                    out.send_message([0xB0 | ch, 123, 0])
                except Exception:
                    pass

    def rewind(self):
        with self._lock:
            self.song['locator_tick'] = 0
            self._sync_event_from_locator()
            # Reset routing per-pista (próximos NOTE re-aplican sus cambios)
            self._track_routing = [{'port':None, 'program':None, 'bank':None}
                                    for _ in range(8)]
            if self.running:
                self._play_start_tick = 0
                self._play_start_time = time.perf_counter()
                self._last_emit_tick = [-1 for _ in self.song['tracks']]

    def fast_forward(self):
        with self._lock:
            self.song['locator_tick'] += self._bar_ticks()
            self._sync_event_from_locator()

    def _bar_ticks(self):
        # Un bar dura siempre lo mismo: TPQ * num_beats (independiente de res)
        return TPQ * self.song['time_sig'][0]

    def extend_selection(self, direction, granularity='step'):
        """Selección con anchor fijo: el locator es el extremo móvil.
        sel = [min(anchor, locator), max(anchor, locator)] + step en cada extremo.
        Si el locator está fuera de la sel actual, reancla (evita saltos raros)."""
        with self._lock:
            step = ticks_per_step(self.song['resolution'])
            delta = self._bar_ticks() if granularity == 'bar' else step
            cur = self.song['locator_tick']
            new_cur = max(0, cur - delta) if direction == 'up' else cur + delta
            sel = self.song.get('sel')
            # Conservar anchor SOLO si el locator está dentro de la sel actual
            if sel and 'anchor' in sel and sel['tick_start'] <= cur < sel['tick_end']:
                anchor = sel['anchor']
            else:
                anchor = cur
            lo = min(anchor, new_cur)
            hi = max(anchor, new_cur) + step
            self.song['sel'] = {'tick_start': lo, 'tick_end': hi, 'anchor': anchor}
            self.song['locator_tick'] = new_cur
            self._sync_event_from_locator()

    def clear_selection(self):
        with self._lock:
            self.song['sel'] = None

    def set_bpm(self, bpm):
        with self._lock:
            self.song['bpm'] = max(20.0, min(300.0, float(bpm)))

    def quantize(self, all_tracks=False):
        """Snap eventos al grid de la resolución actual. Sobre selección si hay,
        si no sobre el bar actual."""
        with self._lock:
            step = ticks_per_step(self.song['resolution'])
            lo, hi = self._sel_range()
            tracks = self.song['tracks'] if all_tracks else [self.song['tracks'][self.song['cur_track']]]
            for tr in tracks:
                for e in tr['events']:
                    if lo <= e['tick'] < hi:
                        e['tick'] = round(e['tick'] / step) * step
                tr['events'].sort(key=lambda e: e['tick'])
            self.dirty = True

    def goto_bar(self, bar):
        with self._lock:
            bar = max(0, int(bar) - 1)
            self.song['locator_tick'] = bar * self._bar_ticks()
            self._sync_event_from_locator()

    def _range_bounds(self, from_bar=None, to_bar=None,
                       tick_start=None, tick_end=None):
        """Calcula (lo, hi) en ticks. tick_start/end tienen prioridad sobre bars."""
        if tick_start is not None and tick_end is not None:
            return int(tick_start), int(tick_end)
        bt = self._bar_ticks()
        lo = max(0, int(from_bar)-1) * bt
        hi = max(int(from_bar), int(to_bar)-1) * bt
        return lo, hi

    def copy_range(self, from_bar=1, to_bar=2, from_track=1, to_track=1,
                    times=1, mode='merge', tick_start=None, tick_end=None):
        """TIMES = N copias adicionales."""
        with self._lock:
            lo, hi = self._range_bounds(from_bar, to_bar, tick_start, tick_end)
            span = hi - lo
            if span <= 0:
                return
            src_t = max(0, min(7, int(from_track)-1))
            dst_t = max(0, min(7, int(to_track)-1))
            # Destino: por defecto inicio justo después del rango copiado
            base_tick = hi
            src_evs = self.song['tracks'][src_t]['events']
            to_copy = [dict(e) for e in src_evs if lo <= e['tick'] < hi]
            n_copies = max(1, int(times))
            if mode == 'replace':
                total = span * n_copies
                dst = self.song['tracks'][dst_t]
                dst['events'] = [e for e in dst['events']
                                  if not (base_tick <= e['tick'] < base_tick + total)]
            for n in range(n_copies):
                offset = base_tick + n * span - lo
                for e in to_copy:
                    ne = dict(e); ne['tick'] = e['tick'] + offset
                    self._insert_sorted(dst_t, ne)
            self.dirty = True

    def loop_range(self, from_bar=1, to_bar=2, times=2, mode='merge',
                    tick_start=None, tick_end=None):
        """TIMES = total de instancias resultantes (incluye original)."""
        with self._lock:
            lo, hi = self._range_bounds(from_bar, to_bar, tick_start, tick_end)
            span = hi - lo
            if span <= 0:
                return
            extra = max(0, int(times) - 1)
            if extra == 0:
                return
            for tr in self.song['tracks']:
                to_copy = [dict(e) for e in tr['events'] if lo <= e['tick'] < hi]
                if mode == 'replace':
                    dst_end = hi + extra * span
                    tr['events'] = [e for e in tr['events']
                                    if not (hi <= e['tick'] < dst_end)]
                for n in range(1, extra + 1):
                    for e in to_copy:
                        ne = dict(e); ne['tick'] = e['tick'] + n * span
                        i = 0
                        while i < len(tr['events']) and tr['events'][i]['tick'] <= ne['tick']:
                            i += 1
                        tr['events'].insert(i, ne)
            self.dirty = True

    def transform_range(self, op, from_bar=1, to_bar=2, track_idx=None,
                         tick_start=None, tick_end=None, **params):
        """Transforma eventos en rango. Acepta bars o ticks directos."""
        with self._lock:
            lo, hi = self._range_bounds(from_bar, to_bar, tick_start, tick_end)
            if hi <= lo:
                return
            span = hi - lo
            if track_idx is None or int(track_idx) == 0:
                tracks = self.song['tracks']
            else:
                tracks = [self.song['tracks'][max(0, min(7, int(track_idx)-1))]]
            for tr in tracks:
                self._apply_transform(tr, op, lo, hi, span, params)
            self.dirty = True

    def _apply_transform(self, tr, op, lo, hi, span, params):
        evs_in_range = [(i, e) for i, e in enumerate(tr['events']) if lo <= e['tick'] < hi]
        if not evs_in_range:
            return
        if op == 'reverse':
            for _, ev in evs_in_range:
                ln = int(ev.get('len_ticks', 1)) if ev.get('type') == 'note' else 0
                ev['tick'] = max(lo, lo + hi - ev['tick'] - max(1, ln))
            tr['events'].sort(key=lambda e: e['tick'])
        elif op == 'stretch':
            factor = float(params.get('factor', 1.0))
            if factor <= 0: return
            for _, ev in evs_in_range:
                rel = ev['tick'] - lo
                ev['tick'] = lo + int(rel * factor)
            tr['events'].sort(key=lambda e: e['tick'])
        elif op == 'shift':
            shift_ticks = int(params.get('shift_ticks', 0))
            for _, ev in evs_in_range:
                ev['tick'] = max(0, ev['tick'] + shift_ticks)
            tr['events'].sort(key=lambda e: e['tick'])
        elif op == 'vel_ramp':
            vfrom = int(params.get('from_vel', 30))
            vto = int(params.get('to_vel', 120))
            notes = [e for _, e in evs_in_range if e.get('type') == 'note']
            if not notes: return
            for ev in notes:
                t01 = (ev['tick'] - lo) / max(1, span)
                ev['vel'] = max(1, min(127, int(vfrom + (vto - vfrom) * t01)))
        elif op == 'humanize':
            t_amt = int(params.get('timing_ticks', 0))
            v_amt = int(params.get('vel_amount', 0))
            for _, ev in evs_in_range:
                if t_amt > 0:
                    ev['tick'] = max(0, ev['tick'] + random.randint(-t_amt, t_amt))
                if v_amt > 0 and ev.get('type') == 'note':
                    ev['vel'] = max(1, min(127, ev['vel'] + random.randint(-v_amt, v_amt)))
            tr['events'].sort(key=lambda e: e['tick'])
        elif op == 'scale_quantize':
            root = int(params.get('root', 0))   # 0..11 (C..B)
            scale = params.get('scale', 'major')
            scales = {
                'major':[0,2,4,5,7,9,11],
                'minor':[0,2,3,5,7,8,10],
                'dorian':[0,2,3,5,7,9,10],
                'pent_maj':[0,2,4,7,9],
                'pent_min':[0,3,5,7,10],
                'blues':[0,3,5,6,7,10],
                'chromatic':list(range(12)),
            }
            sc = scales.get(scale, scales['major'])
            for _, ev in evs_in_range:
                if ev.get('type') != 'note': continue
                rel = (ev['note'] - root) % 12
                # Encontrar el grado más cercano de la escala
                best = min(sc, key=lambda g: min(abs(g - rel), 12 - abs(g - rel)))
                ev['note'] = ev['note'] - rel + best
        elif op == 'thin':
            pct = max(0, min(100, int(params.get('percent', 30))))
            keep = []
            for i, ev in enumerate(tr['events']):
                in_r = lo <= ev['tick'] < hi
                if in_r and random.randint(0, 99) < pct:
                    continue
                keep.append(ev)
            tr['events'] = keep
        elif op == 'double_oct':
            offset = 12 if params.get('direction', 'up') == 'up' else -12
            new_notes = []
            for _, ev in evs_in_range:
                if ev.get('type') != 'note': continue
                ne = dict(ev)
                ne['note'] = max(0, min(127, ev['note'] + offset))
                new_notes.append(ne)
            for ne in new_notes:
                i = 0
                while i < len(tr['events']) and tr['events'][i]['tick'] <= ne['tick']:
                    i += 1
                tr['events'].insert(i, ne)
        elif op == 'set_length':
            steps = max(1, int(params.get('length_steps', 1)))
            for _, ev in evs_in_range:
                if ev.get('type') == 'note':
                    ev['len_steps'] = steps
                    ev.pop('len_ticks', None)
        elif op == 'legato':
            notes = sorted([e for _, e in evs_in_range if e.get('type') == 'note'],
                           key=lambda x: x['tick'])
            for i, ev in enumerate(notes):
                next_t = notes[i+1]['tick'] if i+1 < len(notes) else hi
                gap = max(1, next_t - ev['tick'])
                ev['len_ticks'] = gap
        elif op == 'gate_scale':
            factor = float(params.get('factor', 1.0))
            for _, ev in evs_in_range:
                if ev.get('type') == 'note':
                    ev['gate'] = max(0.05, min(0.99, ev.get('gate', 0.95) * factor))
        elif op == 'reverse_sample':
            on = params.get('reverse_on', True)
            for _, ev in evs_in_range:
                if ev.get('type') == 'note':
                    ev['sample_reverse'] = bool(on)
        elif op == 'sample_start':
            val = max(0.0, min(0.95, float(params.get('start_pct', 0)) / 100.0))
            for _, ev in evs_in_range:
                if ev.get('type') == 'note':
                    ev['sample_start'] = val
        elif op == 'attack':
            ms = max(0, min(2000, int(params.get('attack_ms', 0))))
            for _, ev in evs_in_range:
                if ev.get('type') == 'note':
                    ev['attack_ms'] = ms
        elif op == 'release':
            ms = max(0, min(2000, int(params.get('release_ms', 0))))
            for _, ev in evs_in_range:
                if ev.get('type') == 'note':
                    ev['release_ms'] = ms
        elif op == 'chop':
            # Cada nota: hit corto + espacio antes de la siguiente.
            g = max(0.05, min(0.99, float(params.get('chop_gate', 0.30))))
            steps = max(1, min(16, int(params.get('chop_steps', 2))))
            for _, ev in evs_in_range:
                if ev.get('type') == 'note':
                    ev['gate'] = g
                    ev['ratchet'] = 1
                    ev['len_steps'] = steps
                    ev.pop('len_ticks', None)

    def transform_move(self, from_bar=1, to_bar=2, src_track=1, dst_track=2,
                        tick_start=None, tick_end=None):
        with self._lock:
            lo, hi = self._range_bounds(from_bar, to_bar, tick_start, tick_end)
            src_t = max(0, min(7, int(src_track)-1))
            dst_t = max(0, min(7, int(dst_track)-1))
            src = self.song['tracks'][src_t]
            dst = self.song['tracks'][dst_t]
            move = [e for e in src['events'] if lo <= e['tick'] < hi]
            src['events'] = [e for e in src['events'] if not (lo <= e['tick'] < hi)]
            for ev in move:
                self._insert_sorted(dst_t, ev)
            self.dirty = True

    def erase_range(self, from_bar=1, to_bar=2, track_idx=None,
                    event_type='all', note_lo=0, note_hi=127, cc_num=-1,
                    tick_start=None, tick_end=None):
        """Borra eventos en rango. Acepta bars o ticks directos."""
        with self._lock:
            lo, hi = self._range_bounds(from_bar, to_bar, tick_start, tick_end)
            if track_idx is None:
                tracks = self.song['tracks']
            else:
                tracks = [self.song['tracks'][max(0, min(7, int(track_idx)-1))]]
            for tr in tracks:
                kept = []
                for e in tr['events']:
                    if not (lo <= e['tick'] < hi):
                        kept.append(e); continue
                    t = e.get('type', 'note')
                    # Filtro por tipo
                    if event_type not in ('all', t):
                        kept.append(e); continue
                    # Filtro extra
                    if t == 'note':
                        n = e.get('note', 0)
                        if not (note_lo <= n <= note_hi):
                            kept.append(e); continue
                    elif t == 'cc' and cc_num >= 0 and e.get('cc') != cc_num:
                        kept.append(e); continue
                    # Borrar (no añadir a kept)
                tr['events'] = kept
            self.dirty = True

    # ── Playback loop ───────────────────────────────────────────────────────
    def _ticks_per_sec(self):
        return self.song['bpm'] * TPQ / 60.0

    def _play_loop(self):
        last_poke = 0.0
        while self.running:
            tps = self._ticks_per_sec()
            elapsed = time.perf_counter() - self._play_start_time
            cur_tick = self._play_start_tick + int(elapsed * tps)
            # Loop sobre selección si loop_on + sel activa
            with self._lock:
                sel = self.song['sel']
                loop = self.loop_on and sel is not None
                if loop and cur_tick >= sel['tick_end']:
                    self._play_start_tick = sel['tick_start']
                    self._play_start_time = time.perf_counter()
                    self._last_emit_tick = [sel['tick_start'] - 1
                                             for _ in self.song['tracks']]
                    cur_tick = sel['tick_start']
                tracks = list(self.song['tracks'])
            # Solo logic: si alguna pista tiene solo, las que no lo tengan se silencian
            any_solo = any(t.get('solo') for t in tracks)
            for ti, track in enumerate(tracks):
                if track['mute']:
                    continue
                if any_solo and not track.get('solo'):
                    continue
                last = self._last_emit_tick[ti]
                for ev in track['events']:
                    if last < ev['tick'] <= cur_tick:
                        self._emit_event(track, ev)
                self._last_emit_tick[ti] = cur_tick
            with self._lock:
                self.song['locator_tick'] = cur_tick
            now = time.perf_counter()
            if now - last_poke > 0.016:   # ~60fps de poke a SSE
                _poke()
                last_poke = now
            time.sleep(0.003)

    def _emit_event(self, track, ev):
        # Track tipo sampler: dispara sample local en lugar de MIDI
        if track.get('type') == 'sampler' and ev.get('type') == 'note':
            self._sampler_play_note(track, ev)
            return
        ch = track['channel']
        port = track['port']
        out = self._outs[port] if port < len(self._outs) else None
        if not out:
            return
        # NOTE: aplicar bank/program inline (estado vigente por pista)
        ti = self.song['tracks'].index(track) if track in self.song['tracks'] else 0
        rt = self._track_routing[ti]
        if ev.get('type') == 'note':
            new_bank = ev.get('bank')
            new_prog = ev.get('program')
            if new_bank is not None and new_bank != rt.get('bank'):
                rt['bank'] = new_bank
                try: out.send_message([0xB0|ch, 0, new_bank])
                except: pass
            if new_prog is not None and new_prog != rt.get('program'):
                rt['program'] = new_prog
                try: out.send_message([0xC0|ch, new_prog])
                except: pass
        t = ev.get('type', 'note')
        if t == 'note':
            if random.random() > ev.get('prob', 1.0):
                return
            note, vel = ev['note'], ev['vel']
            res = ev.get('res', 16)
            ratchet = max(1, int(ev.get('ratchet', 1)))
            rdiv = max(0.05, float(ev.get('rdiv', 1.0)))
            gate = max(0.05, min(0.99, ev.get('gate', 0.95)))
            # Duración total del evento (SIN aplicar gate aún)
            if 'len_ticks' in ev and ev['len_ticks']:
                total_sec = ev['len_ticks'] / self._ticks_per_sec()
            else:
                len_ticks = ev.get('len_steps', 1) * ticks_per_step(res)
                total_sec = len_ticks / self._ticks_per_sec()
            total_sec = max(0.01, total_sec)
            # Ratchet: N hits "cuadrados" — cada hit ocupa total/ratchet
            hit_interval = (total_sec / ratchet) * rdiv
            hit_dur = max(0.005, hit_interval * gate)
            for r in range(ratchet):
                delay = r * hit_interval
                if delay == 0:
                    try: out.send_message([0x90 | ch, note, vel])
                    except: return
                else:
                    threading.Timer(delay, self._send_note_on,
                                    args=(out, ch, note, vel)).start()
                threading.Timer(delay + hit_dur, self._note_off,
                                args=(out, ch, note)).start()
        elif t == 'cc':
            try:
                out.send_message([0xB0 | ch, ev['cc'], ev['val']])
            except Exception:
                pass
        elif t == 'prog':
            try:
                out.send_message([0xC0 | ch, ev['program']])
            except Exception:
                pass
        elif t == 'bank':
            try:
                out.send_message([0xB0 | ch, 0, ev['msb']])
            except Exception:
                pass

    def _note_off(self, out, ch, note):
        try:
            out.send_message([0x80 | ch, note, 0])
        except Exception:
            pass

    def _send_note_on(self, out, ch, note, vel):
        try:
            out.send_message([0x90 | ch, note, vel])
        except Exception:
            pass

    def move_locator(self, direction):
        with self._lock:
            res = self.song['resolution']
            step = ticks_per_step(res)
            if direction == 'left':
                self.song['locator_tick'] = max(0, self.song['locator_tick'] - step)
            elif direction == 'right':
                self.song['locator_tick'] += step
            elif direction == 'bar_left':
                self.song['locator_tick'] = max(0, self.song['locator_tick'] - self._bar_ticks())
            elif direction == 'bar_right':
                self.song['locator_tick'] += self._bar_ticks()
            self._sync_event_from_locator()

    def change_resolution(self, direction):
        with self._lock:
            try:
                idx = RESOLUTIONS.index(self.song['resolution'])
            except ValueError:
                idx = RESOLUTIONS.index(16)
            if direction == 'up':
                idx = min(len(RESOLUTIONS) - 1, idx + 1)
            else:
                idx = max(0, idx - 1)
            self._set_resolution(RESOLUTIONS[idx])

    def _res_base_str(self, res):
        if isinstance(res, str) and (res.endswith('T') or res.endswith('.')):
            return res[:-1]
        return str(res)

    def toggle_triplet(self):
        with self._lock:
            res = self.song['resolution']
            base = self._res_base_str(res)
            if isinstance(res, str) and res.endswith('T'):
                new = int(base) if base.isdigit() else base
            else:
                new = base + 'T'
            if new in RESOLUTIONS:
                self._set_resolution(new)

    def toggle_dotted(self):
        with self._lock:
            res = self.song['resolution']
            base = self._res_base_str(res)
            if isinstance(res, str) and res.endswith('.'):
                new = int(base) if base.isdigit() else base
            else:
                new = base + '.'
            if new in RESOLUTIONS:
                self._set_resolution(new)

    def set_resolution(self, res):
        with self._lock:
            # res puede llegar como string '4' o int o '4T'
            if isinstance(res, str) and not res.endswith('T'):
                try: res = int(res)
                except: pass
            if res in RESOLUTIONS:
                self._set_resolution(res)

    def _set_resolution(self, res):
        self.song['resolution'] = res
        step = ticks_per_step(res)
        self.song['locator_tick'] = (self.song['locator_tick'] // step) * step
        self._sync_event_from_locator()

    # ── Edición ─────────────────────────────────────────────────────────────
    def _insert_event(self, ev, advance=True):
        """Inserta evento ordenado por tick. Avanza locator si advance=True."""
        ct = self.song['cur_track']
        evs = self.song['tracks'][ct]['events']
        i = 0
        while i < len(evs) and evs[i]['tick'] <= ev['tick']:
            i += 1
        evs.insert(i, ev)
        self.song['cur_event'] = i
        self.song['cur_field'] = 0
        if advance:
            res = ev.get('res', self.song['resolution'])
            self.song['locator_tick'] += ticks_per_step(res)
        self.dirty = True

    def _apply_track_routing_to_note(self, ev, track_idx):
        """Asigna program/bank al NOTE event según current_program/bank de la
        pista. Usado por todas las rutas de inserción de notas."""
        t = self.song['tracks'][track_idx]
        ttype = t.get('type', 'midi')
        cur_prog = t.get('current_program', 0)
        cur_bank = t.get('current_bank', 0)
        if ttype == 'sampler':
            ev['program'] = cur_prog
        else:  # midi
            rt = self._track_routing[track_idx]
            if rt.get('program') != cur_prog:
                ev['program'] = cur_prog
                rt['program'] = cur_prog
            if rt.get('bank') != cur_bank:
                ev['bank'] = cur_bank
                rt['bank'] = cur_bank

    def insert_at_locator(self, note, vel):
        with self._lock:
            res = self.song['resolution']
            ev = _new_note(self.song['locator_tick'], res, note, vel)
            self._apply_track_routing_to_note(ev, self.song['cur_track'])
            self._insert_event(ev)

    def insert_cc(self, cc=74, val=64):
        with self._lock:
            res = self.song['resolution']
            self._insert_event(_new_cc(self.song['locator_tick'], res, cc, val))

    def insert_prog(self, program=0):
        with self._lock:
            self._insert_event(_new_prog(self.song['locator_tick'], program))

    def insert_bank(self, msb=0):
        with self._lock:
            self._insert_event(_new_bank(self.song['locator_tick'], msb))

    def _sel_range(self):
        """Devuelve (tick_start, tick_end) — selección o bar actual si no hay."""
        sel = self.song['sel']
        if sel:
            return sel['tick_start'], sel['tick_end']
        # Sin selección → bar actual completo
        bt = self._bar_ticks()
        bar_start = (self.song['locator_tick'] // bt) * bt
        return bar_start, bar_start + bt

    def delete_current_event(self):
        """Borra solo el evento bajo el cursor sin tocar la selección."""
        with self._lock:
            ct = self.song['cur_track']
            evs = self.song['tracks'][ct]['events']
            idx = self.song['cur_event']
            if not (0 <= idx < len(evs)):
                return
            evs.pop(idx)
            self.song['cur_event'] = max(-1, min(len(evs)-1, idx))
            self.dirty = True

    def delete_selected(self, all_tracks=False):
        with self._lock:
            sel = self.song['sel']
            tracks = self.song['tracks'] if all_tracks else [self.song['tracks'][self.song['cur_track']]]
            if sel is None and not all_tracks:
                # Borrar solo el evento seleccionado (comportamiento legacy)
                evs = tracks[0]['events']
                if not evs or self.song['cur_event'] < 0 or self.song['cur_event'] >= len(evs):
                    return
                evs.pop(self.song['cur_event'])
                self.song['cur_event'] = max(-1, min(len(evs) - 1, self.song['cur_event']))
                return
            lo, hi = self._sel_range()
            for tr in tracks:
                tr['events'] = [e for e in tr['events'] if not (lo <= e['tick'] < hi)]
            self.song['cur_event'] = -1

    def copy_selected(self, all_tracks=False):
        with self._lock:
            sel = self.song['sel']
            if sel is None and not all_tracks:
                # Single event
                ct = self.song['cur_track']
                evs = self.song['tracks'][ct]['events']
                if not evs or self.song['cur_event'] < 0 or self.song['cur_event'] >= len(evs):
                    return
                self.clipboard = {'mode': 'single',
                                  'ev': dict(evs[self.song['cur_event']])}
                return
            lo, hi = self._sel_range()
            span = hi - lo
            if all_tracks:
                tracks_evs = []
                for tr in self.song['tracks']:
                    tracks_evs.append([dict(e, tick=e['tick']-lo)
                                       for e in tr['events'] if lo <= e['tick'] < hi])
                self.clipboard = {'mode':'range_all', 'span':span, 'tracks':tracks_evs}
            else:
                ct = self.song['cur_track']
                evs = self.song['tracks'][ct]['events']
                evs_copy = [dict(e, tick=e['tick']-lo) for e in evs if lo <= e['tick'] < hi]
                self.clipboard = {'mode':'range_track', 'span':span, 'events':evs_copy}

    def paste_at_locator(self, all_tracks=False):
        """Pega en locator. NO mueve locator ni borra selección."""
        with self._lock:
            cb = self.clipboard
            if not cb:
                return
            base_tick = self.song['locator_tick']
            if cb['mode'] == 'single':
                ct = self.song['cur_track']
                ev = dict(cb['ev'])
                ev['tick'] = base_tick
                self._insert_sorted(ct, ev)
            elif cb['mode'] == 'range_track':
                ct = self.song['cur_track']
                for e in cb['events']:
                    ev = dict(e); ev['tick'] = base_tick + e['tick']
                    self._insert_sorted(ct, ev)
            elif cb['mode'] == 'range_all':
                for ti, trk_evs in enumerate(cb['tracks']):
                    for e in trk_evs:
                        ev = dict(e); ev['tick'] = base_tick + e['tick']
                        self._insert_sorted(ti, ev)

    def _insert_sorted(self, ti, ev):
        evs = self.song['tracks'][ti]['events']
        i = 0
        while i < len(evs) and evs[i]['tick'] <= ev['tick']:
            i += 1
        evs.insert(i, ev)

    def transpose(self, semitones, all_tracks=False):
        """Transpose de NOTE events. Sobre selección si hay, o evento seleccionado."""
        with self._lock:
            sel = self.song['sel']
            if sel is None and not all_tracks:
                ev = self._selected_event()
                if ev and ev.get('type') == 'note':
                    ev['note'] = max(0, min(127, ev['note'] + semitones))
                return
            lo, hi = self._sel_range()
            tracks = self.song['tracks'] if all_tracks else [self.song['tracks'][self.song['cur_track']]]
            for tr in tracks:
                for e in tr['events']:
                    if lo <= e['tick'] < hi and e.get('type') == 'note':
                        e['note'] = max(0, min(127, e['note'] + semitones))

    def insert_bars(self, n_bars):
        """Inserta N bars vacíos en locator desplazando todo posterior (TODAS pistas)."""
        with self._lock:
            n_bars = max(0, int(n_bars))
            if n_bars == 0:
                return
            bt = self._bar_ticks()
            shift = n_bars * bt
            cur = self.song['locator_tick']
            # Snap a inicio de bar
            cur = (cur // bt) * bt
            for tr in self.song['tracks']:
                for e in tr['events']:
                    if e['tick'] >= cur:
                        e['tick'] += shift

    def delete_bars(self, n_bars):
        """Borra N bars desde locator. Eventos en rango se borran;
        eventos posteriores se desplazan -N bars (TODAS pistas)."""
        with self._lock:
            n_bars = max(0, int(n_bars))
            if n_bars == 0:
                return
            bt = self._bar_ticks()
            shift = n_bars * bt
            cur = (self.song['locator_tick'] // bt) * bt
            for tr in self.song['tracks']:
                new = []
                for e in tr['events']:
                    if e['tick'] < cur:
                        new.append(e)
                    elif e['tick'] >= cur + shift:
                        e['tick'] -= shift
                        new.append(e)
                    # else: dentro del rango → descarta
                tr['events'] = new
            self._sync_event_from_locator()

    # ── Edición de campos del evento seleccionado ───────────────────────────
    # (lo, hi, step, kind)
    _FIELD_RANGES = {
        'note':      (0, 127, 1, 'int'),
        'vel':       (1, 127, 1, 'int'),
        'len_steps': (1, 64, 1, 'int'),
        'prob':      (0.0, 1.0, 0.05, 'float'),
        'micro':     (-0.5, 0.5, 0.05, 'float'),
        'gate':      (0.05, 0.99, 0.05, 'float'),
        'ratchet':   (1, 16, 1, 'int'),
        'rdiv':      (0.0, 2.0, 0.1, 'float'),
        'cc':        (0, 127, 1, 'int'),
        'val':       (0, 127, 1, 'int'),
        'ramp':      (0, 1, 1, 'bool'),
        'program':   (0, 127, 1, 'int'),
        'msb':       (0, 127, 1, 'int'),
        'bank':      (0, 127, 1, 'int'),
    }

    def edit_field(self, delta=0, absolute=None):
        """Edita el campo enfocado (cur_field) del evento seleccionado."""
        preview = None
        preview_key = None
        t = None
        with self._lock:
            ev = self._selected_event()
            if not ev:
                return
            t = ev.get('type', 'note')
            fields = RIBBON_FIELDS.get(t, [])
            fi = self.song['cur_field']
            if fi >= len(fields):
                return
            lbl, key = fields[fi]
            if not lbl or key == '_':
                return
            cur = ev.get(key)
            rng = self._FIELD_RANGES.get(key)
            if not rng:
                return
            if cur is None and key in ('program','bank'):
                cur = 0
            elif cur is None:
                return
            lo, hi, step, kind = rng
            if absolute is not None:
                nv = lo + (absolute / 127.0) * (hi - lo)
            else:
                nv = cur + delta * step
            nv = max(lo, min(hi, nv))
            if kind == 'int':
                ev[key] = int(round(nv))
            elif kind == 'bool':
                ev[key] = bool(int(round(nv)))
            else:
                ev[key] = round(nv, 2)
            preview = ev
            preview_key = key
        if preview and (t in ('cc','prog','bank') or
                        (t == 'note' and preview_key in ('note','program','bank'))):
            self._preview_event(preview)

    def duplicate_selected(self, all_tracks=False):
        """Si hay selección, duplica el rango justo después. Si no, copia evento."""
        with self._lock:
            sel = self.song['sel']
            if sel is None and not all_tracks:
                ct = self.song['cur_track']
                evs = self.song['tracks'][ct]['events']
                if not evs or self.song['cur_event'] < 0 or self.song['cur_event'] >= len(evs):
                    return
                src = evs[self.song['cur_event']]
                ev = dict(src)
                ev['tick'] = src['tick'] + ticks_per_step(self.song['resolution'])
                self._insert_sorted(ct, ev)
                # Mover cur_event y locator al nuevo (así el siguiente ⌘D avanza)
                self.song['cur_event'] = evs.index(ev)
                self.song['locator_tick'] = ev['tick']
                return
            lo, hi = self._sel_range()
            span = hi - lo
            tracks = self.song['tracks'] if all_tracks else [self.song['tracks'][self.song['cur_track']]]
            new_events_per_track = []
            for tr in tracks:
                added = []
                for e in list(tr['events']):
                    if lo <= e['tick'] < hi:
                        ne = dict(e); ne['tick'] = e['tick'] + span
                        added.append(ne)
                new_events_per_track.append((tr, added))
            for tr, added in new_events_per_track:
                for ev in added:
                    i = 0
                    while i < len(tr['events']) and tr['events'][i]['tick'] <= ev['tick']:
                        i += 1
                    tr['events'].insert(i, ev)
            # Mover selección al rango duplicado
            self.song['sel'] = {'tick_start': lo + span, 'tick_end': hi + span}
            self.song['locator_tick'] = lo + span
            self._sync_event_from_locator()

    # ── Display helpers ─────────────────────────────────────────────────────
    def _tick_to_locator(self, tick, res):
        tps = ticks_per_step(res)
        spb = steps_per_bar(res, self.song['time_sig'][0])
        total_steps = tick // tps
        bar = total_steps // spb
        step = total_steps % spb
        return f"{bar+1:03d}.{step+1:02d}.{res_short(res)}"

    def _event_row(self, ev):
        t = ev.get('type', 'note')
        res = ev.get('res', self.song['resolution'])
        loc = self._tick_to_locator(ev['tick'], res)
        if t == 'note':
            p_prog = ev.get('program')
            p_bank = ev.get('bank')
            return {'cells': [loc, 'NOTE',
                              note_name(ev['note']),
                              f"V{ev['vel']}",
                              f"{ev.get('len_steps',1)}st",
                              f"{int(ev['prob']*100)}%",
                              f"{int(ev['micro']*100):+d}%",
                              f"{int(ev['gate']*100)}%",
                              f"{ev.get('ratchet',1)}x",
                              f"{int(ev.get('rdiv',1.0)*100)}%",
                              str(p_prog) if p_prog is not None else '',
                              str(p_bank) if p_bank is not None else '']}
        if t == 'cc':
            return {'cells': [loc, 'CC', str(ev['cc']), str(ev['val']),
                              '', '', '', 'RAMP' if ev.get('ramp') else '',
                              '', '', '', '']}
        if t == 'prog':
            return {'cells': [loc, 'PROG', str(ev['program']),
                              '', '', '', '', '', '', '', '', '']}
        if t == 'bank':
            return {'cells': [loc, 'BANK', str(ev['msb']),
                              '', '', '', '', '', '', '', '', '']}
        return {'cells': [loc, '?', '', '', '', '', '', '', '', '', '', '']}

    def _selected_event(self):
        ct = self.song['cur_track']
        evs = self.song['tracks'][ct]['events']
        idx = self.song['cur_event']
        if not evs or idx < 0 or idx >= len(evs):
            return None
        return evs[idx]

    def _fmt_ribbon_val(self, ev_type, key, v):
        if key == '_': return '—'
        if v is None and key in ('port','program','bank'): return '—'
        if v is None: return '—'
        if ev_type == 'note':
            if key == 'note':       return note_name(int(v))
            if key == 'vel':        return str(v)
            if key == 'len_steps':  return f"{v}st"
            if key == 'prob':       return f"{int(v*100)}%"
            if key == 'micro':      return f"{int(v*100):+d}%"
            if key == 'gate':       return f"{int(v*100)}%"
            if key == 'ratchet':    return f"{v}x"
            if key == 'rdiv':       return f"{int(v*100)}%"
            if key == 'port':       return f"O{int(v)+1}"
            if key == 'program':    return str(v)
            if key == 'bank':       return str(v)
        if ev_type == 'cc':
            if key == 'cc':   return f"CC{v}"
            if key == 'val':  return str(v)
            if key == 'ramp': return 'ON' if v else 'OFF'
        return str(v)

    def _ribbon(self):
        ev = self._selected_event()
        # Tipo por defecto si no hay evento: NOTE (default insert)
        t = ev['type'] if ev else 'note'
        fields = RIBBON_FIELDS.get(t, RIBBON_FIELDS['note'])
        slots = []
        for lbl, key in fields:
            if not lbl:  # slot vacío
                slots.append({'label':'', 'value':'', 'disabled':True})
                continue
            if ev is None:
                slots.append({'label':lbl, 'value':'—', 'disabled':True})
            else:
                disp = self._fmt_ribbon_val(t, key, ev.get(key))
                slots.append({'label':lbl, 'value':disp, 'disabled':False})
        return slots

    # Ventana fija visible: locator en fila ROWS_BEFORE
    ROWS_TOTAL = 18
    ROWS_BEFORE = 5

    def _macro_view(self):
        """8 pistas × N steps. Rango visible se adapta a la selección si la hay."""
        res = self.song['resolution']
        tps = ticks_per_step(res)
        spb = steps_per_bar(res, self.song['time_sig'][0])
        loc = self.song['locator_tick']
        loc_step = loc // tps
        sel = self.song['sel']
        if sel:
            # Selección completa + medio bar de padding
            sel_s0 = sel['tick_start'] // tps
            sel_s1 = sel['tick_end'] // tps
            pad = spb // 2
            start_step = max(0, sel_s0 - pad)
            end_step = sel_s1 + pad
            min_total = spb * 2
            if (end_step - start_step) < min_total:
                end_step = start_step + min_total
        else:
            # Centrado en locator: 0.5 bar antes + 1.5 después = 2 bars total
            start_step = max(0, loc_step - spb // 2)
            end_step = start_step + spb * 2
        steps_total = end_step - start_step
        tracks = []
        for tr in self.song['tracks']:
            ev_steps = []
            for ev in tr['events']:
                s = ev['tick'] // tps
                if start_step <= s < end_step:
                    ev_steps.append(s - start_step)
            tracks.append({'steps': sorted(set(ev_steps)),
                           'has': bool(tr['events']),
                           'mute': tr['mute']})
        sel_range = None
        if sel:
            s0 = (sel['tick_start'] // tps) - start_step
            s1 = (sel['tick_end'] // tps) - start_step
            sel_range = [max(0, s0), min(steps_total, s1)]
        return {
            'total_steps': steps_total,
            'steps_per_bar': spb,
            'start_step': start_step,
            'tracks': tracks,
            'cur_track': self.song['cur_track'],
            'loc_step': max(0, min(steps_total, loc_step - start_step)),
            'sel_range': sel_range,
        }

    def _overview_old(self):
        # placeholder
        return None

    def _overview(self):
        """Mini-mapa por BARS (no por steps): siempre legible."""
        bt = self._bar_ticks()
        max_tick = 0
        for tr in self.song['tracks']:
            for ev in tr['events']:
                if ev['tick'] > max_tick:
                    max_tick = ev['tick']
        total_bars = max(8, (max_tick // bt) + 2)
        tracks = []
        for tr in self.song['tracks']:
            bars = set(ev['tick'] // bt for ev in tr['events'])
            tracks.append(sorted(bars))
        sel = self.song['sel']
        sel_range = None
        if sel:
            sel_range = [sel['tick_start']//bt, max(1, sel['tick_end']//bt)]
        loc_bar = self.song['locator_tick'] // bt
        return {
            'total_steps': total_bars,
            'steps_per_bar': 1,
            'tracks': tracks,
            'cur_track': self.song['cur_track'],
            'loc_step': loc_bar,
            'sel_range': sel_range,
        }

    def _overview_unused(self):
        """Resumen de todas las pistas como step-grid comprimido."""
        res = self.song['resolution']
        tps = ticks_per_step(res)
        # Total steps: hasta el último evento de cualquier pista (mín 1 bar)
        max_tick = 0
        for tr in self.song['tracks']:
            for ev in tr['events']:
                if ev['tick'] > max_tick:
                    max_tick = ev['tick']
        _spb = steps_per_bar(res, self.song['time_sig'][0])
        total_steps = max(_spb * 2, ((max_tick // tps) + 1 + _spb))
        # Por cada pista, set de step indices con eventos
        tracks = []
        for tr in self.song['tracks']:
            steps = set()
            for ev in tr['events']:
                steps.add(ev['tick'] // tps)
            tracks.append(sorted(steps))
        sel = self.song['sel']
        sel_range = None
        if sel:
            sel_range = [sel['tick_start'] // tps, sel['tick_end'] // tps]
        loc_step = self.song['locator_tick'] // tps
        return {
            'total_steps': total_steps,
            'steps_per_bar': _spb,
            'tracks': tracks,
            'cur_track': self.song['cur_track'],
            'loc_step': loc_step,
            'sel_range': sel_range,
        }

    def set_routing(self, idx, key, value):
        """Setea program/bank inline en un NOTE event. idx=-1 → cur_event."""
        with self._lock:
            ct = self.song['cur_track']
            evs = self.song['tracks'][ct]['events']
            if idx == -1:
                idx = self.song.get('cur_event', -1)
            if not (0 <= idx < len(evs)):
                return
            ev = evs[idx]
            if ev.get('type') != 'note':
                return
            if key in ('program', 'bank'):
                if value is not None:
                    value = max(0, min(127, int(value)))
                ev[key] = value
            self.dirty = True

    def select_event(self, idx, field=None):
        """Selecciona evento por índice + opcionalmente field (para clicks)."""
        with self._lock:
            ct = self.song['cur_track']
            evs = self.song['tracks'][ct]['events']
            if not (0 <= idx < len(evs)):
                return
            self.song['cur_event'] = idx
            self.song['locator_tick'] = evs[idx]['tick']
            if field is not None and 0 <= field < 10:
                self.song['cur_field'] = field
            self._clamp_cur_field()
        self._preview_event(self._selected_event())

    def _tracker_rows(self):
        """Ventana fija centrada en el locator. Locator siempre en la fila ROWS_BEFORE.
        No hay scroll: al avanzar el locator, las filas rotan."""
        ct = self.song['cur_track']
        evs = self.song['tracks'][ct]['events']
        res = self.song['resolution']
        tps = ticks_per_step(res)
        loc = self.song['locator_tick']
        loc_step_idx = loc // tps  # índice de step absoluto
        start_step = max(0, loc_step_idx - self.ROWS_BEFORE)
        sel = self.song['sel']
        by_tick = {}
        for idx, ev in enumerate(evs):
            # Snap visual a step contenedor (mantiene tick real intacto en evento)
            step_tick = (ev['tick'] // tps) * tps
            by_tick.setdefault(step_tick, []).append((idx, ev))
        rows = []
        cur_event_row = -1
        for i in range(self.ROWS_TOTAL):
            tick = (start_step + i) * tps
            here = by_tick.get(tick, [])
            is_cur = (tick <= loc < tick + tps)
            in_sel = bool(sel and sel['tick_start'] <= tick < sel['tick_end'])
            if here:
                # Ordenar por tick exacto. Mismo tick = mismo acorde visual.
                here_sorted = sorted(here, key=lambda x: x[1]['tick'])
                prev_tick = None
                for ev_idx, ev in here_sorted:
                    row = self._event_row(ev)
                    row['is_cur'] = is_cur
                    row['empty'] = False
                    row['in_sel'] = in_sel
                    row['ev_idx'] = ev_idx
                    # Oculta LOC si comparte tick con la anterior (acorde)
                    if prev_tick is not None and ev['tick'] == prev_tick:
                        row['cells'] = ['  ↳'] + list(row['cells'][1:])
                    prev_tick = ev['tick']
                    if ev_idx == self.song['cur_event']:
                        cur_event_row = len(rows)
                    rows.append(row)
            else:
                rows.append({'cells': [self._tick_to_locator(tick, res)] + ['']*11,
                             'is_cur': is_cur, 'empty': True,
                             'in_sel': in_sel, 'ev_idx': -1})
        return rows, cur_event_row

    def state(self):
        with self._lock:
            ct = self.song['cur_track']
            evs = self.song['tracks'][ct]['events']
            events_out, cur_row = self._tracker_rows()
            follow_row = cur_row
            return {
                'ts': time.time(),
                'bpm': self.song['bpm'],
                'resolution': self.song['resolution'],
                'locator': self._tick_to_locator(self.song['locator_tick'],
                                                  self.song['resolution']),
                'cur_track': ct,
                'cur_event': self.song['cur_event'],
                'cur_row': cur_row,
                'follow_row': follow_row,
                'cur_field': self.song['cur_field'],
                'running': self.running,
                'recording': self.recording,
                'loop_on': self.loop_on,
                'dirty': self.dirty,
                'last_saved_ts': self._last_saved_ts,
                'tpq': TPQ,
                'time_sig': list(self.song['time_sig']),
                'locator_tick': self.song['locator_tick'],
                'track_name': self.song['tracks'][ct]['name'],
                'track_mute': self.song['tracks'][ct]['mute'],
                'track_port': self.song['tracks'][ct]['port'],
                'track_type': self.song['tracks'][ct].get('type', 'midi'),
                'track_sample_dir': self.song['tracks'][ct].get('sample_dir', ''),
                'track_samples': self.song['tracks'][ct].get('samples', []),
                'track_current_program': self.song['tracks'][ct].get('current_program', 0),
                'flash': self._flash_label if (time.time() - self._flash_ts) < 1.5 else '',
                'modal_capture_armed': self._modal_capture_armed,
                'modal_captured_note': self._modal_captured_note,
                'modal_captured_ts': self._modal_captured_ts,
                'tracks_info': [{'has_events': bool(t['events']),
                                  'mute': t.get('mute', False),
                                  'solo': t.get('solo', False)}
                                 for t in self.song['tracks']],
                'overview': self._overview(),
                'view': self.song.get('view', 'micro'),
                'macro': self._macro_view(),
                'events': events_out,
                'ribbon': self._ribbon(),
                'has_clipboard': self.clipboard is not None,
                'sel': dict(self.song['sel']) if self.song['sel'] else None,
                'settings': self.get_settings_view(),
                'skin': self.song.get('skin', 'modern'),
            }


# ── Flask ────────────────────────────────────────────────────────────────────
app = Flask(__name__)
_seq = None
_state_event = threading.Event()


def _poke():
    _state_event.set()


@app.route('/')
def _index():
    return Response(_VISUAL_HTML, mimetype='text/html',
                    headers={'Cache-Control': 'no-store'})


_DISKET_PATH = '/Users/user/Downloads/Disket Rostype/Disket-Mono-Regular.ttf'


@app.route('/font/disket-mono.ttf')
def _font():
    try:
        with open(_DISKET_PATH, 'rb') as f:
            return Response(f.read(), mimetype='font/ttf',
                            headers={'Cache-Control': 'public, max-age=86400'})
    except Exception:
        return Response(status=404)


@app.route('/events')
def _events():
    def stream():
        last_json = ''
        s = _seq.state()
        j = json.dumps(s)
        last_json = j
        yield f"data: {j}\n\n"
        while True:
            fired = _state_event.wait(timeout=4.0)
            _state_event.clear()
            if fired:
                s = _seq.state()
                j = json.dumps(s)
                if j != last_json:
                    last_json = j
                    yield f"data: {j}\n\n"
            else:
                yield ": ping\n\n"
    return Response(stream(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'X-Accel-Buffering': 'no'})


@app.route('/api/state')
def _api_state():
    return Response(json.dumps(_seq.state()), mimetype='application/json')


@app.route('/api/pick_dir', methods=['POST'])
def _api_pick_dir():
    """Abre un dialog nativo macOS (AppleScript) para elegir carpeta."""
    import subprocess
    script = '''
    tell application "System Events" to activate
    set chosenFolder to choose folder with prompt "Select sample folder"
    POSIX path of chosenFolder
    '''
    try:
        result = subprocess.run(['osascript', '-e', script],
                                capture_output=True, text=True, timeout=120)
        path = result.stdout.strip()
        return Response(json.dumps({'path': path}), mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({'path': '', 'error': str(e)}),
                        mimetype='application/json')


@app.route('/api/song', methods=['GET', 'POST'])
def _api_song():
    if request.method == 'POST':
        data = request.get_json(force=True, silent=True) or {}
        try:
            _seq.load_from_dict(data)
            _poke()
            return Response('{"ok":true}', mimetype='application/json')
        except Exception as e:
            return Response(json.dumps({'ok':False,'error':str(e)}),
                            mimetype='application/json', status=400)
    # GET → devuelve song completo para download
    return Response(json.dumps(_seq.song, indent=2),
                    mimetype='application/json',
                    headers={'Content-Disposition': 'attachment; filename="song.json"'})


_MUTATING_CMDS = {'insert_note','insert_cc','insert_prog','insert_bank',
                   'delete','delete_event','copy','paste','duplicate','edit',
                   'transpose','insert_bars','delete_bars','quantize',
                   'copy_range','loop_range','erase_range','transform',
                   'set_routing','set_track_type','set_track_port',
                   'toggle_mute','toggle_solo','goto_bar','set_bpm',
                   'new','toggle_triplet','toggle_dotted','set_res',
                   'res'}
# Los que no destruyen el song (selección, navegación, etc.) no pushean undo
_UNDO_CMDS = _MUTATING_CMDS - {'copy','edit'}


@app.route('/api/cmd', methods=['POST'])
def _api_cmd():
    data = request.get_json(force=True, silent=True) or {}
    cmd = data.get('cmd')
    if cmd in _UNDO_CMDS:
        _seq.push_undo()
    if cmd == 'select_track':
        _seq.select_track(int(data.get('t', 0)))
    elif cmd == 'nav_event':
        _seq.nav_event(data.get('dir', ''))
    elif cmd == 'nav_event_in_step':
        _seq.nav_event_in_step(data.get('dir', 'next'))
    elif cmd == 'select_event':
        _f = data.get('field')
        _seq.select_event(int(data.get('idx', 0)),
                          field=int(_f) if _f is not None else None)
    elif cmd == 'audition':
        _seq._preview_event(_seq._selected_event())
    elif cmd == 'arm_note_capture':
        _seq._modal_capture_armed = bool(data.get('armed', True))
        if data.get('clear', False):
            _seq._modal_captured_note = None
    elif cmd == 'set_routing':
        v = data.get('value')
        _seq.set_routing(int(data.get('idx', 0)),
                          data.get('key', ''),
                          v if v is None else int(v))
    elif cmd == 'set_track_port':
        _seq.set_track_port(int(data.get('port', 0)))
    elif cmd == 'toggle_mute':
        _seq.toggle_mute()
    elif cmd == 'toggle_solo':
        _seq.toggle_solo()
    elif cmd == 'set_track_type':
        _seq.set_track_type(data.get('type', 'midi'),
                             sample_dir=data.get('sample_dir'))
    elif cmd == 'next_sample':
        _seq.next_sample(int(data.get('delta', 1)))
    elif cmd == 'nav_field':
        _seq.nav_field(data.get('dir', ''))
    elif cmd == 'locator':
        _seq.move_locator(data.get('dir', ''))
    elif cmd == 'res':
        _seq.change_resolution(data.get('dir', ''))
    elif cmd == 'set_res':
        _seq.set_resolution(data.get('res', 16))
    elif cmd == 'toggle_triplet':
        _seq.toggle_triplet()
    elif cmd == 'toggle_dotted':
        _seq.toggle_dotted()
    elif cmd == 'delete':
        _seq.delete_selected(all_tracks=bool(data.get('all', False)))
    elif cmd == 'delete_event':
        _seq.delete_current_event()
    elif cmd == 'copy':
        _seq.copy_selected(all_tracks=bool(data.get('all', False)))
    elif cmd == 'paste':
        _seq.paste_at_locator(all_tracks=bool(data.get('all', False)))
    elif cmd == 'duplicate':
        _seq.duplicate_selected(all_tracks=bool(data.get('all', False)))
    elif cmd == 'extend_sel':
        _seq.extend_selection(data.get('dir', 'down'),
                              granularity=data.get('gran', 'step'))
    elif cmd == 'clear_sel':
        _seq.clear_selection()
    elif cmd == 'transpose':
        _seq.transpose(int(data.get('semi', 0)),
                       all_tracks=bool(data.get('all', False)))
    elif cmd == 'insert_bars':
        _seq.insert_bars(int(data.get('n', 1)))
    elif cmd == 'delete_bars':
        _seq.delete_bars(int(data.get('n', 1)))
    elif cmd == 'save':
        _seq.save()
    elif cmd == 'new':
        _seq.new_song()
    elif cmd == 'set_bpm':
        _seq.set_bpm(float(data.get('bpm', 120)))
    elif cmd == 'quantize':
        _seq.quantize(all_tracks=bool(data.get('all', False)))
    elif cmd == 'cycle_port':
        slot = data.get('slot', '')
        if slot == 'skin':
            _seq.cycle_skin()
        else:
            _seq.cycle_port(slot)
    elif cmd == 'toggle_macro':
        _seq.toggle_view()
    elif cmd == 'undo':
        _seq.undo()
    elif cmd == 'redo':
        _seq.redo()
    elif cmd == 'toggle_loop':
        _seq.toggle_loop()
    elif cmd == 'set_loop_playback':
        _seq.set_loop_playback(data.get('from_bar',1), data.get('to_bar',2))
    elif cmd == 'insert_note':
        _seq.insert_at_locator(int(data.get('note', 60)),
                                int(data.get('vel', 100)))
    elif cmd == 'insert_cc':
        _seq.insert_cc(int(data.get('cc', 74)), int(data.get('val', 64)))
    elif cmd == 'insert_prog':
        _seq.insert_prog(int(data.get('program', 0)))
    elif cmd == 'insert_bank':
        _seq.insert_bank(int(data.get('msb', 0)))
    elif cmd == 'edit':
        _seq.edit_field(delta=int(data.get('delta', 0)))
    elif cmd == 'toggle_record':
        _seq.toggle_record()
    elif cmd == 'play':
        _seq.toggle_play()
    elif cmd == 'stop':
        _seq.stop()
    elif cmd == 'rewind':
        _seq.rewind()
    elif cmd == 'fast_forward':
        _seq.fast_forward()
    elif cmd == 'goto_bar':
        _seq.goto_bar(int(data.get('bar', 1)))
    elif cmd == 'copy_range':
        _seq.copy_range(from_bar=data.get('from_bar',1),
                        to_bar=data.get('to_bar',2),
                        from_track=data.get('from_track',1),
                        to_track=data.get('to_track',1),
                        times=data.get('times',1),
                        mode=data.get('mode','merge'),
                        tick_start=data.get('tick_start'),
                        tick_end=data.get('tick_end'))
    elif cmd == 'loop_range':
        _seq.loop_range(from_bar=data.get('from_bar',1),
                        to_bar=data.get('to_bar',2),
                        times=data.get('times',2),
                        mode=data.get('mode','merge'),
                        tick_start=data.get('tick_start'),
                        tick_end=data.get('tick_end'))
    elif cmd == 'transform':
        op = data.get('op', '')
        ti = data.get('track')
        ts = data.get('tick_start')
        te = data.get('tick_end')
        if op == 'move':
            _seq.transform_move(from_bar=data.get('from_bar',1),
                                 to_bar=data.get('to_bar',2),
                                 src_track=data.get('src_track', 1),
                                 dst_track=data.get('dst_track', 2),
                                 tick_start=ts, tick_end=te)
        else:
            params = {}
            for k in ('from_vel','to_vel','timing_ticks','vel_amount',
                       'root','percent','length_steps','shift_ticks',
                       'chop_steps','start_pct','attack_ms','release_ms'):
                if k in data: params[k] = int(data[k])
            for k in ('scale','direction'):
                if k in data: params[k] = str(data[k])
            if 'reverse_on' in data:
                params['reverse_on'] = data['reverse_on'] in ('on', True, 1, 'ON')
            if 'factor' in data:
                params['factor'] = float(data['factor']) / 100.0
            if 'chop_gate' in data:
                params['chop_gate'] = float(data['chop_gate']) / 100.0
            _seq.transform_range(op,
                                  from_bar=data.get('from_bar',1),
                                  to_bar=data.get('to_bar',2),
                                  track_idx=ti,
                                  tick_start=ts, tick_end=te,
                                  **params)
    elif cmd == 'erase_range':
        ti = data.get('track')
        _seq.erase_range(from_bar=data.get('from_bar',1),
                          to_bar=data.get('to_bar',2),
                          track_idx=ti,
                          event_type=data.get('event_type', 'all'),
                          note_lo=int(data.get('note_lo', 0)),
                          note_hi=int(data.get('note_hi', 127)),
                          cc_num=int(data.get('cc_num', -1)),
                          tick_start=data.get('tick_start'),
                          tick_end=data.get('tick_end'))
    if cmd in _MUTATING_CMDS:
        _seq.dirty = True
    _poke()
    return Response('{}', mimetype='application/json')


# ── HTML inline ─────────────────────────────────────────────────────────────
_VISUAL_HTML = r'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>rrrlinear</title>
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=yes">
<style>
@font-face{font-family:'Disket Mono';src:url('/font/disket-mono.ttf') format('truetype');font-weight:400;font-style:normal}
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:100%;height:100%;overflow:hidden;background:#0a0a0a;
  font-family:'Disket Mono',monospace;user-select:none;color:#fff;
  letter-spacing:0.04em}
#root{position:absolute;width:1440px;height:920px;transform-origin:top left;left:0;top:0}

/* ── Header (compacto) ── */
#hdr{position:absolute;left:32px;top:24px;right:32px;height:72px;
  display:flex;align-items:center;justify-content:space-between}
#hdr-title{font-size:24px;line-height:72px;color:#8C9AA3;letter-spacing:6px}
#hdr-locator{font-size:60px;line-height:72px;color:#FFD13A;letter-spacing:6px;
  transition:color .12s;font-weight:400}
#hdr-locator.playing{color:#5BD075}
#hdr-right{display:flex;align-items:center;gap:22px}
#hdr-res{font-size:22px;line-height:72px;color:#8C9AA3;letter-spacing:2px}
#hdr-loop{font-size:24px;line-height:72px;color:#3A4A53;letter-spacing:1px;
  transition:color .2s}
#hdr-loop.on{color:#5BD075;text-shadow:0 0 10px rgba(91,208,117,0.6)}
#hdr-rec{width:14px;height:14px;border-radius:50%;background:#FF3344;opacity:0.22;
  transition:opacity .2s}
#hdr-rec.on{opacity:1;animation:rec-blink 1s ease-in-out infinite}
@keyframes rec-blink{0%,100%{opacity:1}50%{opacity:0.4}}
#hdr-bpm{font-size:34px;line-height:72px;color:#FFFFFF;letter-spacing:2px}

/* ── Ribbon (10 slots) ── */
#ribbon{position:absolute;left:32px;top:108px;right:32px;height:78px;
  display:grid;grid-template-columns:repeat(10,1fr);gap:6px}
.rb-cell{position:relative;display:flex;flex-direction:column;padding:0 4px;
  justify-content:space-between}
.rb-v{font-size:24px;line-height:30px;color:#A8B0B8;white-space:nowrap;
  letter-spacing:1px}
.rb-cell.sel .rb-v{color:#FFB73A}
.rb-cell.disabled .rb-v{color:#3A4A53}
.rb-n{font-size:13px;line-height:18px;color:#8C9AA3;letter-spacing:2px;
  margin-top:6px}
.rb-cell.sel .rb-n{color:#FFB73A;
  text-decoration:underline;text-decoration-color:#FFB73A;
  text-underline-offset:4px;text-decoration-thickness:1px}
.rb-cell.disabled .rb-n{color:#3A4A53}

/* ── Track header (compacto) ── */
#trk-hdr{position:absolute;left:32px;top:280px;font-size:22px;line-height:28px;
  color:#FFFFFF;letter-spacing:3px;
  text-decoration:underline;text-underline-offset:5px;text-decoration-thickness:1px}
#trk-tabs{position:absolute;right:32px;top:282px;font-size:16px;color:#5A6A73;
  letter-spacing:3px;line-height:24px}
#trk-tabs span{padding:0 7px;color:#3A4A53;transition:color .15s}
#trk-tabs span.has{color:#A8B0B8}
#trk-tabs span.act{color:#FFD13A}
#trk-tabs span.muted{color:#5A3A3A;text-decoration:line-through}
#trk-tabs span.solo{color:#3AC0E0;text-shadow:0 0 6px rgba(58,192,224,0.6)}

/* ── Event list ── */
#evlist-hdr{position:absolute;left:32px;top:300px;right:32px;height:18px;
  display:grid;
  grid-template-columns:130px 60px 70px 70px 60px 70px 70px 70px 50px 70px 60px 60px;
  gap:12px;padding:0 8px;font-size:10px;color:#5A6A73;letter-spacing:2px;
  align-items:end}
#evlist-hdr .routing{color:#3AC0E0}
#evlist{position:absolute;left:32px;top:320px;right:32px;bottom:56px;
  overflow:hidden;font-size:17px;line-height:26px;color:#E0E4E8;
  letter-spacing:1px}
.ev-row{display:grid;
  grid-template-columns:130px 60px 70px 70px 60px 70px 70px 70px 50px 70px 60px 60px;
  gap:12px;padding:0 8px;white-space:nowrap;align-items:baseline}
.ev-cell.routing{color:#3AC0E0}
.ev-cell.routing:empty::before{content:'—';color:#2A4A53;font-size:12px}
.ev-row.sel{background:rgba(255,209,58,0.05)}
.ev-row.in-sel{background:rgba(91,208,117,0.10)}
.ev-row.in-sel.sel{background:rgba(255,209,58,0.10)}
.ev-row.follow{box-shadow:inset 4px 0 0 #5A6A73}
.ev-row.cur{background:rgba(255,209,58,0.18);box-shadow:inset 4px 0 0 #FFD13A}
.ev-row.cur.in-sel{background:linear-gradient(90deg, rgba(255,209,58,0.18) 60%, rgba(91,208,117,0.18))}
.ev-row.empty{color:#3A4A53}
.ev-row.empty .ev-cell:first-child{color:#5A6A73}
.ev-cell{position:relative;padding:0 2px;overflow:hidden;text-overflow:clip}
.ev-cell.kind{color:#8C9AA3}
.ev-row.sel .ev-cell.sel-col{outline:2px solid #3AC0E0;outline-offset:1px;
  background:rgba(58,192,224,0.08)}
.ev-empty{color:#3A4A53;padding:8px 0;font-size:14px;letter-spacing:2px}


/* ── Skin: QX3 (LCD ámbar genuino sobre carcasa negra mate) ── */
@font-face{font-family:'LCD';src:url('/font/disket-mono.ttf') format('truetype')}
body.skin-qx3{background:#000;}
body.skin-qx3 #root{background:#1a1a1a;
  background-image:linear-gradient(180deg, #2a2a2a 0%, #1a1a1a 30%, #0d0d0d 100%);
  box-shadow:inset 0 0 80px rgba(0,0,0,0.8)}

/* LCD: fondo ámbar real, texto oscuro */
body.skin-qx3 #hdr{
  background:#c89312;
  background-image:linear-gradient(180deg, #d4a020 0%, #c89312 100%);
  border:4px solid #0a0a0a;
  border-radius:4px;
  padding:0 32px;
  box-shadow:inset 0 0 24px rgba(140,90,0,0.4),
              0 4px 12px rgba(0,0,0,0.6),
              0 0 30px rgba(255,179,0,0.15);
  font-family:'LCD',monospace}
body.skin-qx3 #hdr-title{color:#2a1a00;text-shadow:0 0 2px rgba(0,0,0,0.3);
  letter-spacing:4px;font-weight:700}
body.skin-qx3 #hdr-locator{color:#1a0a00;text-shadow:0 0 4px rgba(255,179,0,0.4);
  font-weight:700}
body.skin-qx3 #hdr-locator.playing{color:#0a3a00;
  text-shadow:0 0 8px rgba(127,255,0,0.6)}
body.skin-qx3 #hdr-bpm{color:#2a1a00;font-weight:700}
body.skin-qx3 #hdr-res{color:#5a3500}
body.skin-qx3 #hdr-rec{background:#cc0000;
  box-shadow:0 0 8px rgba(204,0,0,0.6), inset 0 -2px 4px rgba(0,0,0,0.4)}
body.skin-qx3 #hdr-rec.on{background:#FF1010;
  box-shadow:0 0 14px rgba(255,30,30,0.9), inset 0 -2px 4px rgba(0,0,0,0.3)}

/* Ribbon (DATA/TRACK LEDs style row) - texto sobre fondo carcasa */
body.skin-qx3 #ribbon{padding:0 8px}
body.skin-qx3 #ribbon .rb-v{color:#5a3a10;text-shadow:0 0 2px rgba(255,179,0,0.2)}
body.skin-qx3 #ribbon .rb-cell.sel .rb-v{color:#FFB300;
  text-shadow:0 0 10px rgba(255,179,0,0.8)}
body.skin-qx3 #ribbon .rb-n{color:#9a7820;letter-spacing:3px}
body.skin-qx3 #ribbon .rb-cell.sel .rb-n{color:#FFD040;
  text-decoration-color:#FFD040}
body.skin-qx3 #ribbon .rb-cell.disabled .rb-v,
body.skin-qx3 #ribbon .rb-cell.disabled .rb-n{color:#3a2a10}

/* TRK header */
body.skin-qx3 #trk-hdr{color:#FFB300;
  text-shadow:0 0 8px rgba(255,179,0,0.5);
  text-decoration-color:#FFB300}

/* LED row (tracks) */
body.skin-qx3 #trk-tabs{
  padding:8px 12px;
  background:#0a0a0a;
  border:2px solid #2a2a2a;
  border-radius:4px;
  box-shadow:inset 0 0 8px rgba(0,0,0,0.8)}
body.skin-qx3 #trk-tabs span{
  color:#1a3a00;
  padding:0 10px;
  text-shadow:0 0 2px rgba(0,0,0,0.5)}
body.skin-qx3 #trk-tabs span.has{
  color:#7FFF00;
  text-shadow:0 0 6px rgba(127,255,0,0.9),
              0 0 14px rgba(127,255,0,0.5)}
body.skin-qx3 #trk-tabs span.act{
  color:#FFC840;
  text-shadow:0 0 8px rgba(255,200,64,0.9),
              0 0 14px rgba(255,179,0,0.7)}
body.skin-qx3 #trk-tabs span.muted{color:#5a1a1a}

/* Event list = LCD secundario ámbar */
body.skin-qx3 #evlist{
  color:#1a0a00;
  background:#c89312;
  background-image:linear-gradient(180deg, #d4a020 0%, #c89312 100%);
  border:4px solid #0a0a0a;
  border-radius:4px;
  padding:8px 16px;
  box-shadow:inset 0 0 24px rgba(140,90,0,0.4),
              0 4px 12px rgba(0,0,0,0.6),
              0 0 30px rgba(255,179,0,0.10);
  font-family:'LCD',monospace;font-weight:700}
body.skin-qx3 .ev-row.cur{
  background:rgba(255,255,255,0.20);
  box-shadow:inset 8px 0 0 #1a0a00}
body.skin-qx3 .ev-row.follow{box-shadow:inset 8px 0 0 #6a4500}
body.skin-qx3 .ev-row.in-sel{background:rgba(40,120,0,0.25)}
body.skin-qx3 .ev-row.empty{color:#8a6510}
body.skin-qx3 .ev-cell.kind{color:#3a2510}

/* Footer */
body.skin-qx3 #ftr{color:#5a4015}
body.skin-qx3 #ftr-status{color:#9a7820}
body.skin-qx3 #ftr-status[data-s="DIRTY"]{color:#FFB300}
body.skin-qx3 #ftr-status[data-s="PLAYING"], body.skin-qx3 #ftr-status[data-s="LOOP"]{color:#7FFF00}
body.skin-qx3 #ftr-status[data-s="REC"], body.skin-qx3 #ftr-status[data-s="REC ●"]{color:#FF4040}
body.skin-qx3 .kbd{color:#c89312}

/* Overlays */
body.skin-qx3 #goto{background:#c89312;
  background-image:linear-gradient(180deg, #d4a020 0%, #c89312 100%);
  border:4px solid #0a0a0a;color:#1a0a00;
  box-shadow:0 8px 24px rgba(0,0,0,0.8)}
body.skin-qx3 #goto-lbl{color:#1a0a00}
body.skin-qx3 #goto-buf{color:#0a3a00;text-shadow:0 0 4px rgba(127,255,0,0.5)}
body.skin-qx3 #conf{background:#1a1a1a;border:3px solid #c89312;
  box-shadow:0 0 30px rgba(255,179,0,0.2)}
body.skin-qx3 #conf-hdr{color:#FFB300;text-shadow:0 0 8px rgba(255,179,0,0.5)}
body.skin-qx3 .cf-row{border-bottom-color:#2a2a2a}
body.skin-qx3 .cf-row:hover{background:#2a2010}
body.skin-qx3 .cf-row .cf-lbl{color:#9a7820}
body.skin-qx3 .cf-row .cf-val{color:#FFB300}

/* ── Settings overlay ── */
#conf{position:absolute;top:80px;left:120px;right:120px;bottom:80px;
  background:#0F0F0F;border:1px solid #FFD13A;padding:32px;display:none;z-index:60;
  border-radius:6px;box-shadow:0 12px 40px rgba(0,0,0,0.7)}
#conf.on{display:block}
#conf-hdr{font-size:22px;color:#FFD13A;letter-spacing:4px;margin-bottom:24px}
.cf-row{display:flex;justify-content:space-between;align-items:center;
  padding:14px 0;border-bottom:1px solid #2a2a2a;cursor:pointer;
  font-size:17px;letter-spacing:2px}
.cf-row:hover{background:#1A1A1A}
.cf-row .cf-lbl{color:#8C9AA3}
.cf-row .cf-val{color:#FFFFFF}
#conf-hint{position:absolute;bottom:16px;left:32px;font-size:12px;color:#5A6A73;
  letter-spacing:2px}

/* ── Mini-tooltip (cambios de sample/PC/Bank) ── */
#flash{position:absolute;top:200px;left:50%;transform:translateX(-50%);
  background:#0F0F0F;border:1px solid #3AC0E0;padding:10px 24px;
  color:#3AC0E0;font-size:18px;letter-spacing:3px;border-radius:4px;
  opacity:0;transition:opacity .15s;pointer-events:none;z-index:65;
  font-family:'Disket Mono',monospace}
#flash.on{opacity:1}

/* ── Toast notification (SAVED flash) ── */
#toast{position:absolute;top:24px;left:50%;transform:translateX(-50%) translateY(-20px);
  background:#5BD075;color:#0a0a0a;padding:8px 20px;border-radius:4px;
  font-size:14px;letter-spacing:3px;font-weight:700;opacity:0;
  transition:opacity .2s,transform .2s;pointer-events:none;z-index:70;
  box-shadow:0 4px 12px rgba(91,208,117,0.4)}
#toast.on{opacity:1;transform:translateX(-50%) translateY(0)}

/* ── Modal genérico (JJOS-style) ── */
#modal{position:absolute;top:160px;left:50%;transform:translateX(-50%);
  background:#0F0F0F;border:1px solid #FFD13A;padding:24px 32px;
  display:none;color:#FFFFFF;letter-spacing:2px;z-index:55;
  border-radius:6px;box-shadow:0 12px 30px rgba(0,0,0,0.8);
  min-width:380px;font-family:'Disket Mono',monospace}
#modal.on{display:block}
#modal-title{font-size:18px;color:#FFD13A;letter-spacing:4px;margin-bottom:18px}
.mod-row{display:flex;justify-content:space-between;align-items:center;
  padding:8px 0;font-size:16px;border-bottom:1px solid #2a2a2a}
.mod-row.act .mod-lbl{color:#FFD13A}
.mod-row.act .mod-val{outline:2px solid #FFD13A;outline-offset:2px;
  background:rgba(255,209,58,0.10);padding:2px 8px;border-radius:2px}
.mod-lbl{color:#8C9AA3;letter-spacing:2px}
.mod-val{color:#FFFFFF;min-width:56px;text-align:right;padding:2px 8px}
.mod-row.readonly .mod-lbl, .mod-row.readonly .mod-val{color:#5A6A73}
#modal-hint{font-size:10px;color:#5A6A73;margin-top:12px;letter-spacing:2px}

/* ── Goto overlay ── */
#goto{position:absolute;left:50%;top:220px;transform:translateX(-50%);
  background:#0F0F0F;border:1px solid #FFD13A;padding:20px 32px;
  display:none;font-size:24px;color:#FFD13A;letter-spacing:4px;z-index:50;
  border-radius:4px;box-shadow:0 12px 30px rgba(0,0,0,0.7)}
#goto.on{display:block}
#goto-buf{color:#FFFFFF;min-width:64px;display:inline-block}

/* ── Overview ── */
#overview{position:absolute;left:32px;top:214px;right:32px;height:58px;
  background:#0d0d0d;border:1px solid #2a2a2a;border-radius:3px;
  display:flex;flex-direction:column;padding:3px 5px;gap:1px}
.ov-lane{flex:1;display:flex;gap:1px;position:relative}
.ov-lane.act{background:rgba(255,209,58,0.05)}
.ov-cell{flex:1;min-width:0;background:#1f2a33}
.ov-cell.has{background:#FFFFFF}
.ov-cell.bar{border-left:1px solid #3A4A53}
.ov-cell.muted{background:#3A1A1A}
#ov-sel{position:absolute;top:2px;bottom:2px;border:1px solid #3AC0E0;
  background:rgba(58,192,224,0.05);pointer-events:none;display:none}
#ov-loc{position:absolute;top:0;bottom:0;width:1px;background:#FFD13A;
  box-shadow:0 0 6px rgba(255,209,58,0.8);pointer-events:none}
#ov-label{position:absolute;left:32px;top:196px;font-size:11px;color:#3AC0E0;
  letter-spacing:2px;display:none}
#ov-label.on{display:block}

/* ── Macro view (8 columnas verticales) ── */
#macro{position:absolute;left:32px;top:320px;right:32px;bottom:56px;
  display:none;background:#0a0a0a;border:1px solid #2a2a2a;border-radius:4px;
  padding:24px 12px 12px;overflow:hidden}
body.view-macro #evlist{display:none}
body.view-macro #macro{display:block}
body.view-macro #trk-hdr,
body.view-macro #trk-tabs{display:none}
#mc-grid{display:grid;grid-template-columns:repeat(8,1fr);gap:10px;height:100%;
  position:relative}
.mc-col{display:flex;flex-direction:column;gap:3px;background:#161616;
  border-radius:3px;padding:5px;position:relative}
.mc-col.act{background:#2a2210;outline:1px solid #FFD13A;
  box-shadow:0 0 6px rgba(255,209,58,0.25)}
.mc-col.muted{opacity:0.35}
.mc-col-lbl{position:absolute;top:-18px;left:50%;transform:translateX(-50%);
  font-size:13px;color:#8C9AA3;letter-spacing:2px}
.mc-col.act .mc-col-lbl{color:#FFD13A}
.mc-cell{flex:1;min-height:0;background:#1f2a33;border-radius:1px}
.mc-cell.has{background:#FFFFFF;box-shadow:0 0 4px rgba(255,255,255,0.45)}
.mc-cell.bar{border-top:1px solid #3A4A53}
.mc-cell.in-sel{background:#1a3a4a}
.mc-cell.in-sel.has{background:#5BD075;
  box-shadow:0 0 6px rgba(91,208,117,0.65)}
#mc-loc{position:absolute;left:6px;right:6px;height:2px;background:#FFD13A;
  box-shadow:0 0 8px rgba(255,209,58,0.95);pointer-events:none;z-index:3;
  border-radius:1px}
#mc-sel{position:absolute;left:6px;right:6px;
  border:2px solid #3AC0E0;background:rgba(58,192,224,0.08);
  pointer-events:none;display:none;z-index:1;border-radius:3px;
  box-shadow:0 0 8px rgba(58,192,224,0.3)}

/* ── Footer (JOB COMMAND TABLE style) ── */
#ftr{position:absolute;left:32px;right:32px;bottom:12px;height:54px;
  display:grid;grid-template-columns:repeat(8, 1fr);grid-auto-rows:18px;
  gap:0;font-size:9px;color:#8C9AA3;letter-spacing:1px;
  border:1px solid #2a2a2a;border-radius:3px;overflow:hidden;
  background:#0a0a0a;font-family:'Disket Mono',monospace}
.ftr-cell{display:flex;justify-content:space-between;align-items:center;
  padding:0 8px;border-right:1px solid #1a1a1a;border-bottom:1px solid #1a1a1a}
.ftr-cell .lbl{color:#5A6A73;text-transform:uppercase}
.ftr-cell .key{color:#A8B0B8}
#ftr-status-cell{grid-column:span 2;background:#101010;
  display:flex;justify-content:space-between;align-items:center;padding:0 10px}
#ftr-status-cell .lbl{color:#5A6A73}
#ftr-status{color:#A8B0B8;font-size:11px;letter-spacing:2px}
#ftr-status[data-s="DIRTY"]{color:#FFB73A}
#ftr-status[data-s="PLAYING"], #ftr-status[data-s="LOOP"]{color:#5BD075}
#ftr-status[data-s="REC"], #ftr-status[data-s="REC ●"]{color:#FF4466}
.kbd{color:#A8B0B8}

</style></head>
<body><div id="root">

  <div id="hdr">
    <div id="hdr-title">SEQ</div>
    <div id="hdr-locator">001.01.16</div>
    <div id="hdr-right">
      <div id="hdr-res">1/16</div>
      <div id="hdr-loop">&#8634;</div>
      <div id="hdr-rec"></div>
      <div id="hdr-bpm">120</div>
    </div>
  </div>

  <div id="ribbon"></div>

  <div id="ov-label">SELECTION</div>
  <div id="overview">
    <div id="ov-loc"></div>
    <div id="ov-sel"></div>
  </div>

  <div id="trk-hdr">TRK1</div>
  <div id="trk-tabs">
    <span data-t="0" class="act">1</span><span data-t="1">2</span><span data-t="2">3</span>
    <span data-t="3">4</span><span data-t="4">5</span><span data-t="5">6</span>
    <span data-t="6">7</span><span data-t="7">8</span>
  </div>

  <div id="evlist-hdr">
    <div>LOC</div><div></div><div>NOTE</div><div>VEL</div><div>LEN</div>
    <div>PROB</div><div>MICRO</div><div>GATE</div><div>RTCH</div><div>RDIV</div>
    <div class="routing">PROG</div><div class="routing">BANK</div>
  </div>
  <div id="evlist"></div>
  <div id="macro">
    <div id="mc-grid"></div>
    <div id="mc-loc"></div>
    <div id="mc-sel"></div>
  </div>

  <div id="toast">SAVED ✓</div>
  <div id="flash"></div>
  <div id="goto"><span id="goto-lbl">GOTO BAR: </span><span id="goto-buf"></span>_</div>

  <div id="modal">
    <div id="modal-title">—</div>
    <div id="modal-fields"></div>
    <div id="modal-hint">TAB next · ENTER ok · ESC cancel · digits edit</div>
  </div>

  <div id="conf">
    <div id="conf-hdr">SETTINGS</div>
    <div id="conf-rows"></div>
    <div id="conf-hint">CLICK ROW TO CYCLE PORT · ESC TO CLOSE</div>
  </div>

  <div id="ftr">
    <div class="ftr-cell"><span class="lbl">TRK</span><span class="key">1-8</span></div>
    <div class="ftr-cell"><span class="lbl">PLAY</span><span class="key">SPC</span></div>
    <div class="ftr-cell"><span class="lbl">REW</span><span class="key">0</span></div>
    <div class="ftr-cell"><span class="lbl">REC</span><span class="key">R</span></div>
    <div class="ftr-cell"><span class="lbl">LOOP</span><span class="key">&#8984;L</span></div>
    <div class="ftr-cell"><span class="lbl">SAVE</span><span class="key">&#8984;S</span></div>
    <div class="ftr-cell"><span class="lbl">OPEN</span><span class="key">&#8984;O</span></div>
    <div class="ftr-cell"><span class="lbl">SETTINGS</span><span class="key">&#8984;,</span></div>

    <div class="ftr-cell"><span class="lbl">LOCATOR</span><span class="key">&#8593;&#8595;</span></div>
    <div class="ftr-cell"><span class="lbl">SEL STEP</span><span class="key">&#8679;&#8593;&#8595;</span></div>
    <div class="ftr-cell"><span class="lbl">FIELD</span><span class="key">&#8592;&#8594;</span></div>
    <div class="ftr-cell"><span class="lbl">RES</span><span class="key">&#8997;n</span></div>
    <div class="ftr-cell"><span class="lbl">TRIPLET</span><span class="key">T</span></div>
    <div class="ftr-cell"><span class="lbl">DOTTED</span><span class="key">.</span></div>
    <div class="ftr-cell"><span class="lbl">GOTO</span><span class="key">&#8679;nnn</span></div>
    <div class="ftr-cell"><span class="lbl">TRANSP</span><span class="key">&lt; &gt;</span></div>

    <div class="ftr-cell"><span class="lbl">COPY EV</span><span class="key">C</span></div>
    <div class="ftr-cell"><span class="lbl">COPY BARS</span><span class="key">B</span></div>
    <div class="ftr-cell"><span class="lbl">LOOP PLAY</span><span class="key">L</span></div>
    <div class="ftr-cell"><span class="lbl">ERASE</span><span class="key">E</span></div>
    <div class="ftr-cell"><span class="lbl">INS BARS</span><span class="key">I</span></div>
    <div class="ftr-cell"><span class="lbl">QUANTIZE</span><span class="key">Q</span></div>
    <div class="ftr-cell"><span class="lbl">AUDITION</span><span class="key">A</span></div>
    <div class="ftr-cell"><span class="lbl">SAMPLE &#177;</span><span class="key">N/&#8679;N</span></div>
    <div class="ftr-cell"><span class="lbl">TRANSFORM</span><span class="key">X</span></div>
    <div class="ftr-cell"><span class="lbl">MUTE/SOLO</span><span class="key">M/S</span></div>
    <div id="ftr-status-cell"><span class="lbl">STATUS</span><span id="ftr-status">SAVED</span></div>
  </div>
  <span id="ftr-name" style="display:none">UNTITLED</span>

</div>
<script>
function fitRoot(){
  const r=document.getElementById('root');
  const sw=window.innerWidth, sh=window.innerHeight;
  const s=Math.min(sw/1440, sh/920);
  r.style.transform='scale('+s+')';
  r.style.left=((sw-1440*s)/2)+'px';
  r.style.top=((sh-920*s)/2)+'px';
}
fitRoot();window.addEventListener('resize',fitRoot);

function renderRibbon(slots, curField){
  const c=document.getElementById('ribbon');
  c.innerHTML='';
  slots.forEach((s,i)=>{
    const el=document.createElement('div');
    el.className='rb-cell'+(i===curField?' sel':'')+(s.disabled?' disabled':'');
    el.innerHTML='<div class="rb-v">'+s.value+'</div><div class="rb-n">'+s.label+'</div>';
    if(!s.disabled){
      el.style.cursor='pointer';
      el.onclick = () => {
        const cur = (_state && _state.cur_event) >= 0 ? _state.cur_event : null;
        if(cur !== null && cur >= 0) cmd({cmd:'select_event', idx: cur, field: i});
      };
    }
    c.appendChild(el);
  });
}

function renderEvents(evs, curRow, curField, followRow){
  const c=document.getElementById('evlist');
  c.innerHTML='';
  if(!evs.length){
    c.innerHTML='<div class="ev-empty">— NO EVENTS · TOCA UNA NOTA MIDI —</div>';
    return;
  }
  // Field index → columna (LOC=0, KIND=1, NOTE=2, VEL=3, LEN=4, PROB=5, MICRO=6, GATE=7, RTCH=8, RDIV=9)
  const fieldToCol={0:2, 1:3, 2:4, 3:5, 4:6, 5:7, 6:8, 7:9, 8:10, 9:11};
  let curCount = 0;
  evs.forEach((ev,i)=>{
    const sel=(i===curRow);
    const row=document.createElement('div');
    const classes=['ev-row'];
    if(sel) classes.push('sel');
    if(ev.is_cur) { classes.push('cur'); curCount++; }
    if(ev.empty) classes.push('empty');
    if(ev.in_sel) classes.push('in-sel');
    if(i===followRow && curRow<0) classes.push('follow');
    row.className=classes.join(' ');
    // colToField inverso de fieldToCol (cols 2-9 → fields 0-7)
    const colToField={2:0, 3:1, 4:2, 5:3, 6:4, 7:5, 8:6, 9:7, 10:8, 11:9};
    const routingCols={10:'program', 11:'bank'};
    (ev.cells||[]).forEach((txt,ci)=>{
      const cell=document.createElement('div');
      let cls = 'ev-cell'+(ci===1?' kind':'');
      if(routingCols[ci]) cls += ' routing';
      cell.className = cls;
      if(sel && !ev.empty && fieldToCol[curField]===ci) cell.classList.add('sel-col');
      cell.textContent=txt;
      if(ev.ev_idx >= 0){
        cell.style.cursor='pointer';
        cell.onclick = () => {
          // Click selecciona evento + field. Para routing cols + ya están mapeadas.
          const payload = {cmd:'select_event', idx: ev.ev_idx};
          if(colToField[ci] !== undefined) payload.field = colToField[ci];
          cmd(payload);
        };
      }
      row.appendChild(cell);
    });
    c.appendChild(row);
  });
  // Sin scroll: la ventana es fija y el locator se centra desde backend
  if(curCount === 0) console.warn('[DEBUG] no row has is_cur=true. locator_tick=', _state && _state.locator_tick, 'res=', _state && _state.resolution);
}

let _confOpen = false;
function renderConf(settings){
  const rows = document.getElementById('conf-rows');
  rows.innerHTML='';
  (settings||[]).forEach(s=>{
    const el = document.createElement('div');
    el.className='cf-row';
    el.innerHTML='<span class="cf-lbl">'+s.label+'</span><span class="cf-val">'+s.value+'</span>';
    el.onclick = ()=> cmd({cmd:'cycle_port', slot:s.slot});
    rows.appendChild(el);
  });
}
function toggleConf(){
  _confOpen = !_confOpen;
  document.getElementById('conf').classList.toggle('on', _confOpen);
}
function renderMacro(m){
  const grid = document.getElementById('mc-grid');
  if(!m) return;
  const total = m.total_steps;
  const spb = m.steps_per_bar;
  // Rebuild columns if steps changed
  const existingCols = grid.querySelectorAll('.mc-col');
  const need = existingCols.length !== 8 ||
    (existingCols[0] && existingCols[0].querySelectorAll('.mc-cell').length !== total);
  if(need){
    grid.innerHTML='';
    for(let t=0; t<8; t++){
      const col = document.createElement('div');
      col.className='mc-col';
      const lbl = document.createElement('div');
      lbl.className='mc-col-lbl';
      lbl.textContent='T'+(t+1);
      col.appendChild(lbl);
      for(let i=0; i<total; i++){
        const c = document.createElement('div');
        c.className = 'mc-cell'+(i%spb===0?' bar':'');
        col.appendChild(c);
      }
      grid.appendChild(col);
    }
  }
  const cols = grid.querySelectorAll('.mc-col');
  for(let t=0; t<8; t++){
    const tr = m.tracks[t] || {steps:[], has:false, mute:false};
    cols[t].classList.toggle('act', t===m.cur_track);
    cols[t].classList.toggle('muted', !!tr.mute);
    const cells = cols[t].querySelectorAll('.mc-cell');
    const hasSet = new Set(tr.steps);
    for(let i=0; i<cells.length; i++){
      const c = cells[i];
      const has = hasSet.has(i);
      const inSel = m.sel_range && i >= m.sel_range[0] && i < m.sel_range[1];
      c.className = 'mc-cell'+(i%spb===0?' bar':'')+(has?' has':'')+(inSel?' in-sel':'');
    }
  }
  // Locator y selección: líneas horizontales cruzando todas las columnas
  // Padding columna interno: 6px arriba/abajo + 4px gap entre celdas
  const gh = grid.clientHeight - 12; // padding inner cell
  const stepH = gh / total;
  const baseTop = 32 + 6; // padding-top del #macro + padding columna
  const loc = document.getElementById('mc-loc');
  loc.style.top = (baseTop + m.loc_step * stepH) + 'px';
  const sel = document.getElementById('mc-sel');
  if(m.sel_range){
    sel.style.display='block';
    sel.style.top    = (baseTop + m.sel_range[0] * stepH) + 'px';
    sel.style.height = ((m.sel_range[1] - m.sel_range[0]) * stepH) + 'px';
  } else {
    sel.style.display='none';
  }
}

function renderOverview(ov){
  const cont = document.getElementById('overview');
  if(!ov){ return; }
  const total = ov.total_steps;
  const spb = ov.steps_per_bar;
  // Rebuild lanes si número de steps cambió
  const need = (cont.querySelectorAll('.ov-lane').length !== 8) ||
    (cont.querySelector('.ov-lane') && cont.querySelector('.ov-lane').children.length !== total);
  if(need){
    // Limpiar manteniendo loc y sel
    Array.from(cont.querySelectorAll('.ov-lane')).forEach(n=>n.remove());
    for(let t=0; t<8; t++){
      const lane = document.createElement('div');
      lane.className = 'ov-lane'+(t===ov.cur_track?' act':'');
      for(let i=0; i<total; i++){
        const c = document.createElement('div');
        c.className = 'ov-cell' + (i%spb===0?' bar':'');
        lane.appendChild(c);
      }
      cont.appendChild(lane);
    }
  }
  // Actualizar cells
  const lanes = cont.querySelectorAll('.ov-lane');
  for(let t=0; t<8; t++){
    lanes[t].classList.toggle('act', t===ov.cur_track);
    const cells = lanes[t].children;
    const hasSet = new Set(ov.tracks[t] || []);
    for(let i=0; i<cells.length; i++){
      const c = cells[i];
      const has = hasSet.has(i);
      c.className = 'ov-cell'+(i%spb===0?' bar':'')+(has?' has':'');
    }
  }
  // Locator vertical line
  const cw = cont.clientWidth - 12;  // padding 6px x 2
  const stepW = cw / total;
  const loc = document.getElementById('ov-loc');
  loc.style.left = (6 + ov.loc_step * stepW) + 'px';
  // Selección
  const sel = document.getElementById('ov-sel');
  const lbl = document.getElementById('ov-label');
  if(ov.sel_range){
    sel.style.display='block';
    sel.style.left  = (6 + ov.sel_range[0] * stepW) + 'px';
    sel.style.width = ((ov.sel_range[1] - ov.sel_range[0]) * stepW) + 'px';
    lbl.classList.add('on');
    lbl.textContent = 'SELECTION · '+ov.sel_range[0]+' → '+ov.sel_range[1]+' STEPS';
  } else {
    sel.style.display='none';
    lbl.classList.remove('on');
  }
}

function renderTabs(curTrack, info){
  document.querySelectorAll('#trk-tabs span').forEach((el,i)=>{
    const ti = info && info[i] ? info[i] : {has_events:false, mute:false, solo:false};
    el.classList.toggle('act', i===curTrack);
    el.classList.toggle('has', !!ti.has_events);
    el.classList.toggle('muted', !!ti.mute);
    el.classList.toggle('solo', !!ti.solo);
  });
}

let _state = null;
let _animTime0 = 0;
let _animTick0 = 0;

function fmtLocatorTick(tick, res, timeSigNum, tpq){
  const isT = (typeof res === 'string') && res.endsWith('T');
  const base = isT ? parseInt(res.slice(0,-1),10) : parseInt(res,10);
  const tps = isT ? Math.floor((tpq*4/base)*2/3) : (tpq*4)/base;
  const stepsPerBar = Math.floor((tpq*timeSigNum)/tps);
  const totalSteps = Math.floor(tick / tps);
  const bar = Math.floor(totalSteps / stepsPerBar);
  const step = totalSteps % stepsPerBar;
  const pad = (n,w)=>String(n).padStart(w,'0');
  return pad(bar+1,3)+'.'+pad(step+1,2)+'.'+String(res);
}

let _lastSavedTs = 0;
let _toastTimer = null;
function showToast(msg){
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('on');
  if(_toastTimer) clearTimeout(_toastTimer);
  _toastTimer = setTimeout(()=> t.classList.remove('on'), 1500);
}

function applyState(s){
  if(s.last_saved_ts && s.last_saved_ts !== _lastSavedTs){
    if(_lastSavedTs !== 0) showToast('SAVED ✓');
    _lastSavedTs = s.last_saved_ts;
  }
  // Captura de nota desde el LK para modales
  if(_modal && s.modal_captured_note != null && s.modal_captured_ts !== _lastModalCapturedTs){
    _lastModalCapturedTs = s.modal_captured_ts;
    const fcur = _modal.fields[_modal.cur];
    if(fcur && fcur.note){
      fcur.value = s.modal_captured_note;
      renderModal();
    }
  }
  _state = s;
  // Skin + view classes
  let classes = '';
  if(s.skin) classes += 'skin-' + s.skin;
  if(s.view === 'macro') classes += ' view-macro';
  document.body.className = classes.trim();
  if(s.running){
    _animTime0 = performance.now()/1000;
    _animTick0 = s.locator_tick;
  }
  document.getElementById('hdr-bpm').textContent=String(Math.round(s.bpm));
  document.getElementById('hdr-res').textContent='1/'+String(s.resolution);
  const loc=document.getElementById('hdr-locator');
  loc.textContent=s.locator;
  loc.classList.toggle('playing', !!s.running);
  document.getElementById('hdr-rec').classList.toggle('on', !!s.recording);
  document.getElementById('hdr-loop').classList.toggle('on', !!s.loop_on);
  let status='SAVED';
  if(s.dirty) status='DIRTY';
  if(s.recording) status='REC';
  if(s.running) status = (s.loop_on && s.sel) ? 'LOOP' : 'PLAYING';
  if(s.running && s.recording) status='REC ●';
  const fs=document.getElementById('ftr-status');
  fs.textContent=status;
  fs.dataset.s=status;
  let trkLabel = s.track_name + ' · OUT' + (s.track_port+1);
  if(s.track_type === 'sampler' && s.track_samples && s.track_samples.length > 0){
    const cur = s.track_current_program || 0;
    const samp = s.track_samples[cur] || '—';
    trkLabel += ' · ' + samp.replace(/\.(wav|aif|aiff|flac)$/i,'');
  }
  document.getElementById('trk-hdr').textContent = trkLabel;
  renderTabs(s.cur_track, s.tracks_info);
  // Mini-flash (sample/PC/Bank)
  const f = document.getElementById('flash');
  if(s.flash){ f.textContent = s.flash; f.classList.add('on'); }
  else { f.classList.remove('on'); }
  renderOverview(s.overview);
  if(s.view === 'macro') renderMacro(s.macro);
  if(_confOpen) renderConf(s.settings);
  renderRibbon(s.ribbon||[], s.cur_field);
  renderEvents(s.events||[], s.cur_row, s.cur_field, s.follow_row);
}

// Animación predictiva del locator entre SSE pokes
function _animFrame(){
  if(_state && _state.running){
    const dt = performance.now()/1000 - _animTime0;
    const tps = _state.bpm * _state.tpq / 60;
    const tick = _animTick0 + dt * tps;
    document.getElementById('hdr-locator').textContent =
      fmtLocatorTick(tick, _state.resolution, _state.time_sig?_state.time_sig[0]:4, _state.tpq||192);
  }
  requestAnimationFrame(_animFrame);
}
requestAnimationFrame(_animFrame);

function cmd(payload){
  fetch('/api/cmd', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(payload)});
}

function openFile(){
  const input=document.createElement('input');
  input.type='file'; input.accept='.json,application/json';
  input.onchange=async ()=>{
    const f=input.files[0]; if(!f) return;
    try{
      const txt=await f.text();
      const data=JSON.parse(txt);
      await fetch('/api/song', {method:'POST',
        headers:{'Content-Type':'application/json'}, body: JSON.stringify(data)});
    }catch(err){ console.error('Load error:', err); }
  };
  input.click();
}

function saveAs(){
  const a=document.createElement('a');
  a.href='/api/song';
  a.download='song.json';
  document.body.appendChild(a); a.click(); a.remove();
}

// ── Modal genérico (JJOS-style) ──
let _modal = null;
function curBar(){
  if(!_state || !_state.locator) return 1;
  return parseInt(_state.locator.split('.')[0], 10);
}
function curTrack(){ return _state ? (_state.cur_track+1) : 1; }

function openModal(name){
  const hasSel = !!(_state && _state.sel);
  const _rangeField = {key:'use_sel', label:'RANGE',
                        value: hasSel ? 1 : 0, bool:true,
                        labels:['BARS','SELECTION']};
  if(name === 'copy'){
    const b = curBar();
    _modal = {cmd:'copy_range', title:'COPY EVENTS', cur:0, fields:[
      _rangeField,
      {key:'from_bar', label:'FROM BAR', value: b,   min:1, hideWhenSel:true},
      {key:'to_bar',   label:'TO BAR',   value: b+1, min:1, hideWhenSel:true},
      {key:'from_track', label:'FROM TRK', value: curTrack(), min:1, max:8},
      {key:'to_track',   label:'TO TRK',   value: curTrack(), min:1, max:8},
      {key:'times',      label:'TIMES',    value: 1,  min:1},
      {key:'mode',       label:'MODE',     value: 0, bool:true, labels:['MERGE','REPLACE']},
    ]};
  } else if(name === 'loop'){
    const b = curBar();
    _modal = {cmd:'loop_range', title:'COPY BARS (REPEAT)', cur:0, fields:[
      _rangeField,
      {key:'from_bar', label:'FROM BAR', value: b,   min:1, hideWhenSel:true},
      {key:'to_bar',   label:'TO BAR',   value: b+1, min:1, hideWhenSel:true},
      {key:'times',    label:'TIMES',    value: 2,   min:1},
      {key:'mode',     label:'MODE',     value: 0, bool:true, labels:['MERGE','REPLACE']},
    ]};
  } else if(name === 'loop_play'){
    const b = curBar();
    _modal = {cmd:'set_loop_playback', title:'LOOP PLAYBACK', cur:0, fields:[
      {key:'from_bar', label:'FROM BAR', value: b,   min:1},
      {key:'to_bar',   label:'TO BAR',   value: b+2, min:1},
    ]};
  } else if(name === 'track_type'){
    const isSampler = _state && _state.track_type === 'sampler';
    const dir = _state && _state.track_sample_dir || '';
    const samples = _state && _state.track_samples || [];
    _modal = {cmd:'set_track_type', title:'TRACK '+(curTrack())+' TYPE', cur:0, fields:[
      {key:'type', label:'TYPE', value: isSampler?1:0, bool:true, labels:['MIDI','SAMPLER']},
      {key:'sample_dir', label:'SAMPLE DIR', value: dir, string:true, picker:'dir'},
      {key:'_info', label:'SAMPLES', value: samples.length, readonly:true},
    ]};
  } else if(name === 'transform'){
    const b = curBar();
    // Campos: showFor especifica en qué ops aparecen (vacío = siempre)
    _modal = {cmd:'transform', title:'TRANSFORM', cur:0, fields:[
      {key:'op', label:'OPERATION', value:0, cycle:true,
       labels:['reverse','stretch','shift','vel_ramp','humanize',
               'scale_quantize','thin','double_oct','set_length',
               'legato','gate_scale','chop',
               'reverse_sample','sample_start','attack','release',
               'move']},
      _rangeField,
      {key:'from_bar', label:'FROM BAR', value: b,   min:1, hideWhenSel:true},
      {key:'to_bar',   label:'TO BAR',   value: b+1, min:1, hideWhenSel:true},
      {key:'track',    label:'TRK (0=ALL)', value: 0, min:0, max:8,
       hideFor:['move']},
      {key:'factor',   label:'FACTOR %', value: 100, min:1, max:400,
       showFor:['stretch','gate_scale']},
      {key:'shift_ticks',label:'SHIFT TICKS', value:0, min:-9999, max:9999,
       showFor:['shift']},
      {key:'from_vel', label:'VEL FROM', value: 30,  min:1, max:127,
       showFor:['vel_ramp']},
      {key:'to_vel',   label:'VEL TO',   value: 120, min:1, max:127,
       showFor:['vel_ramp']},
      {key:'timing_ticks',label:'TIMING ±', value: 6, min:0, max:100,
       showFor:['humanize']},
      {key:'vel_amount', label:'VEL ±', value: 10, min:0, max:64,
       showFor:['humanize']},
      {key:'root', label:'SCALE ROOT (0=C)', value: 0, min:0, max:11,
       showFor:['scale_quantize']},
      {key:'scale', label:'SCALE', value:0, cycle:true,
       labels:['major','minor','dorian','pent_maj','pent_min','blues','chromatic'],
       showFor:['scale_quantize']},
      {key:'percent', label:'THIN %', value: 30, min:1, max:99,
       showFor:['thin']},
      {key:'direction', label:'OCTAVE DIR', value:0, cycle:true,
       labels:['up','down'], showFor:['double_oct']},
      {key:'length_steps',label:'LENGTH STEPS', value:1, min:1, max:16,
       showFor:['set_length']},
      {key:'chop_gate',  label:'CHOP GATE %',  value: 30, min:5, max:99,
       showFor:['chop']},
      {key:'chop_steps', label:'CHOP STEPS',   value: 2,  min:1, max:16,
       showFor:['chop']},
      {key:'reverse_on', label:'REVERSE',      value: 1, bool:true,
       labels:['OFF','ON'], showFor:['reverse_sample']},
      {key:'start_pct',  label:'START %',      value: 0,  min:0, max:95,
       showFor:['sample_start']},
      {key:'attack_ms',  label:'ATTACK ms',    value: 0,  min:0, max:2000,
       showFor:['attack']},
      {key:'release_ms', label:'RELEASE ms',   value: 0,  min:0, max:2000,
       showFor:['release']},
      {key:'src_track', label:'SRC TRK', value: curTrack(), min:1, max:8,
       showFor:['move']},
      {key:'dst_track', label:'DST TRK', value: curTrack(), min:1, max:8,
       showFor:['move']},
    ]};
  } else if(name === 'erase'){
    const b = curBar();
    _modal = {cmd:'erase_range', title:'ERASE EVENTS', cur:0, fields:[
      _rangeField,
      {key:'from_bar', label:'FROM BAR', value: b,   min:1, hideWhenSel:true},
      {key:'to_bar',   label:'TO BAR',   value: b+1, min:1, hideWhenSel:true},
      {key:'track',    label:'TRK (0=ALL)', value: 0, min:0, max:8},
      {key:'event_type', label:'TYPE', value: 0, cycle:true,
       labels:['all','note','cc','prog','bank']},
      {key:'note_lo', label:'NOTE FROM', value: 0,   min:0, max:127, note:true},
      {key:'note_hi', label:'NOTE TO',   value: 127, min:0, max:127, note:true},
      {key:'cc_num',  label:'CC# (-1=ALL)', value: -1, min:-1, max:127},
    ]};
  } else return;
  renderModal();
}
function _visibleFields(){
  if(!_modal) return [];
  const opField = _modal.fields.find(f => f.key === 'op');
  const opName = opField ? opField.labels[opField.value] : null;
  const useSelF = _modal.fields.find(f => f.key === 'use_sel');
  const useSel = useSelF ? !!useSelF.value : false;
  return _modal.fields.filter(f => {
    if(f.hideWhenSel && useSel) return false;
    if(f.showFor && opName) return f.showFor.includes(opName);
    if(f.hideFor && opName) return !f.hideFor.includes(opName);
    return true;
  });
}

let _lastModalCapturedTs = 0;
function renderModal(){
  const m = document.getElementById('modal');
  if(!_modal){
    m.classList.remove('on');
    if(_state && _state.modal_capture_armed) cmd({cmd:'arm_note_capture', armed:false});
    return;
  }
  m.classList.add('on');
  document.getElementById('modal-title').textContent = _modal.title;
  const c = document.getElementById('modal-fields');
  c.innerHTML='';
  const visible = _visibleFields();
  // Asegurar cur dentro de visible
  const curField = _modal.fields[_modal.cur];
  if(!visible.includes(curField)){
    _modal.cur = _modal.fields.indexOf(visible[0]);
  }
  visible.forEach((f,vi)=>{
    const i = _modal.fields.indexOf(f);
    const row = document.createElement('div');
    row.className = 'mod-row' + (i===_modal.cur?' act':'');
    let v;
    if(f.cycle) v = (f.labels[f.value]||'?').toUpperCase();
    else if(f.bool) v = f.labels ? f.labels[f.value?1:0] : (f.value?'ON':'OFF');
    else if(f.string) v = (f.value && f.value.length > 32 ? '…'+f.value.slice(-30) : (f.value||'(empty)'));
    else if(f.note){
      const n = f.value|0;
      const NAMES=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
      v = NAMES[n%12] + (Math.floor(n/12)-2) + ' ('+n+')';
      if(_modal && _modal.fields[_modal.cur] === f && _state && _state.modal_capture_armed){
        v = '◉ TOUCH KEY';
      }
    }
    else v = String(f.value);
    if(f.readonly) row.classList.add('readonly');
    row.innerHTML = '<span class="mod-lbl">'+f.label+'</span><span class="mod-val">'+v+'</span>';
    c.appendChild(row);
  });
}
function modalHandle(e){
  if(!_modal) return false;
  if(e.key==='Escape'){ _modal=null; renderModal(); e.preventDefault(); return true; }
  if(e.key==='Tab'){
    const visible = _visibleFields();
    if(visible.length === 0) return true;
    const curField = _modal.fields[_modal.cur];
    let vi = visible.indexOf(curField);
    if(vi < 0) vi = 0;
    const d = e.shiftKey ? -1 : 1;
    let attempts = visible.length;
    while(attempts-- > 0){
      vi = (vi + d + visible.length) % visible.length;
      if(!visible[vi].readonly) break;
    }
    _modal.cur = _modal.fields.indexOf(visible[vi]);
    // Si el nuevo field es note, armar captura
    const nf = visible[vi];
    if(nf && nf.note) cmd({cmd:'arm_note_capture', armed:true, clear:true});
    else if(_state && _state.modal_capture_armed) cmd({cmd:'arm_note_capture', armed:false});
    renderModal(); e.preventDefault(); return true;
  }
  if(e.key==='Enter'){
    // Si el campo activo es string + picker, abrir Finder nativo
    const fcur = _modal.fields[_modal.cur];
    if(fcur && fcur.string){
      if(fcur.picker === 'dir'){
        fetch('/api/pick_dir', {method:'POST'})
          .then(r => r.json())
          .then(d => {
            if(d.path){ fcur.value = d.path; renderModal(); }
          });
      } else {
        const v = prompt(fcur.label+':', fcur.value||'');
        if(v !== null){ fcur.value = v; renderModal(); }
      }
      e.preventDefault(); return true;
    }
    const p = {cmd: _modal.cmd};
    const visibleSubmit = _visibleFields();
    visibleSubmit.forEach(f=>{
      if(f.readonly || f.key.startsWith('_')) return;
      if(f.cycle){
        p[f.key] = f.labels[f.value];
      } else if(f.bool){
        p[f.key] = f.labels ? f.labels[f.value?1:0].toLowerCase() : !!f.value;
      } else if(f.string){
        p[f.key] = f.value || '';
      } else {
        p[f.key] = parseInt(f.value,10)||0;
      }
    });
    const opF = _modal.fields.find(f => f.key === 'op');
    if(opF && !p.op) p.op = opF.labels[opF.value];
    // RANGE: si use_sel = 'selection' y hay sel, mandar tick_start/end
    if((p.use_sel === 'selection' || p.use_sel === true) && _state && _state.sel){
      p.tick_start = _state.sel.tick_start;
      p.tick_end = _state.sel.tick_end;
      delete p.from_bar;
      delete p.to_bar;
    }
    delete p.use_sel;
    if(p.all !== undefined) p.all = !!p.all;
    if(p.track === 0) p.track = null;
    cmd(p);
    _modal=null; renderModal();
    e.preventDefault(); return true;
  }
  const f = _modal.fields[_modal.cur];
  if(f.readonly || f.string){
    return true;
  }
  if(f.cycle){
    if(e.key==='ArrowUp' || e.key==='ArrowDown'){
      const n = f.labels.length;
      f.value = (f.value + (e.key==='ArrowUp'?-1:1) + n) % n;
      renderModal(); e.preventDefault(); return true;
    }
    return true;
  }
  if(f.bool){
    if(e.key==='ArrowUp' || e.key==='ArrowDown' || e.code && e.code.startsWith('Digit')){
      f.value = !f.value;
      renderModal(); e.preventDefault(); return true;
    }
  } else {
    const mn = f.min!==undefined ? f.min : 0;
    const mx = f.max!==undefined ? f.max : 9999;
    if(e.code && e.code.startsWith('Digit')){
      const d = e.code.slice(5);
      let v = parseInt(String(f.value)+d, 10) || 0;
      f.value = Math.max(mn, Math.min(mx, v));
      renderModal(); e.preventDefault(); return true;
    }
    if(e.key==='Backspace'){
      const s = String(f.value);
      let v = s.length>1 ? parseInt(s.slice(0,-1),10)||0 : 0;
      f.value = Math.max(mn, Math.min(mx, v));
      renderModal(); e.preventDefault(); return true;
    }
    if(e.key==='ArrowUp'){ f.value = Math.min(mx, parseInt(f.value,10)+1); renderModal(); e.preventDefault(); return true; }
    if(e.key==='ArrowDown'){ f.value = Math.max(mn, parseInt(f.value,10)-1); renderModal(); e.preventDefault(); return true; }
  }
  return true;  // bloquea cualquier otra tecla
}

// ── Goto bar buffer (Shift+digits+Enter) ──
let _gotoBuf = '';
let _gotoActive = false;
function showGoto(){
  const g=document.getElementById('goto');
  document.getElementById('goto-lbl').textContent='GOTO BAR: ';
  document.getElementById('goto-buf').textContent=_gotoBuf;
  g.classList.toggle('on', _gotoActive);
}
// ── Modos numéricos: I (insert bars), D (delete bars), T (tempo) ──
let _barBuf = '';
let _barMode = null;  // 'I' | 'D' | 'T'
function showBar(){
  const labels = {I:'INSERT BARS: ', D:'DELETE BARS: ', T:'TEMPO BPM: ',
                  P:'PROGRAM (0-127): ',
                  O:'TRACK PORT (1 o 2): '};
  const g=document.getElementById('goto');
  document.getElementById('goto-lbl').textContent = labels[_barMode] || '';
  document.getElementById('goto-buf').textContent = _barBuf;
  g.classList.toggle('on', !!_barMode);
}

document.addEventListener('keydown', e=>{

  // Modal genérico tiene prioridad absoluta
  if(modalHandle(e)) return;

  // ── Modo numérico activo (I/D/T): solo acepta dígitos, Enter, Esc; cualquier otra cancela
  if(_barMode){
    if(e.code && e.code.startsWith('Digit')){
      _barBuf += e.code.slice(5);
      showBar(); e.preventDefault(); return;
    }
    if(e.key==='Enter'){
      const raw = _barBuf.trim();
      const n = raw === '' ? null : parseInt(raw,10);
      if(_barMode==='I') cmd({cmd:'insert_bars', n: n||1});
      else if(_barMode==='D') cmd({cmd:'delete_bars', n: n||1});
      else if(_barMode==='T') cmd({cmd:'set_bpm', bpm: n||120});
      else if(_barMode==='P') cmd({cmd:'set_routing', idx:-1, key:'program', value:n});
      else if(_barMode==='B') cmd({cmd:'set_routing', idx:-1, key:'bank', value:n});
      else if(_barMode==='O') cmd({cmd:'set_track_port',
                                    port: n!==null ? Math.max(0,Math.min(1,n-1)) : 0});
      _barBuf=''; _barMode=null; showBar();
      e.preventDefault(); return;
    }
    // Cualquier otra tecla: cancelar modo (no consume el evento)
    _barBuf=''; _barMode=null; showBar();
  }

  // ── Goto bar: Shift + dígito empieza modo, dígitos acumulan, Enter confirma
  if(e.shiftKey && e.code && e.code.startsWith('Digit') && !_barMode){
    _gotoActive=true;
    _gotoBuf += e.code.slice(5);
    showGoto();
    e.preventDefault();
    return;
  }
  if(_gotoActive){
    if(e.code && e.code.startsWith('Digit')){
      _gotoBuf += e.code.slice(5);
      showGoto();
      e.preventDefault();
      return;
    }
    if(e.key==='Enter'){
      cmd({cmd:'goto_bar', bar: parseInt(_gotoBuf,10)||1});
      _gotoBuf=''; _gotoActive=false; showGoto();
      e.preventDefault();
      return;
    }
    // Cualquier otra tecla: cancelar modo
    _gotoBuf=''; _gotoActive=false; showGoto();
  }

  if(e.metaKey){
    const all = e.shiftKey;
    if(e.key==='c' || e.key==='C'){ cmd({cmd:'copy', all:all}); e.preventDefault(); }
    else if(e.key==='v' || e.key==='V'){ cmd({cmd:'paste', all:all}); e.preventDefault(); }
    else if(e.key==='d' || e.key==='D'){ cmd({cmd:'duplicate', all:all}); e.preventDefault(); }
    else if(e.key==='s' || e.key==='S'){
      if(e.shiftKey) saveAs(); else cmd({cmd:'save'});
      e.preventDefault();
    }
    else if(e.key==='o' || e.key==='O'){ openFile(); e.preventDefault(); }
    else if(e.key==='l' || e.key==='L'){ cmd({cmd:'toggle_loop'}); e.preventDefault(); }
    else if(e.altKey && (e.key==='n' || e.key==='N' || e.code==='KeyN')){
      if(confirm('¿Nuevo song? Se perderán cambios no guardados.')) cmd({cmd:'new'});
      e.preventDefault();
    }
    else if(e.key===','){ toggleConf(); renderConf(_state&&_state.settings); e.preventDefault(); }
    else if(e.key==='z' || e.key==='Z'){
      cmd({cmd: e.shiftKey?'redo':'undo'}); e.preventDefault();
    }
    else if(e.key==='t' || e.key==='T'){
      if(e.shiftKey){
        _barMode='T'; _barBuf=''; showBar();
      } else {
        openModal('track_type');
      }
      e.preventDefault();
    }
    return;
  }

  if(e.key>='1' && e.key<='8'){
    cmd({cmd:'select_track', t: parseInt(e.key,10)-1});
    e.preventDefault();
  } else if(e.key==='ArrowUp')    {
    if(e.shiftKey) cmd({cmd:'extend_sel', dir:'up', gran: e.altKey?'bar':'step'});
    else if(e.altKey) cmd({cmd:'nav_event_in_step', dir:'prev'});
    else cmd({cmd:'locator', dir:'left'});
    e.preventDefault();
  }
  else if(e.key==='ArrowDown')    {
    if(e.shiftKey) cmd({cmd:'extend_sel', dir:'down', gran: e.altKey?'bar':'step'});
    else if(e.altKey) cmd({cmd:'nav_event_in_step', dir:'next'});
    else cmd({cmd:'locator', dir:'right'});
    e.preventDefault();
  }
  else if(e.key==='PageUp'){ cmd({cmd:'nav_event_in_step', dir:'prev'}); e.preventDefault(); }
  else if(e.key==='PageDown'){ cmd({cmd:'nav_event_in_step', dir:'next'}); e.preventDefault(); }
  else if(e.key==='Escape'){
    if(_confOpen){ toggleConf(); }
    else cmd({cmd:'clear_sel'});
    e.preventDefault();
  }
  else if(e.key==='ArrowLeft')    { cmd({cmd:'nav_field', dir:'prev'}); e.preventDefault(); }
  else if(e.key==='ArrowRight')   { cmd({cmd:'nav_field', dir:'next'}); e.preventDefault(); }
  else if(e.key==='Tab'){
    cmd({cmd:'nav_field', dir: e.shiftKey?'prev':'next'});
    e.preventDefault();
  }
  else if(e.key==='['){ cmd({cmd:'res', dir:'down'}); e.preventDefault(); }
  else if(e.key===']'){ cmd({cmd:'res', dir:'up'}); e.preventDefault(); }
  else if(e.key==='Delete' || e.key==='Backspace'){
    if(e.altKey) cmd({cmd:'delete_event'});
    else cmd({cmd:'delete', all: e.shiftKey});
    e.preventDefault();
  }
  else if(e.code==='KeyP'){ _barMode='P'; _barBuf=''; showBar(); console.log('[BAR] P'); e.preventDefault(); }
  else if(e.code==='KeyO'){ _barMode='O'; _barBuf=''; showBar(); console.log('[BAR] O'); e.preventDefault(); }
  else if(e.code==='KeyK'){ cmd({cmd:'insert_cc'}); e.preventDefault(); }
  else if(e.key==='i' || e.key==='I'){ _barMode='I'; _barBuf=''; showBar(); e.preventDefault(); }
  else if(e.code==='KeyT'){ cmd({cmd:'toggle_triplet'}); e.preventDefault(); }
  else if(e.code==='Period'){ cmd({cmd:'toggle_dotted'}); e.preventDefault(); }
  else if(e.code==='KeyA'){ cmd({cmd:'audition'}); e.preventDefault(); }
  else if(e.code==='KeyN'){
    cmd({cmd:'next_sample', delta: e.shiftKey?-1:1}); e.preventDefault();
  }
  else if(e.code==='KeyX'){ openModal('transform'); e.preventDefault(); }
  else if(e.code==='KeyC'){ openModal('copy'); e.preventDefault(); }
  else if(e.code==='KeyB'){ openModal('loop'); e.preventDefault(); }
  else if(e.code==='KeyL'){ openModal('loop_play'); e.preventDefault(); }
  else if(e.code==='KeyE'){ openModal('erase'); e.preventDefault(); }
  else if(e.key==='q'){ cmd({cmd:'quantize', all:false}); e.preventDefault(); }
  else if(e.key==='Q'){ cmd({cmd:'quantize', all:true}); e.preventDefault(); }
  else if(e.code==='KeyM'){ cmd({cmd:'toggle_mute'}); e.preventDefault(); }
  else if(e.code==='KeyS'){ cmd({cmd:'toggle_solo'}); e.preventDefault(); }
  else if(e.code==='KeyV'){ cmd({cmd:'toggle_macro'}); e.preventDefault(); }
  // Transpose: solo con < > (shift+, shift+.) para no chocar con T/. de resolución
  else if(e.key==='<'){
    cmd({cmd:'transpose', semi: -1, all: false}); e.preventDefault();
  }
  else if(e.key==='>'){
    cmd({cmd:'transpose', semi: 1, all: false}); e.preventDefault();
  }
  else if(e.key==='+' || e.key==='='){
    cmd({cmd:'edit', delta: e.shiftKey?10:1}); e.preventDefault();
  }
  else if(e.key==='-' || e.key==='_'){
    cmd({cmd:'edit', delta: e.shiftKey?-10:-1}); e.preventDefault();
  }
  else if(e.key===' '){ cmd({cmd:'play'}); e.preventDefault(); }
  else if(e.key==='0'){ cmd({cmd:'rewind'}); e.preventDefault(); }
  else if(e.key==='r' || e.key==='R'){ cmd({cmd:'toggle_record'}); e.preventDefault(); }
  // F1-F8 → set resolución directa (1, 2, 4, 8, 16, 32, 64, 128)
  else if(e.key && e.key.startsWith('F') && e.key.length<=3){
    const n=parseInt(e.key.slice(1),10);
    if(n>=1 && n<=8){
      const res=[1,2,4,8,16,32,64,128][n-1];
      cmd({cmd:'set_res', res:res}); e.preventDefault();
    }
  }
  // Alt+digit → resolución rápida. Alt+Shift+digit → triplete.
  // 1=1/1 2=1/2 3=1/4 4=1/8 5=1/16 6=1/32 7=1/64 8=1/128
  else if(e.altKey && e.code && e.code.startsWith('Digit')){
    const n = parseInt(e.code.slice(5), 10);
    if(n >= 1 && n <= 8){
      const recto = [1,2,4,8,16,32,64,128][n-1];
      // Triplete solo desde 1/4 hacia abajo (4T, 8T, 16T, 32T, 64T)
      let res = recto;
      if(e.shiftKey && recto >= 4 && recto <= 64) res = recto + 'T';
      cmd({cmd:'set_res', res: res}); e.preventDefault();
    }
  }
});

const es=new EventSource('/events');
es.onmessage=e=>{ try{ applyState(JSON.parse(e.data)); }catch(_){} };
</script>
</body></html>
'''


# ── Server bootstrap ────────────────────────────────────────────────────────
def _start_server(port):
    import logging
    from werkzeug.serving import make_server
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    srv = make_server('0.0.0.0', port, app, threaded=True)
    srv.serve_forever()


def main():
    global _seq
    _seq = Sequencer()
    _seq.load()
    _seq.open_midi()
    port = config.FLASK_PORT
    print(f"[rrrlinear] http://localhost:{port}")
    threading.Thread(target=_start_server, args=(port,), daemon=True).start()
    try:
        while True:
            time.sleep(0.5)
            _poke()
    except KeyboardInterrupt:
        print("\n[rrrlinear] bye")


if __name__ == '__main__':
    main()
