import threading
import time

import tinysoundfont
from marimo import Thread


class MicrotonalPlayer:
    def __init__(self, sf2_path, bend_range=2):
        self.synth = tinysoundfont.Synth()
        sfid = self.synth.sfload(sf2_path)

        # Determine accessible channels (0-15), skip Ch 9 (Drums)
        self.channels = [i for i in range(32) if i != 9]

        # Initialize channels
        for ch in self.channels:
            self.synth.program_select(ch, sfid, 0, 0)

        self.synth.start()

        # Configuration
        self.bend_range = bend_range

        # Thread Safety Tools
        self.lock = threading.Lock()  # Protects channel allocation
        self.stop_event = threading.Event()  # Allows killing the thread

        # State
        self.free_channels = list(self.channels)
        self.active_notes = {}  # Map: unique_note_id -> (channel, midi_note)

    def _get_pitch_data(self, float_pitch):
        """Calculates MIDI Note and Pitch Bend for 1/8 tone precision."""
        # 1. Quantize to 1/8 tone (0.25 semitones)
        q_pitch = round(float_pitch * 4) / 4.0

        # 2. Nearest integer MIDI note
        midi_note = int(round(q_pitch))

        # 3. Deviation (-0.5 to +0.5)
        deviation = q_pitch - midi_note

        # 4. Map to 14-bit Pitch Bend (Center 8192)
        # 1 Semitone = 8192 / bend_range
        units = 8192 / self.bend_range
        bend_val = int(8192 + (deviation * units))

        return midi_note, max(0, min(16383, bend_val))

    def play(self, note_data):
        """
        Starts playback in a background thread. Returns immediately.

        Args:
            note_data: List of (pitch, velocity, delta_time, duration)
        """
        # Clear any previous stop signals
        self.stop_event.clear()

        # Create and start the thread
        t = Thread(target=self._playback_loop, args=(note_data,))
        t.daemon = True  # Thread dies if main program exits
        t.start()
        return t

    def stop(self):
        """Interrupts playback immediately."""
        self.stop_event.set()

    def _playback_loop(self, note_data):
        """The actual logic running in the background."""

        # 1. Pre-calculate the absolute Event Timeline
        events = []  # (timestamp, type, id, pitch, vel)
        current_cursor = 0.0

        for i, (pitch, vel, delta, dur) in enumerate(note_data):
            current_cursor += delta

            # Type 'ON'
            events.append((current_cursor, "ON", i, pitch, vel))

            # Type 'OFF'
            events.append((current_cursor + dur, "OFF", i, pitch, 0))

        # Sort: Time first, then OFF events before ON events (to recycle channels immediately)
        events.sort(key=lambda x: (x[0], 0 if x[1] == "OFF" else 1))

        start_time = time.time()

        for timestamp, etype, uid, pitch, vel in events:
            # Check if user requested stop
            if self.stop_event.is_set():
                self._all_notes_off()
                return

            # Calculate wait time
            now = time.time() - start_time
            wait_time = timestamp - now

            if wait_time > 0:
                # wait() blocks for 'wait_time', but returns immediately if stop_event is set
                if self.stop_event.wait(wait_time):
                    self._all_notes_off()
                    return

            # --- CRITICAL SECTION (Channel Management) ---
            with self.lock:
                if etype == "ON":
                    if self.free_channels:
                        channel = self.free_channels.pop(0)
                        midi_note, bend = self._get_pitch_data(pitch)

                        self.active_notes[uid] = (channel, midi_note)

                        self.synth.pitchbend(channel, bend)
                        self.synth.noteon(channel, midi_note, int(vel))
                    else:
                        print("Polyphony Limit: Note dropped.")

                elif etype == "OFF":
                    if uid in self.active_notes:
                        channel, midi_note = self.active_notes.pop(uid)
                        self.synth.noteoff(channel, midi_note)
                        self.free_channels.append(channel)

    def _all_notes_off(self):
        """Panic button to silence everything."""
        with self.lock:
            for ch in self.channels:
                self.synth.notes_off(ch)
            # Reset state
            self.free_channels = list(self.channels)
            self.active_notes = {}
