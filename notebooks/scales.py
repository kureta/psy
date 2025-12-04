import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from pathlib import Path
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    import numpy as np
    import time
    import librosa
    import scipy.signal
    from psy.psy import MicrotonalPlayer
    from collections import defaultdict


    sf2_path = "/usr/share/soundfonts/FluidR3_GM.sf2"
    player = MicrotonalPlayer(sf2_path)


@app.cell(hide_code=True)
def _():
    file_browser1 = mo.ui.file_browser(
        initial_path=Path("~/Music").expanduser(),
        filetypes=["wav", "mp3"],
        selection_mode="file",
        multiple=False,
        restrict_navigation=False,
        label="Select audio file",
        on_change=None,
        ignore_empty_dirs=False,
    )

    file_browser1
    return (file_browser1,)


@app.cell(hide_code=True)
def _():
    file_browser2 = mo.ui.file_browser(
        initial_path=Path("~/Music").expanduser(),
        filetypes=["wav", "mp3"],
        selection_mode="file",
        multiple=False,
        restrict_navigation=False,
        label="Select audio file",
        on_change=None,
        ignore_empty_dirs=False,
    )

    file_browser2
    return (file_browser2,)


@app.function(hide_code=True)
def load_audio(file_path, normalize=True):
    sample, sample_rate = librosa.load(file_path, sr=None, mono=True)
    if normalize:
        max_val = np.abs(sample).max()
        if max_val > 0:
            sample = sample / max_val
    return sample, sample_rate


@app.function(hide_code=True)
def show_audio(y, sr, label=""):
    try:
        audio_load_display = mo.vstack(
            [
                mo.md(f"**Loaded:** `{label}`"),
                mo.audio(src=y, rate=sr, normalize=False),
            ]
        )
    except Exception as e:
        y = None
        audio_load_display = mo.md(f"**Error loading file:** {e}")

    return audio_load_display


@app.cell
def _(file_browser1):
    y1, sr1 = load_audio(file_browser1.path(), normalize=True)
    show_audio(y1, sr1, file_browser1.path())
    return sr1, y1


@app.cell
def _(file_browser2):
    y2, sr2 = load_audio(file_browser2.path(), normalize=True)
    show_audio(y2, sr2, file_browser2.path())
    return sr2, y2


@app.function(hide_code=True)
def merge_nearby_partials(partials, distance_cents=50):
    """
    Combines partials that are closer than 'distance_cents'.
    Weighted average is used for Frequency. Sum is used for Amplitude (energy preservation).
    """
    if len(partials) == 0:
        return partials

    # Sort by frequency (ascending) for easier clustering
    # partials shape is (N, 2) -> (Freq, Amp)
    p = partials[partials[:, 0].argsort()]

    merged = []

    # Initialize the first cluster
    current_freqs = [p[0, 0]]
    current_amps = [p[0, 1]]

    for i in range(1, len(p)):
        f_current = p[i, 0]
        a_current = p[i, 1]

        f_prev_avg = np.average(current_freqs, weights=current_amps)

        # Calculate distance in cents
        # Cents = 1200 * log2(f2 / f1)
        if f_prev_avg > 0:
            cents_diff = 1200 * np.log2(f_current / f_prev_avg)
        else:
            cents_diff = float("inf")

        if cents_diff < distance_cents:
            # Add to current cluster
            current_freqs.append(f_current)
            current_amps.append(a_current)
        else:
            # Cluster is done. Calculate weighted properties and save.
            # 1. Frequency = Amplitude-Weighted Average
            f_final = np.average(current_freqs, weights=current_amps)

            # 2. Amplitude = Sum (preserving total energy of the smear)
            # OR Max, depending on preference. Sum is usually physically correct for splitting.
            # But since vibrato is one signal moving, 'Max' might be safer to prevent loudness boost.
            # Let's use Root Mean Square (RMS) summation for uncorrelated signals,
            # or simply MAX if we assume it's the same partial.
            # Let's stick to MAX for safety against "loudness" issues.
            a_final = np.sum(current_amps)

            merged.append([f_final, a_final])

            # Start new cluster
            current_freqs = [f_current]
            current_amps = [a_current]

    # Append the last cluster
    f_final = np.average(current_freqs, weights=current_amps)
    a_final = np.max(current_amps)
    merged.append([f_final, a_final])

    return np.array(merged)


@app.function(hide_code=True)
def get_partials(
    y, sr, n_fft=2048, fmin=25, fmax=4400, threshold=0.1, bin_width_cents=25
):
    # 1. Compute STFT and Window Sum
    window = scipy.signal.get_window("hann", n_fft)
    window_sum = np.sum(window)
    S = np.abs(librosa.stft(y, n_fft=n_fft, window=window))

    # 2. Run Piptrack
    # 'freqs' and 'mags' are size (bins, time_frames)
    freqs, mags = librosa.piptrack(
        S=S, sr=sr, fmin=fmin, fmax=fmax, threshold=threshold
    )

    # 3. Apply True Amplitude Normalization
    mags = mags * (2.0 / window_sum)

    # --- AVERAGING LOGIC ---

    # A. Calculate Magnitude Sum per bin (needed for weighting)
    # Shape: (bins,)
    sum_mags = np.sum(mags, axis=1)

    # B. Calculate Weighted Average Frequency per bin
    # Formula: Sum(freq * mag) / Sum(mag)
    # This ignores the frames where freq is 0 (because mag is also 0)
    weighted_freq_sum = np.sum(freqs * mags, axis=1)

    # Safe division (handle bins where sum_mags is 0)
    avg_freqs = np.divide(
        weighted_freq_sum,
        sum_mags,
        out=np.zeros_like(weighted_freq_sum),
        where=sum_mags != 0,
    )

    # C. Calculate Average Magnitude per bin
    # We take the mean across time.
    # Note: This averages silence (zeros) too. If a partial is short,
    # its average amplitude will be low. This is usually desired for a static snapshot.
    avg_mags = np.mean(mags, axis=1)

    # --- CLEANUP ---

    # 4. Filter out empty bins
    mask = avg_mags > 0

    # 5. Stack into matrix
    partials = np.column_stack((avg_freqs[mask], avg_mags[mask]))

    # --- FIX FOR VIBRATO: MERGE ---
    # Merge peaks closer than 50 cents (1/2 semitone)
    partials = merge_nearby_partials(partials, distance_cents=bin_width_cents)

    # 6. Sort by Amplitude (descending)
    partials = partials[partials[:, 1].argsort()[::-1]]

    return partials


@app.cell
def _(sr1, sr2, y1, y2):
    partials1 = get_partials(y1, sr1)
    partials2 = get_partials(y2, sr2)
    mo.md(f"np1 = {len(partials1)} - np2 = {len(partials2)}")
    return partials1, partials2


@app.function(hide_code=True)
def additive_synthesis(partials, duration, sr=44100, random_phases=True):
    """
    Synthesizes sound with random phases to improve realism and
    normalizes the final sum to prevent clipping/loudness issues.
    """
    # 1. Convert to NumPy arrays
    freqs = partials[:, 0]
    amps = partials[:, 1]

    # Validation: Ensure they are the same length
    if len(freqs) != len(amps):
        raise ValueError(
            "Frequencies and Amplitudes must have the same length."
        )

    # 2. Time Vector (1, N_samples)
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)

    # 3. Generate Phases
    if random_phases:
        # Generate one random phase per partial between 0 and 2pi
        # Shape: (N_partials,)
        phases = np.random.uniform(0, 2 * np.pi, len(freqs))
    else:
        phases = np.zeros(len(freqs))

    # 4. Vectorized Synthesis
    # Broadcast shapes:
    # Freqs:  (N, 1)
    # Time:   (1, Samples)
    # Phases: (N, 1) -> Must reshape to column vector to add correctly

    # Arguments for sin: 2 * pi * f * t + phase
    theta = (2 * np.pi * freqs[:, np.newaxis] * t) + phases[:, np.newaxis]

    # Calculate sines and apply partial amplitudes
    partials_matrix = amps[:, np.newaxis] * np.sin(theta)

    # 5. Summation (Mixing)
    audio_signal = np.sum(partials_matrix, axis=0)

    # 6. Safety Normalization (Fixes the "Louder" issue)
    # This ensures the loudest peak in the entire file is exactly -1.0 or 1.0
    peak_amplitude = np.max(np.abs(audio_signal))

    if peak_amplitude > 0:
        audio_signal = audio_signal / peak_amplitude
        # Optional: Multiply by 0.9 to leave a little headroom (approx -1dB)
        audio_signal = audio_signal * 0.9

    return audio_signal


@app.cell
def _(partials1, partials2, sr1, sr2):
    resynthesized1 = additive_synthesis(partials1, 3.0, sr1)
    resynthesized2 = additive_synthesis(partials2, 3.0, sr2)
    return resynthesized1, resynthesized2


@app.cell
def _(resynthesized1, sr1):
    show_audio(resynthesized1, sr1, "Resynthesized partials 1")
    return


@app.cell
def _(resynthesized2, sr2):
    show_audio(resynthesized2, sr2, "Resynthesized partials 2")
    return


@app.function(hide_code=True)
def calculate_consonant_intervals(
    partials_base, partials_interval, ratio_min=1.0, ratio_max=2.0
):
    # Unpack into arrays for vectorized calculation
    # normalize frequency
    freqs0 = partials_base[:, 0] / partials_base[:, 0].min()
    amps0 = partials_base[:, 1]
    # normalize frequency
    freqs1 = partials_interval[:, 0] / partials_interval[:, 0].min()
    amps1 = partials_interval[:, 1]

    # 1. Calculate all possible ratios k = p0 / p1
    # Using an outer division to get a matrix of all combinations
    # shape: (len(freqs0), len(freqs1))
    ratios_k = freqs0[:, np.newaxis] / freqs1

    # 2. Calculate weights (product of amplitudes)
    # Stronger partials colliding = stronger consonance
    weights = amps0[:, np.newaxis] * amps1

    # 3. Flatten arrays to 1D lists
    ratios_flat = ratios_k.flatten()
    weights_flat = weights.flatten()

    # 4. Filter by range (e.g., 1.0 to 2.0)
    mask = (ratios_flat >= ratio_min) & (ratios_flat <= ratio_max)
    valid_ratios = ratios_flat[mask]
    valid_weights = weights_flat[mask]

    # 5. Group by unique ratio (rounding to avoid floating point errors)
    # We round to 5 decimal places to group "near-identical" ratios
    unique_ratios, inverse_indices = np.unique(
        valid_ratios.round(5), return_inverse=True
    )

    # Sum weights for duplicate ratios
    summed_weights = np.zeros_like(unique_ratios)
    np.add.at(summed_weights, inverse_indices, valid_weights)

    # 6. Sort by weight (descending) to find the "best" intervals
    sorted_indices = np.argsort(summed_weights)[::-1]
    # sorted_indices = np.arange(len(summed_weights))

    results = np.column_stack(
        (unique_ratios[sorted_indices], summed_weights[sorted_indices])
    )

    return results


@app.function
def round_to(x, cents=25):
    return cents * np.round(x / cents)


@app.cell
def _(partials1, partials2):
    partials1[:, 0] = round_to(partials1[:, 0], 1)
    partials2[:, 0] = round_to(partials2[:, 0], 1)
    intervals = calculate_consonant_intervals(
        partials1[:8], partials2[:8], ratio_min=1 / 8, ratio_max=2
    )
    return (intervals,)


@app.cell
def _(intervals):
    cents = [int(round_to(1200 * np.log2(ratio), 1)) for ratio, weight in intervals]

    print("(" + " ".join([str(c) for c in cents]) + ")")
    return


@app.cell
def _(partials1):
    " ".join(
        [
            str(int(a))
            for a in np.round(
                ((partials1[:, 1] / partials1[:, 1].max()) ** 0.1) * 127
            )
        ]
    )
    return


@app.cell
def _(partials2):
    " ".join(
        [
            str(int(a))
            for a in np.round(
                ((partials2[:, 1] / partials2[:, 1].max()) ** 0.1) * 127
            )
        ]
    )
    return


@app.cell
def _(partials1):
    " ".join(
        [
            str(int(round_to(a, 1)))
            for a in librosa.hz_to_midi(partials1[:, 0]) * 100.0
        ]
    )
    return


@app.cell
def _(partials2):
    " ".join(
        [
            str(int(round_to(a, 1)))
            for a in librosa.hz_to_midi(partials2[:, 0]) * 100.0
        ]
    )
    return


@app.cell
def _():
    # 2. Define Music (Pitch, Velocity, Delta from previous, Duration)
    # Let's play a C Major chord where the E is a Neutral Third (3.5 semitones)
    # And the G is slightly sharp.
    sequence = [
        # Pitch, Vel, Delta, Dur
        (60.0, 100, 0.0, 2.0),  # C (Start at 0)
        (63.5, 100, 0.5, 2.0),  # Neutral Third (Start at 0.5s) - 1/8 tone precision
        (67.12, 100, 0.5, 2.0),  # G (Start at 1.0s) - .12 will round to .0 or .25
        (72.0, 100, 1.0, 1.0),  # High C (Start at 2.0s)
        (72.25, 100, 1.0, 1.0),
        (72.50, 100, 1.0, 1.0),
        (72.75, 100, 1.0, 1.0),
        (73.0, 100, 1.0, 1.0),
    ]

    print("Playing microtonal sequence...")
    player.play(sequence)
    print("Done.")
    return


@app.cell
def _(partials1):
    player._all_notes_off()
    time.sleep(0.1)
    seq = {}
    for p, v in partials1:
        seq[round_to(librosa.hz_to_midi(p) * 100, 25) / 100.0] = (
            int(v**0.01 * 127),
            0.125,
            4.0,
        )
    seqs = [(k, *v) for k, v in seq.items()]
    player.play(seqs)
    return


@app.cell
def _(intervals, partials1):
    centos = [(1200 * np.log2(ratio), weight) for ratio, weight in intervals][:16]

    print(len(centos))
    mek = {}
    for pp, vv in centos:
        mek[
            round_to(librosa.hz_to_midi(partials1[0, 0]) * 100 + pp, 25) / 100.0
        ] = (
            int(vv**0.25 * 127) + 35,
            0.125,
            0.5,
        )
    meks = [(k, *v) for k, v in mek.items()]
    player.play(meks)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
