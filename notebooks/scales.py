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
    import librosa
    import scipy.signal


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
    y1, sr1 = load_audio(file_browser1.path(), normalize=False)
    show_audio(y1, sr1, file_browser1.path())
    return sr1, y1


@app.cell
def _(file_browser2):
    y2, sr2 = load_audio(file_browser2.path(), normalize=False)
    show_audio(y2, sr2, file_browser2.path())
    return sr2, y2


@app.function(hide_code=True)
def get_partials(y, sr, n_fft=2048):
    # Compute the partials tracking
    # S: magnitude, phase components (optional, piptrack can compute S internally)

    window = scipy.signal.get_window("hann", n_fft)
    window_sum = np.sum(window)

    S = np.abs(librosa.stft(y, n_fft=n_fft, window="hann"))

    # piptrack returns two 2D arrays:
    # 'freqs': the estimated frequency of the partial at each bin/time
    # 'mags': the magnitude of that partial
    freqs, mags = librosa.piptrack(S=S, sr=sr)
    mags = mags * (2.0 / window_sum)

    # Select a specific frame to analyze (e.g., the midpoint of the audio)
    # or you could average over time. Here we take the loudest frame.
    frame_idx = np.argmax(np.sum(S, axis=0))

    # Extract data for that frame
    f_col = freqs[:, frame_idx]
    m_col = mags[:, frame_idx]

    # Filter out zeros (non-peaks)
    mask = m_col > 0

    # Combine frequency and magnitude arrays into a (N, 2) matrix
    # Column 0 = Frequency, Column 1 = Amplitude
    partials = np.column_stack((f_col[mask], m_col[mask]))

    # Sort by Amplitude (Column 1) in descending order
    # argsort() gives indices for ascending order, [::-1] reverses it
    partials = partials[partials[:, 1].argsort()[::-1]]

    return partials


@app.cell
def _(sr1, sr2, y1, y2):
    partials1 = get_partials(y1, sr1)
    partials2 = get_partials(y2, sr2)
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
        partials1, partials2, ratio_min=1 / 16, ratio_max=2
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
                ((partials1[:, 1] / partials1[:, 1].max()) ** 0.5) * 127
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
                ((partials2[:, 1] / partials2[:, 1].max()) ** 0.5) * 127
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
    return


if __name__ == "__main__":
    app.run()
