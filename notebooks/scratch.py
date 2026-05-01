import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots


@app.function
def erb(x):
    return x / 9.2645 + 24.7


@app.function
def roughness(x, f0=110.0):
    x /= erb(f0)
    return 4 * np.abs(x) * np.exp(1 - 4 * np.abs(x))


@app.cell
def _():
    x = np.linspace(0.0, 110.0, 1000)
    y = roughness(x)
    return x, y


@app.cell
def _(x, y):
    plt.plot(x, y)
    return


@app.function
def get_harmonics(n_partials=8, exp=1.0):
    amps = 1.0 / np.arange(1, n_partials + 1)
    parts = np.arange(1, n_partials + 1) ** exp

    return amps, parts


@app.function
def dissonance_curve(f_base, alpha_range, partials1, partials2):
    # 1. Calculate amplitudes and pairwise weights
    # amps shape: (N,) -> weights shape: (N, N)
    amps1, p1 = partials1
    amps2, p2 = partials2
    weights = amps1[:, np.newaxis] * amps2[np.newaxis, :]

    # 2. Setup partials with explicit dimensions for broadcasting
    # p1 shape: (1, N, 1) - Tone 1 partials
    p1 = (f_base * p1)[np.newaxis, :, np.newaxis]

    # p2 shape: (M, 1, N) - Tone 2 partials over all alphas
    p2 = (
        f_base
        * alpha_range[:, np.newaxis, np.newaxis]
        * p2[np.newaxis, np.newaxis, :]
    )

    # 3. Vectorized math. Resulting shapes are (M, N, N)
    df = np.abs(p1 - p2)
    f_mean = (p1 + p2) / 2.0

    # 4. Calculate roughness, apply weights, and sum across the N x N grid
    r = weights * roughness(df, f_mean)

    # Sum over axes 1 and 2 (the two partial dimensions) to get total roughness per alpha
    return np.sum(r, axis=(1, 2))


@app.cell(hide_code=True)
def _():
    alpha_slider = mo.ui.slider(
        start=1.0, stop=2.0, step=0.01, value=1.5, label="Frequency Ratio (Alpha)"
    )
    exp_slider = mo.ui.slider(
        start=0.5,
        stop=2.5,
        step=0.01,
        value=1.0,
        label="Stretch Factor (Inharmonicity)",
    )

    mo.vstack([mo.md("### Interval & Spectra Explorer"), alpha_slider, exp_slider])
    return alpha_slider, exp_slider


@app.cell(hide_code=True)
def _(alpha_slider, exp_slider):
    f_base = 440.0
    n_partials = 8

    # 1. Grab current values from the UI
    current_alpha = alpha_slider.value
    current_exp = exp_slider.value

    # 2. Compute Partials
    partials1 = get_harmonics(n_partials, current_exp)
    partials2 = get_harmonics(n_partials, current_exp)

    # 3. Compute Dissonance Data
    alpha_range = np.linspace(1.0, 2.0, 400)
    diss = dissonance_curve(f_base, alpha_range, partials1, partials2)
    current_diss = dissonance_curve(
        f_base, np.array([current_alpha]), partials1, partials2
    )[0]

    # 4. Compute Spectral Data
    f1 = f_base * partials1[1]
    f2 = f_base * current_alpha * partials2[1]
    a1 = partials1[0]
    a2 = partials2[0]

    # --- Plotting with Plotly ---
    # Create a figure with 2 subplots
    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0.15
    )

    # Top Plot: The Dissonance Curve
    fig.add_trace(
        go.Scatter(
            x=alpha_range,
            y=diss,
            mode="lines",
            line=dict(color="black", width=2),
            name="Dissonance Curve",
        ),
        row=1,
        col=1,
    )

    # Marker for the current interval
    fig.add_trace(
        go.Scatter(
            x=[current_alpha],
            y=[current_diss],
            mode="markers",
            marker=dict(color="red", size=12),
            name="Current Ratio",
        ),
        row=1,
        col=1,
    )

    # Vertical guideline
    fig.add_vline(
        x=current_alpha,
        line_width=2,
        line_dash="dash",
        line_color="red",
        opacity=0.5,
        row=1,
        col=1,
    )

    # Bottom Plot: Spectral Alignment
    # Using Bar charts with a very thin width creates a great looking "stem" plot for spectra
    fig.add_trace(
        go.Bar(
            x=f1,
            y=a1,
            name="Tone 1 (Base)",
            marker_color="#1f77b4",
            width=10,
            opacity=0.75,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=f2,
            y=a2,
            name="Tone 2 (Shifted)",
            marker_color="#ff7f0e",
            width=10,
            opacity=0.75,
        ),
        row=2,
        col=1,
    )

    # Layout styling
    fig.update_layout(
        title=f"Sensory Dissonance (Stretch = {current_exp:.2f})",
        height=650,
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        barmode="overlay",
    )

    fig.update_yaxes(title_text="Total Roughness", row=1, col=1)
    fig.update_xaxes(title_text="Frequency Ratio", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)

    # In Marimo, returning the figure renders it natively as an interactive web element
    fig
    return


@app.cell(hide_code=True)
def _():
    # Create options 1 through 16
    partial_options = {f"Partial {i}": i for i in range(1, 65)}

    p1_dropdown = mo.ui.dropdown(
        options=partial_options,
        value="Partial 3",  # Default to 3
        label="Tone 1 Partial (n)",
    )

    p2_dropdown = mo.ui.dropdown(
        options=partial_options,
        value="Partial 2",  # Default to 2 (creates a 3/2 Perfect 5th by default)
        label="Tone 2 Partial (m)",
    )

    exp_slider2 = mo.ui.slider(
        start=0.95,
        stop=1.05,
        step=0.001,
        value=1.0,
        label="Stretch Factor (Inharmonicity)",
    )

    exp_slider3 = mo.ui.slider(
        start=0.95,
        stop=1.05,
        step=0.001,
        value=1.0,
        label="Stretch Factor (Inharmonicity)",
    )

    mo.vstack(
        [
            mo.md("### Target Coincidence Explorer"),
            mo.hstack([p1_dropdown, p2_dropdown], justify="start"),
            mo.hstack([exp_slider2, exp_slider3], justify="start"),
        ]
    )
    return exp_slider2, exp_slider3, p1_dropdown, p2_dropdown


@app.cell
def _(plot):
    plot()
    return


@app.cell(hide_code=True)
def _(exp_slider2, exp_slider3, p1_dropdown, p2_dropdown):
    def plot():
        f_base = 220.0
        n_partials = 64

        # 1. Grab values from UI
        n = p1_dropdown.value
        m = p2_dropdown.value
        exp1 = exp_slider2.value
        exp2 = exp_slider3.value

        # 2. Calculate the exact ratio and convert to Cents
        current_alpha = (n**exp1) / (m**exp2)
        current_cents = 1200 * np.log2(current_alpha)

        # 3. Compute Partials independently
        partials1 = get_harmonics(n_partials, exp1)
        partials2 = get_harmonics(n_partials, exp2)

        # 4. Compute Dissonance Curve Data in Cents
        min_cents = min(0.0, current_cents - 600)
        max_cents = max(1200.0, current_cents + 600)
        cents_range = np.linspace(min_cents, max_cents, 400)
        alpha_range = 2.0 ** (cents_range / 1200.0)

        diss = dissonance_curve(f_base, alpha_range, partials1, partials2)
        current_diss = dissonance_curve(
            f_base, np.array([current_alpha]), partials1, partials2
        )[0]

        # 5. Compute Spectral Data
        f1 = f_base * partials1[1]
        f2 = f_base * current_alpha * partials2[1]
        a1 = partials1[0]
        a2 = partials2[0]

        # --- Color Highlighting Logic ---
        # Create arrays for colors, making the target partials pop out
        colors1 = ["#1f77b4"] * n_partials
        colors1[n - 1] = "#d62728"  # Highlight target Tone 1 partial (Red)

        colors2 = ["#ff7f0e"] * n_partials
        colors2[m - 1] = "#d62728"  # Highlight target Tone 2 partial (Red)

        # --- Plotting ---
        fig = make_subplots(
            rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0.15
        )

        # Dissonance Curve (Top)
        fig.add_trace(
            go.Scatter(
                x=cents_range,
                y=diss,
                mode="lines",
                line=dict(color="black", width=2),
                name="Dissonance Curve",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[current_cents],
                y=[current_diss],
                mode="markers",
                marker=dict(color="red", size=12),
                name=f"Interval: {current_cents:.1f} ¢",
            ),
            row=1,
            col=1,
        )
        fig.add_vline(
            x=current_cents,
            line_width=2,
            line_dash="dash",
            line_color="red",
            opacity=0.5,
            row=1,
            col=1,
        )

        # --- Spectral Stem Plots (Bottom - MIRRORED) ---
        # Define the exact physical width in Hz, and half of it for the center offset
        bar_width = 40
        bar_offset = -bar_width / 2

        fig.add_trace(
            go.Bar(
                x=f1,
                y=a1,
                name="Tone 1 (Base)",
                marker_color=colors1,
                width=bar_width,
                offset=bar_offset,  # <-- This forces perfect centering
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=f2,
                y=-a2,
                name="Tone 2 (Shifted)",
                marker_color=colors2,
                width=bar_width,
                offset=bar_offset,  # <-- This forces perfect centering
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            title=f"Coincidence: Tone 1 (Partial {n}, Stretch {exp1:.2f}) == Tone 2 (Partial {m}, Stretch {exp2:.2f})",
            title_font_size=12,
            height=650,
            margin=dict(l=40, r=40, t=60, b=40),
            # Change hovermode to 'closest' to fix the snapping issue
            hovermode="closest",
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        fig.update_yaxes(title_text="Total Roughness", row=1, col=1)
        fig.update_xaxes(title_text="Interval (Cents)", row=1, col=1)

        # Format the bottom Y-axis to hide the negative signs, making it look like pure amplitude
        fig.update_yaxes(title_text="Amplitude", tickformat=".2f", row=2, col=1)
        fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)

        # Add a subtle zero-line to the bottom plot to anchor the mirrored bars
        fig.add_hline(y=0, line_width=1, line_color="black", row=2, col=1)

        return fig
    return (plot,)


@app.cell
def _():
    5/4, 3/4
    return


@app.cell
def _():
    np.sqrt(2)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
