# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.0",
#     "numpy==2.4.4",
#     "plotly==6.6.0",
# ]
# ///

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    return go, mo, np


@app.cell(hide_code=True)
def header(mo):
    mo.md(r"""
    # Frequency Line Tracking Using HMM-based Schemes
    **An Interactive Guide to Paris & Jauffret (2003)**

    In passive sonar systems, detecting and tracking moving targets (like submarines or surface ships) often relies on identifying narrow acoustic emissions, such as the noise from rotating motor pieces. These emissions appear as **fluctuating frequency lines** embedded in severe background noise on a time-frequency image known as a *lofargram* (Low Frequency Analysis and Recording).

    The challenge is twofold:
    1. The target's frequency line is non-deterministic (it wanders randomly).
    2. The Signal-to-Noise Ratio (SNR) is extremely low and often unknown.

    This guide walks step-by-step through the solution proposed by Paris & Jauffret: modeling the frequency line as a **Hidden Markov Model (HMM)** and using "probabilistic integration of spectral power" to emancipate the tracker from needing to know the SNR.
    """)
    return


@app.cell(hide_code=True)
def theory_state_dynamics(mo):
    mo.md(r"""
    ---
    ## 1. System State and Dynamics

    To accurately track a moving frequency, the authors define a two-dimensional state vector $\mathbf{x}_k$ at time step $k$, representing both the current frequency bin and its slope (velocity):

    $$ \mathbf{x}_k = \begin{bmatrix} f_k \\ \Delta f_k \end{bmatrix} $$

    where $f_k \in \{0, \dots, M-1\}$ represents the discrete frequency bin, and $\Delta f_k \in \{-J, \dots, J\}$ represents the discrete velocity limits.

    The physical motion of the frequency line is modeled as a first-order Markov chain (a random walk with momentum):

    $$ \mathbf{x}_k = \mathbf{H} \mathbf{x}_{k-1} + \mathbf{\eta}_k $$

    Where the transition matrix $\mathbf{H}$ and the plant noise covariance $\mathbf{R}$ are defined as:

    $$ \mathbf{H} = \begin{bmatrix} 1 & \zeta \\ 0 & 1 \end{bmatrix}, \quad \mathbf{R} = g \begin{bmatrix} 1/3 & 1/(2\zeta) \\ 1/(2\zeta) & 1/\zeta^2 \end{bmatrix} $$

    Here, $\zeta$ represents a multiplicative constant for the slope, and $g$ scales the process noise variance. The transition probabilities $a_{ji} = P(\mathbf{x}_k = i \mid \mathbf{x}_{k-1} = j)$ are calculated using the Mahalanobis distance derived from $\mathbf{R}^{-1}$.
    """)
    return


@app.cell
def compute_transition_matrix(np):
    def build_transition_matrix(M, J, S, g, zeta):
        states = np.arange(S)

        # Decode state index into frequency and velocity components
        f_states = states % M
        v_states = (states // M) - J

        N_f, M_f = np.meshgrid(f_states, f_states, indexing="ij")
        N_v, M_v = np.meshgrid(v_states, v_states, indexing="ij")

        # Calculate state deltas based on the H matrix formulation
        delta_f = M_f - N_f - zeta * N_v
        delta_v = M_v - N_v

        # Mahalanobis distance D^2 = X^T R^{-1} X
        dist2 = (1.0 / g) * (
            12 * delta_f**2
            - 12 * zeta * delta_f * delta_v
            + 4 * zeta**2 * delta_v**2
        )

        A = np.exp(-0.5 * dist2)

        # Normalize to ensure valid probability distributions (rows sum to 1)
        A_row_sums = A.sum(axis=1, keepdims=True)
        A = np.divide(A, A_row_sums, out=np.zeros_like(A), where=A_row_sums != 0)
        return A

    return (build_transition_matrix,)


@app.cell(hide_code=True)
def theory_observations(mo):
    mo.md(r"""
    ---
    ## 2. Measurement Genesis (The Lofargram)

    The raw signal is a real, narrowband sinusoid corrupted by additive Gaussian white noise:

    $$ s_k(n\Delta t) = a_k \sin(2\pi f_k n \Delta t + \varphi_k) + \epsilon_k(n\Delta t) $$

    To analyze this, the sonar processor computes the discrete periodogram (squared magnitude of the FFT) for each time block $k$. This sequence of periodograms forms the image known as the **Lofargram**, defined by cells $\mathcal{P}_{k, j}$.

    Because the target is weak, the visual SNR on the lofargram is extremely low, making it difficult for simple thresholding techniques to detect the line without triggering massive false alarms.
    """)
    return


@app.cell
def generate_lofargram(np):
    def simulate_signal(A_matrix, M, N, K, J, SNR_lin):
        true_freq = np.zeros(K, dtype=int)

        f_curr = M // 2
        s_curr = 0
        curr_idx = f_curr + (s_curr + J) * M

        S = M * (2 * J + 1)
        lofargram = np.zeros((K, M))

        sigma = 1.0
        amplitude = np.sqrt(2 * SNR_lin * sigma**2)
        n_vec = np.arange(N)

        # Ensure the random walk remains consistent for a given set of dynamics
        np.random.seed(int(A_matrix[0, 0] * 1000000) % 100000)

        for k in range(K):
            true_freq[k] = f_curr

            # Time-domain signal with random phase
            phi = np.random.uniform(0, 2 * np.pi)
            clean_signal = amplitude * np.sin(2 * np.pi * f_curr * n_vec / N + phi)
            noise = np.random.normal(0, sigma, size=N)

            # Periodogram computation (Lofargram row)
            fft_s = np.fft.fft(clean_signal + noise)
            lofargram[k, :] = (1.0 / N) * np.abs(fft_s[:M]) ** 2

            # Walk to the next state
            if k < K - 1:
                curr_idx = np.random.choice(S, p=A_matrix[curr_idx, :])
                f_curr = curr_idx % M
                s_curr = (curr_idx // M) - J

        return lofargram, true_freq

    return (simulate_signal,)


@app.cell(hide_code=True)
def theory_likelihood(mo):
    mo.md(r"""
    ---
    ## 3. The Nonparametric Likelihood Model

    Traditional tracking methods require knowledge of the exact background noise variance $\sigma_k^2$ and the target's amplitude $a_k$ to formulate the likelihood of an observation. In real-world sonar, these are rarely known.

    The authors circumvent this by proposing a **Probabilistic Integration of Spectral Power**. Instead of attempting to estimate the SNR, they directly use the *normalized periodogram* as the measurement likelihood $\mathbf{B}$:

    $$ b^s_i(z_k) = \frac{\mathcal{P}_{k, i_1}}{\sum_{l=0}^{M-1} \mathcal{P}_{k, l}} $$

    This clever normalization turns the row of the lofargram directly into a probability mass function. It treats the signal energy itself as the probability that the target occupies that frequency bin.
    """)
    return


@app.cell
def compute_likelihoods(np):
    def build_likelihood_matrix(lofargram, M, K, J):
        S = M * (2 * J + 1)
        B = np.zeros((K, S))

        # Normalize the lofargram rows (Equation 66)
        row_sums = np.sum(lofargram, axis=1, keepdims=True)
        norm_P = lofargram / row_sums

        # Assign likelihoods. The likelihood depends ONLY on the position i_1,
        # not the velocity i_2. Therefore, we copy the likelihood across all velocity sub-states.
        for v in range(-J, J + 1):
            B[:, (v + J) * M : (v + J + 1) * M] = norm_P

        return B

    return (build_likelihood_matrix,)


@app.cell(hide_code=True)
def theory_forward_backward(mo):
    mo.md(r"""
    ---
    ## 4. The Forward-Backward Algorithm (FB)

    The Forward-Backward algorithm is a *local optimization* scheme. It computes the exact marginal probability distribution of the target's location at every single time step, given the *entire* history of the lofargram.

    **1. Forward Pass:** Computes the probability of seeing the partial observation sequence up to time $k$ and ending in state $i$.
    $$ \tilde{\alpha}_k(i) = b_i(z_k) \sum_{j} a_{ji} \alpha_{k-1}(j) $$

    **2. Backward Pass:** Computes the probability of the ending observation sequence from $k+1$ to $K$, given that the system is currently in state $i$.
    $$ \tilde{\beta}_k(i) = \sum_{j} b_j(z_{k+1}) a_{ij} \beta_{k+1}(j) $$

    **3. Marginalization:** The total probability $\gamma_k(i)$ that the target is in state $i$ at time $k$ is the normalized product:
    $$ \gamma_k(i) = \frac{\alpha_k(i) \beta_k(i)}{\sum_l \alpha_k(l) \beta_k(l)} $$

    *Note on Numerical Stability:* Because we multiply hundreds of probabilities, the numbers quickly underflow to zero. The algorithm must use scaling factors $c_k$ at each step to keep the math stable.

    Because the FB algorithm provides a full probability distribution, we can extract not only the maximum likelihood track but also the **Expected Frequency** and the **Standard Deviation ($\pm 1\sigma$)**, which acts as a confidence interval for the tracker!
    """)
    return


@app.cell
def run_forward_backward(np):
    def execute_fb(A, B, K, M, J):
        S = M * (2 * J + 1)

        # --- Forward Pass ---
        alpha = np.zeros((K, S))
        c_scale = np.zeros(K)

        pi_init = np.ones(S) / S
        alpha_tilde = pi_init * B[0, :]
        c_scale[0] = np.sum(alpha_tilde)
        alpha[0, :] = alpha_tilde / c_scale[0]

        for k in range(1, K):
            alpha_tilde = B[k, :] * (alpha[k - 1, :] @ A)
            c_scale[k] = np.sum(alpha_tilde)
            alpha[k, :] = alpha_tilde / c_scale[k]

        # --- Backward Pass ---
        beta = np.zeros((K, S))
        beta[K - 1, :] = 1.0 / S

        for k in range(K - 2, -1, -1):
            beta_tilde = A @ (B[k + 1, :] * beta[k + 1, :])
            beta[k, :] = beta_tilde / c_scale[k + 1]

        # --- Marginals ---
        gamma = alpha * beta
        gamma_sums = np.sum(gamma, axis=1, keepdims=True)
        gamma = np.divide(
            gamma, gamma_sums, out=np.zeros_like(gamma), where=gamma_sums != 0
        )

        fb_states = np.argmax(gamma, axis=1)
        fb_freqs = fb_states % M

        # Extract Confidence Intervals (Standard Deviation)
        gamma_f = gamma.reshape(K, 2 * J + 1, M).sum(axis=1)
        freq_bins = np.arange(M)
        expected_freq = np.sum(gamma_f * freq_bins, axis=1)
        variance_freq = np.sum(
            gamma_f * (freq_bins - expected_freq[:, None]) ** 2, axis=1
        )
        std_freq = np.sqrt(variance_freq)

        return fb_freqs, expected_freq, std_freq

    return (execute_fb,)


@app.cell(hide_code=True)
def theory_viterbi(mo):
    mo.md(r"""
    ---
    ## 5. The Viterbi Algorithm

    While Forward-Backward optimizes the state locally at each time $k$, the **Viterbi Algorithm** is a *global optimization* scheme. It relies on the Bellman principle of optimality to find the single most likely continuous path through the entire lofargram.

    $$ \delta_k(i) = b_i(z_k) \max_{j} \left[ \delta_{k-1}(j) a_{ji} \right] $$

    It tracks the best previous state in a path matrix $\psi_k(i)$. Once the end of the lofargram is reached, the algorithm backtracks through $\psi$ to retrieve the optimal sequence.

    To maintain numerical stability without needing scaling factors, the Viterbi algorithm is executed entirely in the logarithmic domain.
    """)
    return


@app.cell
def run_viterbi(np):
    def execute_viterbi(A, B, K, M, J):
        S = M * (2 * J + 1)

        # Execute in Log Space
        log_A = np.log(A + 1e-300)
        log_B = np.log(B + 1e-300)

        delta = np.zeros((K, S))
        psi = np.zeros((K, S), dtype=int)

        pi_init = np.ones(S) / S
        delta[0, :] = np.log(pi_init) + log_B[0, :]

        for k in range(1, K):
            # Broadcast delta across the transition matrix
            val_matrix = delta[k - 1, :][:, None] + log_A

            best_prev_state = np.argmax(val_matrix, axis=0)
            max_val = np.max(val_matrix, axis=0)

            delta[k, :] = max_val + log_B[k, :]
            psi[k, :] = best_prev_state

        # Backtracking
        vit_states = np.zeros(K, dtype=int)
        vit_states[-1] = np.argmax(delta[-1, :])

        for k in range(K - 2, -1, -1):
            vit_states[k] = psi[k + 1, vit_states[k + 1]]

        vit_freqs = vit_states % M

        return vit_freqs

    return (execute_viterbi,)


@app.cell(hide_code=True)
def interactive_dashboard_setup(mo):
    mo.md(r"""
    ---
    ## 6. Interactive Exploration

    The control panel below allows you to alter the physical conditions of the signal and observe how the mathematical algorithms respond.

    *   **SNR (dB):** Adjusts the amount of background noise. The paper successfully tested down to -19 dB. Notice how the Forward-Backward confidence interval (the shaded cyan region) mathematically "widens" when the SNR drops too low, accurately reflecting the model's loss of certainty.
    *   **Process Noise ($g$) & Transition ($\zeta$):** Alters the rigidity and chaotic nature of the Markov random walk.
    """)
    return


@app.cell(hide_code=True)
def ui_controls(mo):
    # Control Panel UI
    snr_slider = mo.ui.slider(
        start=-30.0,
        stop=0.0,
        step=1.0,
        value=-19.0,
        label="SNR (dB)",
        show_value=True,
        debounce=True,
    )
    g_slider = mo.ui.slider(
        start=1.0,
        stop=10.0,
        step=0.1,
        value=3.46,
        label="Process Noise (g)",
        show_value=True,
        debounce=True,
    )
    zeta_slider = mo.ui.slider(
        start=0.1,
        stop=2.0,
        step=0.1,
        value=1.0,
        label="Transition \u03b6",
        show_value=True,
        debounce=True,
    )

    show_true = mo.ui.checkbox(value=True, label="Show True Frequency")
    show_fb = mo.ui.checkbox(value=True, label="Show Forward-Backward (FB)")
    show_vit = mo.ui.checkbox(value=True, label="Show Viterbi")
    show_confidence = mo.ui.checkbox(
        value=True, label="Show FB Confidence (\u00b11 \u03c3)"
    )

    controls = mo.vstack(
        [
            mo.hstack(
                [snr_slider, g_slider, zeta_slider],
                justify="space-around",
                wrap=True,
            ),
            mo.hstack(
                [show_true, show_fb, show_vit, show_confidence],
                justify="center",
                wrap=True,
            ),
        ]
    )
    return (
        controls,
        g_slider,
        show_confidence,
        show_fb,
        show_true,
        show_vit,
        snr_slider,
        zeta_slider,
    )


@app.cell(hide_code=True)
def runner(
    build_likelihood_matrix,
    build_transition_matrix,
    execute_fb,
    execute_viterbi,
    g_slider,
    simulate_signal,
    snr_slider,
    zeta_slider,
):
    # Orchestrates the simulation based on UI parameters
    N_const = 512
    M_const = 256
    K_const = 350
    J_const = 1
    S_const = M_const * (2 * J_const + 1)

    SNR_lin_val = 10 ** (snr_slider.value / 10.0)

    # 1. Build Matrices
    A_mat = build_transition_matrix(
        M_const, J_const, S_const, g_slider.value, zeta_slider.value
    )

    # 2. Simulate Signal
    lofar_sim, true_f = simulate_signal(
        A_mat, M_const, N_const, K_const, J_const, SNR_lin_val
    )

    # 3. Compute Likelihoods
    B_mat = build_likelihood_matrix(lofar_sim, M_const, K_const, J_const)

    # 4. Run Trackers
    fb_f, expected_f, std_f = execute_fb(A_mat, B_mat, K_const, M_const, J_const)
    vit_f = execute_viterbi(A_mat, B_mat, K_const, M_const, J_const)
    return expected_f, fb_f, lofar_sim, std_f, true_f, vit_f


@app.cell(hide_code=True)
def render_dashboard(
    controls,
    expected_f,
    fb_f,
    go,
    lofar_sim,
    mo,
    np,
    show_confidence,
    show_fb,
    show_true,
    show_vit,
    std_f,
    true_f,
    vit_f,
):
    # Plotly visualization
    fig = go.Figure()

    # Apply rendering limits to make the weak signal visually discernible
    vmin = np.percentile(lofar_sim, 5)
    vmax = np.percentile(lofar_sim, 99.5)
    time_steps = np.arange(len(true_f))

    # 1. Background Lofargram Heatmap
    fig.add_trace(
        go.Heatmap(
            z=lofar_sim.T,
            colorscale="Greys",
            zmin=vmin,
            zmax=vmax,
            showscale=False,
            hoverinfo="skip",
        )
    )

    # 2. True Frequency Track
    if show_true.value:
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=true_f,
                mode="lines",
                line=dict(color="green", width=4),
                name="True Frequency",
            )
        )

    # 3. Viterbi Track
    if show_vit.value:
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=vit_f,
                mode="lines",
                line=dict(color="#FF1493", width=2.5),
                name="Viterbi Track",
            )
        )

    # 4. Forward-Backward Track & Confidence Interval
    if show_fb.value:
        if show_confidence.value:
            # Upper bound (invisible line)
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=expected_f + std_f,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            # Lower bound (fills area up to the upper bound)
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=expected_f - std_f,
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(0, 255, 255, 0.3)",
                    line=dict(width=0),
                    name="FB ±1 Std Dev",
                )
            )

        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=fb_f,
                mode="lines",
                line=dict(color="#00FFFF", width=2.5, dash="dash"),
                name="FB Track (MAP)",
            )
        )

    fig.update_layout(
        title="Interactive Lofargram & Frequency Line Tracker",
        xaxis_title="Time Block (k)",
        yaxis_title="Frequency Bin (m)",
        plot_bgcolor="white",
        height=550,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
    )

    # Compute Metrics
    mse_fb = np.mean((true_f - fb_f) ** 2)
    mse_vit = np.mean((true_f - vit_f) ** 2)

    metrics_md = f"""
    ### 📊 Tracking Performance Metrics

    **Mean Squared Error (MSE):**
    - **Forward-Backward:** `{mse_fb:.4f}`
    - **Viterbi:** `{mse_vit:.4f}`

    *As noted in the paper, both algorithms perform nearly identically, proving the efficacy of the SNR-free probabilistic integration approach.*
    """

    dashboard = mo.vstack([controls, mo.md("<br>"), fig, mo.md(metrics_md)])

    dashboard
    return


if __name__ == "__main__":
    app.run()
