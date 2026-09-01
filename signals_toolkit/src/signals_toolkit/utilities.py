import numpy as np


def simulated_adc(
    fundamental_frequency: float,
    fs: float,
    sample_size: int,
    resolution: int,
    signal_amplitude_dbfs: float,
    target_snr_db: float,
    harmonics_dbc: list[float],
    signed: bool = False,
) -> np.ndarray:

    if signal_amplitude_dbfs > 0:
        raise ValueError("Target level (dBFS) should be <= 0!")

    t_span = (1 / fs) * float(sample_size)
    t = np.linspace(0, t_span, sample_size, endpoint=False)

    # 1. Generate fundamental signal
    f0_amplitude = 10 ** (signal_amplitude_dbfs / 20)
    signal = f0_amplitude * np.sin(2 * np.pi * fundamental_frequency * t)

    # 2. Add harmonics (Harmonic multiplier h >= 2)
    i = 2
    for harmonic_level in harmonics_dbc:
        amplitude = f0_amplitude * (
            10 ** (harmonic_level / 20.0)
        )  # Relative to fundamental
        signal += amplitude * np.sin(2 * np.pi * i * fundamental_frequency * t)
        i += 1

    # 3. Add noise relative to fundamental signal RMS
    signal_rms = f0_amplitude / np.sqrt(2)
    noise_rms = signal_rms / (10 ** (target_snr_db / 20.0))
    noise_signal = np.random.normal(0, noise_rms, sample_size)
    signal += noise_signal

    # 4. Scale to ADC Range & Quantize
    if signed:
        max_code = (2 ** (resolution - 1)) - 1
        min_code = -(2 ** (resolution - 1))

        scaled_signal = signal * max_code
        adc_codes = np.clip(np.round(scaled_signal), min_code, max_code).astype(
            np.int32
        )
    else:
        max_code = (2**resolution) - 1
        mid_code = (2 ** (resolution - 1)) - 0.5
        amplitude = (2 ** (resolution - 1)) - 1

        scaled_signal = mid_code + (signal * amplitude)
        adc_codes = np.clip(np.round(scaled_signal), 0, max_code).astype(np.uint32)

    return adc_codes


def sinefit_4param(
    adc_samples: np.ndarray,
    fs: float,
    f0_initial: float,
    max_iter: int = 30,
    tol: float = 1e-12,
    damping_factor: float = 1e-3,
):
    """IEEE Std 1241 4-Parameter Sine Fit using ONLY NumPy.

    Features time-centering for conditioning and Levenberg-Marquardt damping
    for stable convergence without Scipy dependencies.

    Returns:
        f0_fitted (Hz)
        amplitude_fitted (LSBs)
        phase_fitted (radians, relative to t=0)
        dc_offset_fitted (LSBs)
    """
    N = len(adc_samples)
    y = np.asarray(adc_samples, dtype=np.float64)

    # 1. Time-centering: keeps Jacobian terms balanced in float64 precision
    t_raw = np.arange(N, dtype=np.float64) / fs
    t_mid = t_raw[N // 2]
    t = t_raw - t_mid

    omega = 2 * np.pi * f0_initial

    # Initial 3-Parameter fit at centered time base
    D3 = np.column_stack([np.cos(omega * t), np.sin(omega * t), np.ones(N)])
    Ac, As, C = np.linalg.lstsq(D3, y, rcond=None)[0]

    lm_lambda = damping_factor

    for _ in range(max_iter):
        cos_wt = np.cos(omega * t)
        sin_wt = np.sin(omega * t)

        # Model prediction & residual
        y_hat = Ac * cos_wt + As * sin_wt + C
        r = y - y_hat
        res_sum_sq = np.dot(r, r)

        # Partial derivative wrt omega on centered time
        d_omega = -t * Ac * sin_wt + t * As * cos_wt

        # Analytical Jacobian: [d/dAc, d/dAs, d/dC, d/dOmega]
        J = np.column_stack([cos_wt, sin_wt, np.ones(N), d_omega])

        # Normal equations assembly (4x4 matrix)
        JTJ = J.T @ J
        JTr = J.T @ r

        # Add Levenberg-Marquardt damping to frequency parameter
        JTJ_damped = JTJ.copy()
        JTJ_damped[3, 3] += lm_lambda * JTJ[3, 3]

        try:
            # 4x4 solve via pure NumPy
            delta = np.linalg.solve(JTJ_damped, JTr)
        except np.linalg.LinAlgError:
            break

        # Proposed parameter updates
        Ac_new = Ac + delta[0]
        As_new = As + delta[1]
        C_new = C + delta[2]
        omega_new = omega + delta[3]

        # Evaluate candidate residual sum of squares
        r_new = y - (
            Ac_new * np.cos(omega_new * t) + As_new * np.sin(omega_new * t) + C_new
        )
        new_res_sum_sq = np.dot(r_new, r_new)

        # Step acceptance logic
        if new_res_sum_sq < res_sum_sq:
            Ac, As, C, omega = Ac_new, As_new, C_new, omega_new
            lm_lambda = max(1e-7, lm_lambda / 3.0)
            if np.abs(delta[3]) < tol:
                break
        else:
            lm_lambda *= 2.0  # Reject step, increase damping

    # 2. Convert parameters back to original time origin (t=0)
    amplitude = np.hypot(Ac, As)
    phi_centered = np.arctan2(-As, Ac)

    # Re-reference phase to t=0
    phi_0 = (phi_centered - omega * t_mid) % (2 * np.pi)
    f0_fitted = omega / (2 * np.pi)

    return f0_fitted, amplitude, phi_0, C
