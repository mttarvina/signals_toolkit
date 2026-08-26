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
