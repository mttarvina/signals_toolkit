import numpy as np


def ADC_ComputeMetricsRaw(adc_samples: np.ndarray, fs: float, resolution: int):
    N = len(adc_samples)
    adc_samples = adc_samples - np.mean(adc_samples)

    # Apply windowing
    window = np.blackman(N)
    windowed_signal = adc_samples * window

    # Compute Positive Frequency Spectrum (Raw Complex RFFT)
    fft_raw = np.fft.rfft(windowed_signal)
    fft_freqs = np.fft.rfftfreq(N, d=(1 / fs))

    # Window Correction Factors
    S1 = np.sum(window)
    S2 = np.sum(window**2)
    ENBW_bins = N * S2 / (S1**2)

    # Linear Magnitude Spectrum (Peak LSBs)
    fft_spectrum = np.abs(fft_raw) / S1
    fft_spectrum[1:-1] *= 2.0

    # Unified Bin Power Spectrum (LSB^2) using Coherent Gain
    # Each bin represents power scaled relative to a peak sine wave
    fft_bin_power = (np.abs(fft_raw) / S1) ** 2
    fft_bin_power[1:-1] *= 2.0  # Fold negative frequencies

    # Define window-specific mainlobe margin
    # Blackman window requires margin of 5 or 6 bins to capture skirts
    bin_margin = 25

    # Define DC Leakage Zone (Exclude from Noise)
    # DC is at bin 0, but Blackman window spreads it across 0 to bin_margin
    dc_bins = set(range(0, bin_margin + 1))

    # Identify Fundamental Frequency Bin
    f0_idx = np.argmax(fft_spectrum[1:]) + 1  # Peak search starting from bin 1
    f0_freq = fft_freqs[f0_idx]

    fund_bins = set(
        range(
            max(1, f0_idx - bin_margin), min(len(fft_spectrum), f0_idx + bin_margin + 1)
        )
    )

    # Fundamental Power
    # f0_pwr = np.sum(fft_bin_power[list(fund_bins)])
    f0_pwr = np.sum(fft_bin_power[list(fund_bins)]) / ENBW_bins

    # Extract Harmonics Power
    harmonic_bins = set()
    harmonic_freqs = []
    num_harmonics = 9
    for h in range(2, num_harmonics + 1):
        target_freq = (h * f0_freq) % fs
        if target_freq > (fs / 2):
            target_freq = fs - target_freq

        h_idx = np.argmin(np.abs(fft_freqs - target_freq))
        harmonic_freqs.append(fft_freqs[h_idx])

        # Capture harmonic peak skirts excluding overlap with fundamental/DC
        h_range = (
            set(
                range(
                    max(1, h_idx - bin_margin),
                    min(len(fft_spectrum), h_idx + bin_margin + 1),
                )
            )
            - fund_bins
            - dc_bins
        )

        harmonic_bins.update(h_range)
    harmonics_pwr = np.sum(fft_bin_power[list(harmonic_bins)]) / ENBW_bins

    # Pure Noise Bins: Exclude DC Zone, Fundamental Zone, and Harmonic Zones
    all_bins = set(range(0, len(fft_spectrum)))
    noise_bins = list(all_bins - dc_bins - fund_bins - harmonic_bins)

    # Calculate Corrected Noise Power
    avg_noise_power_per_bin = np.mean(fft_bin_power[noise_bins])
    noise_pwr = (avg_noise_power_per_bin * (N / 2)) / ENBW_bins

    noise_plus_distortion_pwr = noise_pwr + harmonics_pwr

    # Performance Metrics
    snr_db = 10 * np.log10(f0_pwr / noise_pwr)
    sinad_db = 10 * np.log10(f0_pwr / noise_plus_distortion_pwr)
    thd_db = 10 * np.log10(harmonics_pwr / f0_pwr)
    enob = (sinad_db - 1.76) / 6.02

    # Spectrum for Display in dBFS relative to full-scale continuous peak
    fft_spectrum_dbfs = 20 * np.log10(fft_spectrum / (2 ** (resolution - 1)))

    return {
        "f0": f0_freq,
        "f0_mag": fft_spectrum_dbfs[f0_idx],
        "harmonic_freqs": harmonic_freqs,
        "snr": snr_db,
        "sinad": sinad_db,
        "thd": thd_db,
        "enob": enob,
        "spectrum_freqs": fft_freqs,
        "spectrum_mag_dbfs": fft_spectrum_dbfs,
    }
