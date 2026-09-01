import numpy as np

from .utilities import sinefit_4param


def ADC_ComputeMetricsRaw(
    adc_samples: np.ndarray,
    fs: float,
    resolution: int,
    num_harmonics: int = 9,
    use_sinefit: bool = False,
):
    N = len(adc_samples)
    adc_samples = adc_samples - np.mean(adc_samples)

    # --- Signal level in dBFS
    # Identify raw local peak candidate indices
    raw_peaks = (
        np.where(
            (adc_samples[1:-1] > adc_samples[:-2])
            & (adc_samples[1:-1] > adc_samples[2:])
        )[0]
        + 1
    )

    # --- Optional
    # # Filter peaks: keep only those exceeding 95% of theoretical peak value
    # rms = np.sqrt(np.mean(adc_samples**2))
    # pk_avg = rms * np.sqrt(2)
    # threshold = 0.95 * pk_avg
    # valid_peak_indices = raw_peaks[adc_samples[raw_peaks] > threshold]

    # # Compute true average peak
    # signal_pk_avg = np.mean(adc_samples[valid_peak_indices])

    # # Compute peak and average signal level in dbFS
    # signal_lvl_avg_dbfs = 20 * np.log10(signal_pk_avg / (2 ** (resolution - 1)))
    # signal_lvl_pk_dbfs = 20 * np.log10(np.max(adc_samples) / (2 ** (resolution - 1)))

    # --- Apply windowing
    window = np.blackman(N)
    windowed_signal = adc_samples * window

    # --- Compute Positive Frequency Spectrum (Raw Complex RFFT)
    fft_raw = np.fft.rfft(windowed_signal)
    fft_freqs = np.fft.rfftfreq(N, d=(1 / fs))

    # --- Window Correction Factors
    S1 = np.sum(window)
    S2 = np.sum(window**2)
    ENBW_bins = N * S2 / (S1**2)

    # --- Linear Magnitude Spectrum (Peak LSBs)
    fft_spectrum = np.abs(fft_raw) / S1
    fft_spectrum[1:-1] *= 2.0

    # --- Unified Bin Power Spectrum (LSB^2) using Coherent Gain
    # Each bin represents power scaled relative to a peak sine wave
    fft_bin_power = (np.abs(fft_raw) / S1) ** 2
    fft_bin_power[1:-1] *= 2.0  # Fold negative frequencies

    # --- Identify Fundamental Frequency Bin
    f0_init_idx = np.argmax(fft_spectrum[1:]) + 1  # Peak search starting from bin 1
    f0_freq = fft_freqs[f0_init_idx]

    # --- Compute magnitude dBFS and fundamental power using sinefit values
    if use_sinefit:
        f0_freq, f0_amplitude, f0_phase, dc_signal = sinefit_4param(
            adc_samples=adc_samples, fs=fs, f0_initial=f0_freq
        )
        f0_magnitude_dbfs = 20 * np.log10(f0_amplitude / (2 ** (resolution - 1)))
        f0_pwr = (f0_amplitude**2) / 2.0

    # --- Define window-specific mainlobe margin
    # Blackman window requires margin of 5 or 6 bins to capture skirts - increase to 25
    bin_margin = 25

    # --- Define DC Leakage Zone (Exclude from Noise)
    # DC is at bin 0, but Blackman window spreads it across 0 to bin_margin
    dc_bins = set(range(0, bin_margin + 1))

    # --- Define Fundamental Bins
    fund_bins = set(
        range(
            max(1, f0_init_idx - bin_margin),
            min(len(fft_spectrum), f0_init_idx + bin_margin + 1),
        )
    )

    # --- Compute magnitude dBFS and fundamental power using fft spectrum & fundamental bins
    if not use_sinefit:
        f0_pwr = np.sum(fft_bin_power[list(fund_bins)]) / ENBW_bins
        f0_amplitude = np.sqrt(2 * f0_pwr)
        f0_magnitude_dbfs = 20 * np.log10(f0_amplitude / (2 ** (resolution - 1)))

    # --- Extract & Compute Harmonics Power
    harmonic_bins = set()
    harmonic_freqs = []
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

    # --- Pure Noise Bins: Exclude DC Zone, Fundamental Zone, and Harmonic Zones
    all_bins = set(range(0, len(fft_spectrum)))
    noise_bins = list(all_bins - dc_bins - fund_bins - harmonic_bins)

    # --- Calculate Corrected Noise Power
    avg_noise_power_per_bin = np.mean(fft_bin_power[noise_bins])
    noise_pwr = (avg_noise_power_per_bin * (N / 2)) / ENBW_bins

    noise_plus_distortion_pwr = noise_pwr + harmonics_pwr

    # --- Finally, compute performance metrics
    snr_db = 10 * np.log10(f0_pwr / noise_pwr)
    sinad_db = 10 * np.log10(f0_pwr / noise_plus_distortion_pwr)
    thd_db = 10 * np.log10(harmonics_pwr / f0_pwr)
    enob = (sinad_db - 1.76) / 6.02

    # --- Spectrum for Display in dBFS relative to full-scale continuous peak
    fft_spectrum_dbfs = 20 * np.log10(fft_spectrum / (2 ** (resolution - 1)))
    # f0_magnitude_dbfs = np.log10(np.sum(10 ** fft_spectrum_dbfs[list(fund_bins)]))

    return {
        "f0": f0_freq,
        "f0_mag": f0_magnitude_dbfs,
        "harmonic_freqs": harmonic_freqs,
        "snr": snr_db,
        "sinad": sinad_db,
        "thd": thd_db,
        "enob": enob,
        "spectrum_freqs": fft_freqs,
        "spectrum_mag_dbfs": fft_spectrum_dbfs,
    }


def ADC_ComputeDynamicRange(adc_samples: np.ndarray, resolution: int) -> float:
    adc_samples_ac = adc_samples - np.mean(adc_samples)
    noise_rms = np.sqrt(np.mean(adc_samples_ac**2))
    full_scale_rms = (2**resolution) / (2 * np.sqrt(2))
    dynami_range_db = 20 * np.log10(full_scale_rms / noise_rms)
    return dynami_range_db
