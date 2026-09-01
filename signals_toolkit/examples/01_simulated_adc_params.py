import numpy as np
import pyqtgraph as pg

from signals_toolkit import ADC_ComputeMetricsRaw, SignalPlotter, simulated_adc


def main():
    fs = 1000000  # 1Msps
    num_samples = 8192  # 65536
    adc_resolution = 16
    target_snr = 75

    adc_data = simulated_adc(
        fundamental_frequency=10000,
        fs=fs,
        sample_size=num_samples,
        resolution=adc_resolution,
        signal_amplitude_dbfs=-0.5,
        target_snr_db=target_snr,
        harmonics_dbc=[-120, -135],
        signed=False,
    )

    t_span = (1 / fs) * float(num_samples)
    t = np.linspace(0, t_span, num_samples, endpoint=False)

    result = ADC_ComputeMetricsRaw(
        adc_samples=adc_data, fs=fs, resolution=adc_resolution, use_sinefit=False
    )

    print(f"f0: {result['f0']:.2f}")
    print(f"f0 Magnitude: {result['f0_mag']:.4f} dbFS")
    print(f"SNR: {result['snr']:.2f} dB")
    print(f"SINAD: {result['sinad']:.2f} dB")
    print(f"THD: {result['thd']:.2f} dB")
    print(f"ENOB: {result['enob']:.2f} bits")

    # Plot results of final iteration
    plot = SignalPlotter(title="ADC Plots", size=(1920, 1080))
    plot.add_plot(
        title="ADC Codes",
        x_label="Time",
        x_unit="s",
        x_data=t,
        y_label="Code",
        y_unit="",
        y_data=adc_data,
        y_range=(0, 2**adc_resolution),
        pen_color="#AAAA00",
    )
    plot.add_plot(
        title="FFT Spectrum",
        x_label="Frequency",
        x_unit="Hz",
        x_data=result["spectrum_freqs"],
        y_label="Level",
        y_unit="dBFS",
        y_data=result["spectrum_mag_dbfs"],
        y_range=(-150, 0),
        pen_color="#AAAA00",
    )
    plot.show()


if __name__ == "__main__":
    main()
