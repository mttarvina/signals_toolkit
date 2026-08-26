import numpy as np
import pyqtgraph as pg

from signals_toolkit import ADC_ComputeMetricsRaw, simulated_adc


def main():
    fs = 1000000  # 1Msps
    num_samples = 8192  # 65536
    adc_resolution = 16
    target_snr = 90

    adc_codes = simulated_adc(
        fundamental_frequency=15000,
        fs=fs,
        sample_size=num_samples,
        resolution=adc_resolution,
        signal_amplitude_dbfs=-0.5,
        target_snr_db=target_snr,
        harmonics_dbc=[-100, -110, -120],
        signed=False,
    )

    t_span = (1 / fs) * float(num_samples)
    t = np.linspace(0, t_span, num_samples, endpoint=False)

    result = ADC_ComputeMetricsRaw(
        adc_samples=adc_codes, fs=fs, resolution=adc_resolution
    )

    print(f"f0: {result['f0']:.2f}")
    print(f"f0_mag: {result['f0_mag']:.4f} dbFS")
    # print(f"harmonics: {result['harmonic_freqs']}")
    print(f"snr: {result['snr']:.2f} dB")
    print(f"sinad: {result['sinad']:.2f} dB")
    print(f"thd: {result['thd']:.2f} dB")
    print(f"enob: {result['enob']:.2f} bits")

    # Plot results
    plot = pg.mkQApp("Signal Plots")
    canvas = pg.GraphicsLayoutWidget(show=True)
    canvas.resize(1920, 1280)

    p1 = canvas.addPlot(title="ADC Codes")  # type: ignore
    p1.setLabel("bottom", "Time", unit="s")
    p1.setLabel("left", "Codes", unit="LSB")
    p1.showGrid(x=True, y=True, alpha=0.1)
    p1.plot(t, adc_codes, pen=pg.mkPen("#FFAA00", width=1.0))

    canvas.nextRow()  # type: ignore

    p2 = canvas.addPlot(title="FFT Spectrum (dBFS)")  # type: ignore
    p2.setLabel("bottom", "Frequency", unit="Hz")
    p2.setLabel("left", "Magnitude", unit="dBFS")
    p2.showGrid(x=True, y=True, alpha=0.1)
    p2.plot(
        result["spectrum_freqs"],
        result["spectrum_mag_dbfs"],
        pen=pg.mkPen("#00AAFF", width=1.0),
    )
    plot.exec()


if __name__ == "__main__":
    main()
