import numpy as np
import pyqtgraph as pg


class SignalPlotter:
    def __init__(self, title: str, size: tuple[int, int]) -> None:
        self.plot = pg.mkQApp(f"{title}")
        self.canvas = pg.GraphicsLayoutWidget(show=True)
        self.canvas.resize(size[0], size[1])
        self.plot_areas: list = []
        self.plot_waveforms: list = []
        self.text_display: list = []
        self.timer = None

    def add_plot(
        self,
        title: str,
        x_label: str,
        x_unit: str,
        x_data: np.ndarray | list,
        y_label: str,
        y_unit: str,
        y_data: np.ndarray | list,
        y_range: tuple[int | float, int | float],
        pen_color: str,
    ):
        p = self.canvas.addPlot(title=title)  # type: ignore
        p.setLabel("bottom", x_label, unit=x_unit)
        p.setLabel("left", y_label, unit=y_unit)
        p.showGrid(x=True, y=True, alpha=0.1)
        p.setRange(xRange=(np.min(x_data), np.max(x_data)), yRange=y_range, padding=0)
        waveform = p.plot(x_data, y_data, pen=pg.mkPen(pen_color, width=1.0))
        self.plot_areas.append(p)
        self.plot_waveforms.append(waveform)
        self.canvas.nextRow()  # type: ignore

    def update_plot(self, index: int, x_data: np.ndarray, y_data: np.ndarray):
        self.plot_waveforms[index].setData(x_data, y_data)

    def add_text(self, text: str, plot_index: int, pos: tuple[int, int], color: str):
        text_item = pg.TextItem(text=text, color=color)
        vb = self.plot_areas[plot_index].getViewBox()
        vb.addItem(text_item)
        text_item.setParentItem(vb)
        text_item.setPos(pos[0], pos[1])
        self.text_display.append(text_item)

    def update_text(self, index: int, text: str):
        self.text_display[index].setText(text)

    def setup_stream(self, plot_interval: float, callback_fn):
        self.timer = pg.QtCore.QTimer()
        self.timer.setInterval(int(plot_interval * 1000))
        self.timer.timeout.connect(callback_fn)

    def start_stream(self):
        if self.timer is not None:
            self.timer.start()

    def show(self):
        self.plot.exec()
