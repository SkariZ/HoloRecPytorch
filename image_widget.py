import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class ImageWidget(QWidget):
    """Reusable widget for displaying an image or histogram"""
    def __init__(self, data=None, is_histogram=False, title=""):
        super().__init__()
        self.data = data if data is not None else np.zeros((10,10))
        self.is_histogram = is_histogram
        self.title = title
        self.colorbar = None
        self.cmap = 'gray'
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.canvas = FigureCanvas(plt.Figure())
        layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.add_subplot(111)
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        if self.is_histogram:
            self.ax.hist(self.data.flatten(), bins=255, color='darkblue', alpha=0.7)
        else:
            im = self.ax.imshow(self.data, cmap=self.cmap, interpolation='none', aspect='auto')
            if self.colorbar:
                self.colorbar.remove()
            self.colorbar = self.ax.figure.colorbar(im, ax=self.ax)
        self.ax.set_title(self.title)
        self.canvas.draw()

    def update_data(self, data, is_histogram=False, title=""):
        self.data = data
        self.is_histogram = is_histogram
        self.title = title
        self.update_plot()
