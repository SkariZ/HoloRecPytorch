# particle_tracking_module.py
import os
import glob
import numpy as np
import sys
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
                             QLineEdit, QPushButton, QLabel, QFileDialog, QComboBox, QSlider)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

sys.path.append('code')
import image_utils as iu  # optional, if you want cropping, etc.

class ImageWidget(QWidget):
    def __init__(self, data, title=""):
        super().__init__()
        self.data = data
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
        self.ax.set_title(self.title)

    def update_plot(self):
        if self.colorbar:
            self.colorbar.remove()
            self.colorbar = None
        self.ax.clear()
        im = self.ax.imshow(self.data, cmap=self.cmap, interpolation='none', aspect='auto')
        self.colorbar = self.ax.figure.colorbar(im, ax=self.ax)
        self.ax.set_title(self.title)
        self.canvas.draw()

    def update_data(self, data, title=""):
        self.data = data
        self.title = title
        self.update_plot()


class ParticleTrackingModule(QWidget):
    def __init__(self):
        super().__init__()
        self.frames = []
        self.current_frame_idx = 0
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Particle Tracking")
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # ---------------- Left panel: parameters ----------------
        param_layout = QVBoxLayout()

        # Frame loading
        self.folder_input = QLineEdit("")
        browse_btn = QPushButton("Browse Folder")
        browse_btn.clicked.connect(self.select_folder)
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(self.folder_input)
        folder_layout.addWidget(browse_btn)
        param_layout.addLayout(folder_layout)

        self.load_btn = QPushButton("Load Frames")
        self.load_btn.clicked.connect(self.load_frames)
        param_layout.addWidget(self.load_btn)

        # Preprocessing options
        self.preproc_combo = QComboBox()
        self.preproc_combo.addItems(["None", "Gaussian Blur", "Background Subtraction"])
        param_layout.addWidget(QLabel("Preprocessing:"))
        param_layout.addWidget(self.preproc_combo)

        self.update_preview_btn = QPushButton("Update Preview")
        self.update_preview_btn.clicked.connect(self.update_preview)
        param_layout.addWidget(self.update_preview_btn)

        # Frame slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.slider_update)
        param_layout.addWidget(QLabel("Frame:"))
        param_layout.addWidget(self.frame_slider)

        param_layout.addStretch(1)
        main_layout.addLayout(param_layout)

        # ---------------- Right panel: image ----------------
        self.image_widget = ImageWidget(np.zeros((100, 100)), title="Frame Preview")
        main_layout.addWidget(self.image_widget)

        self.show()

    # ---------------- Handlers ----------------
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folder_input.setText(folder)

    def load_frames(self):
        folder = self.folder_input.text()
        if not os.path.exists(folder):
            return
        imgs = sorted(glob.glob(os.path.join(folder, "*.png")))
        if not imgs:
            return
        # Load frames as grayscale float32
        self.frames = [plt.imread(f).astype(np.float32) for f in imgs]
        self.current_frame_idx = 0
        self.frame_slider.setMaximum(len(self.frames)-1)
        self.frame_slider.setEnabled(True)
        self.image_widget.update_data(self.frames[0], title=f"Frame {0}")

    def update_preview(self):
        if not self.frames:
            return
        frame = self.frames[self.current_frame_idx]
        option = self.preproc_combo.currentText()
        processed = frame.copy()
        if option == "Gaussian Blur":
            from scipy.ndimage import gaussian_filter
            processed = gaussian_filter(processed, sigma=1)
        elif option == "Background Subtraction":
            processed = processed - np.mean(processed)
        self.image_widget.update_data(processed, title=f"Frame {self.current_frame_idx} ({option})")

    def slider_update(self):
        if not self.frames:
            return
        self.current_frame_idx = self.frame_slider.value()
        self.update_preview()

