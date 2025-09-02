# particle_tracking_module.py
import sys
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGridLayout
)
from PyQt5.QtCore import Qt

class ParticleTrackingModule(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Particle Tracking Module")
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Info label
        self.info_label = QLabel("Load a folder of frames to start tracking")
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)

        # Buttons for loading frames and starting tracking
        btn_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Frames")
        self.track_button = QPushButton("Start Tracking")
        btn_layout.addWidget(self.load_button)
        btn_layout.addWidget(self.track_button)
        layout.addLayout(btn_layout)

        # Image display area (for example: big frame + small zoomed-in subframes)
        self.image_layout = QGridLayout()
        layout.addLayout(self.image_layout)

        # Connect buttons to dummy functions (replace with real logic)
        self.load_button.clicked.connect(self.load_frames)
        self.track_button.clicked.connect(self.start_tracking)

    # ---------------- Dummy methods ----------------
    def load_frames(self):
        self.info_label.setText("Frames loaded (replace with actual loading code)")

    def start_tracking(self):
        self.info_label.setText("Tracking started (replace with actual tracking code)")
