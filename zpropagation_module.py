from PyQt5.QtWidgets import QWidget, QVBoxLayout
from image_widget import ImageWidget
import numpy as np


class ZPropagationModule(QWidget):
    """GUI for Z-propagation with vertical slices"""
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.slice1 = ImageWidget(np.zeros((128, 128)), title="Slice 1")
        self.slice2 = ImageWidget(np.zeros((128, 128)), title="Slice 2")
        layout.addWidget(self.slice1)
        layout.addWidget(self.slice2)
