# MAIN.py
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QPushButton, QLabel
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

# Import your modules
from reconstruction_module import ReconstructionModule
from particle_tracking_module import ParticleTrackingModule
from zpropagation_module import ZPropagationModule


class WelcomeWidget(QWidget):
    """Front page / welcome screen"""
    def __init__(self, asset_path: str = None):
        super().__init__()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        if asset_path:
            pixmap = QPixmap(asset_path)
            img_label = QLabel()
            img_label.setPixmap(pixmap.scaledToWidth(600, Qt.SmoothTransformation))
            img_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(img_label)

        msg = QLabel("Welcome to the Holography Toolbox\n\nSelect a module above to begin your analysis \n\n\nWritten by Fredrik Skärberg @ Department of Physics, Gothenburg University")
        msg.setAlignment(Qt.AlignCenter)
        msg.setStyleSheet("font-size: 16px; color: #333;")
        layout.addWidget(msg)

        self.setLayout(layout)


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Holography Toolbox")
        self.resize(1400, 900)

        # Central widget & layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        self.btn_recon = QPushButton("Reconstruction")
        self.btn_track = QPushButton("Particle Tracking")
        self.btn_zprop = QPushButton("Z-Propagation")

        # Create top-level horizontal layout
        nav_bar = QHBoxLayout()
        nav_bar.setAlignment(Qt.AlignCenter)

        # Configure buttons first
        for btn in (self.btn_recon, self.btn_track, self.btn_zprop):
            btn.setMinimumHeight(60)          # taller buttons
            btn.setMinimumWidth(200)          # wider buttons
            btn.setCheckable(True)
            btn.setStyleSheet(self._style_inactive())
            
            font = btn.font()
            font.setPointSize(12)
            btn.setFont(font)

        # Then add buttons to the layout (only once)
        nav_bar.addWidget(self.btn_recon)
        nav_bar.addWidget(self.btn_track)
        nav_bar.addWidget(self.btn_zprop)

        # Add the nav_bar to the main layout
        main_layout.addLayout(nav_bar)

        # Stacked widget for welcome & modules
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack, stretch=1)

        # Welcome screen
        self.welcome = WelcomeWidget(asset_path="Utils/yeast_template.png")
        self.stack.addWidget(self.welcome)

        # Create module instances
        self.recon_module = ReconstructionModule()        # fully working module
        self.tracking_module = ParticleTrackingModule()   # dummy GUI for tracking
        self.zprop_module = ZPropagationModule()          # dummy GUI for z-propagation

        # Add modules to stack
        self.stack.addWidget(self.recon_module)
        self.stack.addWidget(self.tracking_module)
        self.stack.addWidget(self.zprop_module)

        # Connect buttons to toggle modules
        self.btn_recon.clicked.connect(lambda: self._toggle_module(self.recon_module, self.btn_recon))
        self.btn_track.clicked.connect(lambda: self._toggle_module(self.tracking_module, self.btn_track))
        self.btn_zprop.clicked.connect(lambda: self._toggle_module(self.zprop_module, self.btn_zprop))

        # Nav buttons tracking
        self.nav_buttons = [self.btn_recon, self.btn_track, self.btn_zprop]
        self.active_button = None

        # Start with welcome screen
        self.stack.setCurrentWidget(self.welcome)
        self._set_active_button(None)

    # ------------ Helpers ------------
    def _toggle_module(self, widget, button):
        """Show widget if not active; if already active -> show welcome."""
        if self.active_button is button:
            self.stack.setCurrentWidget(self.welcome)
            self._set_active_button(None)
            return
        self.stack.setCurrentWidget(widget)
        self._set_active_button(button)

    def _set_active_button(self, button):
        """Update styles and checked state for nav buttons."""
        for b in self.nav_buttons:
            if b is button:
                b.setChecked(True)
                b.setStyleSheet(self._style_active())
            else:
                b.setChecked(False)
                b.setStyleSheet(self._style_inactive())
        self.active_button = button

    def _style_active(self):
        return (
            "background-color: #0078D7;"
            "color: white;"
            "font-weight: bold;"
            "border-radius: 4px;"
        )

    def _style_inactive(self):
        return (
            "background-color: none;"
            "color: black;"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())

