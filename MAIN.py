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

        msg = QLabel(
            "Welcome to the Holography Toolbox\n\n"
            "Select a module above to begin your analysis \n\n\n"
            "Written by Fredrik Skärberg @ Department of Physics, Gothenburg University"
        )
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

        # Create top-level horizontal layout
        nav_bar = QHBoxLayout()
        nav_bar.setAlignment(Qt.AlignCenter)

        # Configure buttons first
        for btn in (self.btn_recon, self.btn_track):
            btn.setMinimumHeight(60)
            btn.setMinimumWidth(200)
            btn.setCheckable(True)
            btn.setStyleSheet(self._style_inactive())

            font = btn.font()
            font.setPointSize(12)
            btn.setFont(font)

        # Then add buttons to the layout
        nav_bar.addWidget(self.btn_recon)
        nav_bar.addWidget(self.btn_track)

        # Add the nav_bar to the main layout
        main_layout.addLayout(nav_bar)

        # Stacked widget for welcome & modules
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack, stretch=1)

        # Welcome screen
        self.welcome = WelcomeWidget(asset_path="Utils/yeast_template.png")
        self.stack.addWidget(self.welcome)

        # Placeholders for modules (created when needed)
        self.recon_module = None
        self.tracking_module = None

        # Connect buttons to toggle modules
        self.btn_recon.clicked.connect(lambda: self._toggle_module("recon", self.btn_recon))
        self.btn_track.clicked.connect(lambda: self._toggle_module("track", self.btn_track))

        # Nav buttons tracking
        self.nav_buttons = [self.btn_recon, self.btn_track]
        self.active_button = None
        self.active_module = None

        # Start with welcome screen
        self.stack.setCurrentWidget(self.welcome)
        self._set_active_button(None)

    # ------------ Helpers ------------
    def _toggle_module(self, module_name, button):
        """Show module, cleanup previous one if switching, toggle back to welcome if same pressed"""
        # If same button pressed again → go back to welcome
        if self.active_button is button:
            self._cleanup_active_module()
            self.stack.setCurrentWidget(self.welcome)
            self._set_active_button(None)
            return

        # Switching to another module → cleanup current
        self._cleanup_active_module()

        # Create module if not already made
        if module_name == "recon":
            self.recon_module = ReconstructionModule()
            widget = self.recon_module
        elif module_name == "track":
            self.tracking_module = ParticleTrackingModule()
            widget = self.tracking_module
        else:
            widget = self.welcome

        self.stack.addWidget(widget)
        self.stack.setCurrentWidget(widget)
        self._set_active_button(button)
        self.active_module = widget

    def _cleanup_active_module(self):
        """Release memory from active module if it defines cleanup()"""
        if self.active_module:
            if hasattr(self.active_module, "cleanup"):
                self.active_module.cleanup()
            self.stack.removeWidget(self.active_module)
            self.active_module.deleteLater()
            self.active_module = None

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

    screen = app.primaryScreen()
    size = screen.availableGeometry().size()
    window.resize(min(1400, size.width()), min(900, size.height()))
    #window.setMaximumSize(size.width(), size.height())

    window.show()
    sys.exit(app.exec_())


