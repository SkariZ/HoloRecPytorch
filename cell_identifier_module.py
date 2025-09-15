# cell_identifier_module_clean_fixed.py
import os
import sys
import time
import numpy as np
import random
import torch

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel,
    QFileDialog, QSpinBox, QCheckBox, QComboBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import tifffile

sys.path.append('code')
import fft_loader as fl
import propagation as prop

import cfg as CFG

class ImageWidget(QWidget):
    def __init__(self, data, title=""):
        super().__init__()
        self.data_original = data.copy()   # true uncropped frame
        self.data_display = data.copy()    # currently displayed
        self.fov_coords = None             # crop rectangle
        self.title = title
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.canvas = FigureCanvas(plt.Figure())
        layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.add_subplot(111)
        self.update_plot()
        self.selector = RectangleSelector(
            self.ax,
            onselect=self.onselect,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True
        )

    def onselect(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        self.fov_coords = (min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2))
        self.update_plot()
        print(f"Crop rectangle set: {self.fov_coords}")

    def update_plot(self):
        self.ax.clear()
        if self.fov_coords:
            x1, y1, x2, y2 = self.fov_coords
            self.data_display = self.data_original[y1:y2, x1:x2]
        else:
            self.data_display = self.data_original
        self.ax.imshow(self.data_display, cmap='gray')
        self.ax.set_title(self.title)
        self.canvas.draw()

    def update_data(self, data, title=""):
        self.data_original = data.copy()  # reset original
        self.title = title
        self.fov_coords = None            # clear crop
        self.update_plot()

    def reset_crop(self):
        self.fov_coords = None
        self.update_plot()


class CellIdentifierModule(QWidget):
    def __init__(self):
        super().__init__()
        self.full_data = None          # memmap or np.array of all frames
        self.preview_frame = None
        self.fov_coords = None
        self.focused_frames = None
        self.save_folder = None   # will be set when load_frame is used
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Cell Identifier")
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # ---------------- Left panel: controls ----------------
        param_layout = QVBoxLayout()
        
        # ---------------- FFT Settings ----------------
        self.fft_checkbox = QCheckBox("Apply FFT")
        param_layout.addWidget(self.fft_checkbox)

        param_layout.addWidget(QLabel("FFT Filter Radius:"))
        self.fft_radius_input = QLineEdit()
        self.fft_radius_input.setPlaceholderText("e.g., 200 (leave empty for no filter)")
        param_layout.addWidget(self.fft_radius_input)
        # Set default value
        self.fft_radius_input.setText(str(CFG.cell_id_params['fft_radius']))

        param_layout.addWidget(QLabel("Original Image Size (HxW):"))
        size_layout = QHBoxLayout()
        self.height_input = QLineEdit()
        self.height_input.setPlaceholderText("Height")
        self.height_input.setText(str(CFG.cell_id_params['orig_height']))
        self.width_input = QLineEdit()
        self.width_input.setPlaceholderText("Width")
        self.width_input.setText(str(CFG.cell_id_params['orig_width']))
        size_layout.addWidget(self.height_input)
        size_layout.addWidget(self.width_input)
        param_layout.addLayout(size_layout)

        param_layout.addWidget(QLabel("FFT Mask Shape:"))
        self.mask_shape_combo = QComboBox()
        self.mask_shape_combo.addItems(["ellipse", "circle"])
        param_layout.addWidget(self.mask_shape_combo)

        # ---------------- Load Data Button ----------------
        self.load_btn = QPushButton("Load Frames / Data")
        self.load_btn.clicked.connect(self.load_frame)
        param_layout.addWidget(self.load_btn)

        # Frame index selection
        param_layout.addWidget(QLabel("Frame index for preview:"))
        self.frame_index_spin = QSpinBox()
        self.frame_index_spin.setMinimum(0)
        self.frame_index_spin.valueChanged.connect(self.update_preview)
        param_layout.addWidget(self.frame_index_spin)

        param_layout.addWidget(QLabel("Channel for preview:"))
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["real", "imag", "abs"])
        self.channel_combo.setCurrentText("imag")  # <- pre-select "imag"
        self.channel_combo.currentTextChanged.connect(self.update_preview)
        param_layout.addWidget(self.channel_combo)

        # Crop buttons
        self.crop_btn = QPushButton("Apply Crop")
        self.crop_btn.clicked.connect(self.apply_crop)
        param_layout.addWidget(self.crop_btn)

        self.reset_crop_btn = QPushButton("Reset Crop")
        self.reset_crop_btn.clicked.connect(self.reset_crop)
        param_layout.addWidget(self.reset_crop_btn)

        # Add some spacing
        param_layout.addSpacing(70)


        # --- Save folder selector ---
        param_layout.addWidget(QLabel("Save Folder:"))
        folder_layout = QHBoxLayout()
        self.save_folder_input = QLineEdit()
        self.save_folder_input.setPlaceholderText("Select folder for saving results")
        folder_btn = QPushButton("Browse")
        folder_btn.clicked.connect(self.select_save_folder)
        folder_layout.addWidget(self.save_folder_input)
        folder_layout.addWidget(folder_btn)
        param_layout.addLayout(folder_layout)

        # ---------------- Z-propagation / focus settings ----------------
        param_layout.addWidget(QLabel("Z-Propagation Settings:"))

        # z_min
        z_min_layout = QHBoxLayout()
        z_min_layout.addWidget(QLabel("z_min (µm):"))
        self.z_min_input = QLineEdit(str(CFG.zprop_defaults["z_min"]))
        z_min_layout.addWidget(self.z_min_input)
        param_layout.addLayout(z_min_layout)

        # z_max
        z_max_layout = QHBoxLayout()
        z_max_layout.addWidget(QLabel("z_max (µm):"))
        self.z_max_input = QLineEdit(str(CFG.zprop_defaults["z_max"]))
        z_max_layout.addWidget(self.z_max_input)
        param_layout.addLayout(z_max_layout)

        # z_steps
        z_steps_layout = QHBoxLayout()
        z_steps_layout.addWidget(QLabel("z_steps:"))
        self.z_steps_input = QLineEdit(str(CFG.zprop_defaults["z_steps"]))
        z_steps_layout.addWidget(self.z_steps_input)
        param_layout.addLayout(z_steps_layout)

        # wavelength
        wl_layout = QHBoxLayout()
        wl_layout.addWidget(QLabel("Wavelength (µm):"))
        self.wavelength_input = QLineEdit(str(CFG.zprop_defaults["wavelength"]))
        wl_layout.addWidget(self.wavelength_input)
        param_layout.addLayout(wl_layout)

        # Return unprocessed checkbox
        self.return_unprocessed_checkbox = QCheckBox("Return unprocessed best focus plane")
        param_layout.addWidget(self.return_unprocessed_checkbox)
        self.return_unprocessed_checkbox.setChecked(False)
        self.return_unprocessed_checkbox.setChecked(CFG.zprop_defaults.get('return_unprocessed', False))

        # Method name
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.method_input = QLineEdit(CFG.zprop_defaults.get("method", "default"))
        method_layout.addWidget(self.method_input)
        param_layout.addLayout(method_layout)

        # Button for running focus finding + propagation on ALL frames
        self.focus_all_btn = QPushButton("Run Focus Search & Propagate (All Frames) and Save")
        self.focus_all_btn.clicked.connect(self.run_focus_search_all)
        param_layout.addWidget(self.focus_all_btn)
        param_layout.addSpacing(20)

        # --- Number of random TIFF frames ---
        param_layout.addWidget(QLabel("Number of random TIFF frames to save:"))
        self.num_tiffs_spin = QSpinBox()
        self.num_tiffs_spin.setRange(1, 9999)      # sensible upper bound
        self.num_tiffs_spin.setValue(5)            # default matches old hardcoded value
        param_layout.addWidget(self.num_tiffs_spin)

        # Saving channels
        param_layout.addWidget(QLabel("Channels to save in TIFF:"))
        self.save_real_checkbox = QCheckBox("Real part")
        self.save_real_checkbox.setChecked(True)
        param_layout.addWidget(self.save_real_checkbox)

        self.save_imag_checkbox = QCheckBox("Imag part")
        self.save_imag_checkbox.setChecked(True)
        param_layout.addWidget(self.save_imag_checkbox)

        self.save_abs_checkbox = QCheckBox("Absolute value")
        self.save_abs_checkbox.setChecked(True)
        param_layout.addWidget(self.save_abs_checkbox)

        self.save_tiff_btn = QPushButton("Save Random Frames (.tiff)")
        self.save_tiff_btn.clicked.connect(self.save_random_tiffs)
        param_layout.addWidget(self.save_tiff_btn)

        # After all the buttons
        param_layout.addStretch(1)
        main_layout.addLayout(param_layout, 1)

        # ---------------- Right panel: image ----------------
        self.image_widget = ImageWidget(np.zeros((128, 128)), title="Preview")
        main_layout.addWidget(self.image_widget, 3)  # image panel stretch factor 3

        # Status label at the very bottom
        self.status_label = QLabel("Ready.")
        param_layout.addWidget(self.status_label)

        self.show()

    # ---------------- Handlers ----------------
    def load_frame(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Numpy File", "", "Numpy Files (*.npy)")
        if not fname:
            return

        # Keep dataset as memmap
        self.full_data = np.load(fname, mmap_mode='r')
        self.frame_index_spin.setMaximum(self.full_data.shape[0]-1)

        # Update status label
        self.status_label.setText(f"Loaded data: {fname} with {self.full_data.shape[0]} frames")

        # Folder where the file is located
        file_folder = os.path.dirname(fname)

        # Go up one level
        parent_folder = os.path.dirname(file_folder)

        # Create "cell_analysis"
        self.save_folder = os.path.join(parent_folder, "cell_analysis")
        os.makedirs(self.save_folder, exist_ok=True)

        # Update GUI input
        self.save_folder_input.setText(self.save_folder)

        # Load only the selected frame
        idx = self.frame_index_spin.value()
        frame = self.full_data[idx]

        # Apply FFT if requested
        if self.fft_checkbox.isChecked():
            orig_size = self.get_original_image_size()
            pupil_radius = self.get_fft_radius()
            mask_shape = self.mask_shape_combo.currentText()

            if orig_size is None:
                print("Invalid original size, skipping FFT")
            else:
                if pupil_radius is None:
                    pupil_radius = max(orig_size) // 2  # default: full mask

                frame_complex = fl.vec_to_field(
                    frame,
                    pupil_radius=pupil_radius,
                    shape=orig_size,
                    mask_shape=mask_shape
                )
                frame = frame_complex.imag
                # Frame is a tensor, convert to numpy
                frame = frame.cpu().numpy()
        elif not self.fft_checkbox.isChecked() and frame.ndim == 1:
            # Print warning that frame probably needs FFT- print to qt label
            self.status_label.setText("Warning: Loaded frame is 1D, consider applying FFT")
            return
        else:
            frame = frame.astype(np.float32)

        self.preview_frame = frame.copy()
        self.image_widget.update_data(frame, title=f"Frame {idx}")

    def select_save_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Folder", self.save_folder or "")
        if folder:
            self.save_folder = folder
            self.save_folder_input.setText(folder)

    def get_fft_radius(self):
        text = self.fft_radius_input.text().strip()
        if text == "":
            return None  # no filter if empty
        try:
            radius = int(text)
            if radius < 0:
                raise ValueError
            return radius
        except ValueError:
            print("Invalid FFT radius input. Using no filter.")
            return None
    
    def get_original_image_size(self):
        try:
            h = int(self.height_input.text().strip())
            w = int(self.width_input.text().strip())
            return (h, w)
        except ValueError:
            print("Invalid image size input. Using current frame size.")
            if self.preview_frame is not None:
                return self.preview_frame.shape
            else:
                return None

    def update_preview(self):
        if self.full_data is None:
            return
        idx = self.frame_index_spin.value()
        frame = self.full_data[idx]  # still memmap slice

        if self.fft_checkbox.isChecked():
            orig_size = self.get_original_image_size()
            pupil_radius = self.get_fft_radius()
            mask_shape = self.mask_shape_combo.currentText()

            if orig_size is None:
                print("Invalid original size, skipping FFT")
            else:
                frame_complex = fl.vec_to_field(
                    frame,
                    pupil_radius=pupil_radius,
                    shape=orig_size,
                    mask_shape=mask_shape
                )
                frame_display = frame_complex
                # Frame is a tensor, convert to numpy
                frame_display = frame_display.cpu().numpy()
        elif not self.fft_checkbox.isChecked() and frame.ndim == 1:
            # Print warning that frame probably needs FFT- print to qt label
            self.status_label.setText("Warning: Loaded frame is 1D, consider applying FFT")
            return
        else:
            frame_display = frame.astype(np.complex64)

        # Select channel for display
        channel = self.channel_combo.currentText()
        if channel == "real":
            frame_display = np.real(frame_display)
        elif channel == "imag":
            frame_display = np.imag(frame_display)
        elif channel == "abs":
            frame_display = np.abs(frame_display)

        self.preview_frame = frame_display.copy()
        self.image_widget.update_data(frame_display, title=f"Frame {idx}")

    def apply_crop(self):
        if not self.image_widget.fov_coords:
            print("Select a crop rectangle first")
            return
        # just store fov_coords, do NOT overwrite original frame
        self.fov_coords = self.image_widget.fov_coords
        self.image_widget.update_plot()
        
    def reset_crop(self):
        if self.preview_frame is not None:
            self.fov_coords = None
            # Restore the original preview frame
            self.image_widget.update_data(self.preview_frame,
                                        title=f"Frame {self.frame_index_spin.value()}")

    def get_zprop_settings(self):
        try:
            z_min = float(self.z_min_input.text().strip())
        except ValueError:
            z_min = CFG.zprop_defaults["z_min"]

        try:
            z_max = float(self.z_max_input.text().strip())
        except ValueError:
            z_max = CFG.zprop_defaults["z_max"]

        try:
            z_steps = int(self.z_steps_input.text().strip())
        except ValueError:
            z_steps = CFG.zprop_defaults["z_steps"]

        try:
            wavelength = float(self.wavelength_input.text().strip())
        except ValueError:
            wavelength = CFG.zprop_defaults["wavelength"]

        method = self.method_input.text().strip() or CFG.zprop_defaults["method"]

        return_unprocessed = self.return_unprocessed_checkbox.isChecked()

        return {
            "z_min": z_min,
            "z_max": z_max,
            "z_steps": z_steps,
            "wavelength": wavelength,
            "method": method,
            "return_unprocessed": return_unprocessed
        }

    def run_focus_search_all(self):
        """
        Run focus search and propagation on all frames in self.full_data.
        Results (focused complex frames and best z-indexes) are streamed
        to disk as .npy files (still readable later with mmap_mode='r').
        """
        if self.full_data is None:
            self.status_label.setText("Load data first")
            return
        if self.fov_coords is None:
            self.status_label.setText("Apply a crop first")
            return

        z_settings = self.get_zprop_settings()
        self.status_label.setText(f"Running focus search with settings: {z_settings}")
        
        start_time = time.time()

        x1, y1, x2, y2 = self.fov_coords
        n_frames = self.full_data.shape[0]
        crop_h, crop_w = y2 - y1, x2 - x1

        # Ensure save folder exists
        if not self.save_folder:
            self.status_label.setText("Select a save folder first")
            return
        os.makedirs(self.save_folder, exist_ok=True)

        # Inside save folder, create subfolder for this run called cell and a number, if cell1 exists, use cell2, etc.
        run_idx = 1
        while os.path.exists(os.path.join(self.save_folder, f"cell_{run_idx}")):
            run_idx += 1
        os.makedirs(os.path.join(self.save_folder, f"cell_{run_idx}"), exist_ok=True)
        self.status_label.setText(f"Results will be saved to folder: {os.path.join(self.save_folder, f'cell_{run_idx}')}")

        # Paths for saving results
        self.current_run_folder = os.path.join(self.save_folder, f"cell_{run_idx}")

        # Prepare output files (disk-backed .npy arrays)
        focused_path = os.path.join(self.current_run_folder, "focused_frames.npy")
        zvalues_path  = os.path.join(self.current_run_folder, "focus_z_values.npy")

        self.focused_frames = np.lib.format.open_memmap(
            focused_path, mode="w+", dtype=np.complex64, shape=(n_frames, crop_h, crop_w)
        )
        self.z_values = np.lib.format.open_memmap(
            zvalues_path, mode="w+", dtype=np.float32, shape=(n_frames,)
        )

        # Initialise propagator once
        zv = np.linspace(z_settings["z_min"], z_settings["z_max"], z_settings["z_steps"])

        propagator = prop.Propagator(
            image_size=(crop_h, crop_w),
            padding=128,
            wavelength=z_settings["wavelength"],
            pixel_size=0.114,
            ri_medium=1.33,
            zv=zv,    
        )

        best_index_prev = None

        for k, frame in enumerate(self.full_data):
            if k % 50 == 0:
                self.status_label.setText(f"Processing frame {k}/{n_frames}")
                
            # ----- optional FFT -----
            if self.fft_checkbox.isChecked():
                orig_size = self.get_original_image_size()
                pupil_radius = self.get_fft_radius()
                mask_shape = self.mask_shape_combo.currentText()

                if orig_size is not None and pupil_radius is not None:
                    frame_complex = fl.vec_to_field(
                        frame,
                        pupil_radius=pupil_radius,
                        shape=orig_size,
                        mask_shape=mask_shape
                    )
                elif not self.fft_checkbox.isChecked() and frame.ndim == 1:
                    self.status_label.setText("Warning: Loaded frame is 1D, consider applying FFT")
                    return
            else:
                frame_complex = frame

            # crop
            frame_cropped = frame_complex[y1:y2, x1:x2]

            # ensure torch tensor for propagator
            if isinstance(frame_cropped, np.ndarray):
                frame_cropped = torch.tensor(frame_cropped, dtype=torch.complex64, device=propagator.device)

            # propagate and find best focus
            focused_field, best_index, best_z_value = propagator.focus_field(
                frame_cropped,
                sigma_background=30,
                previous_index=best_index_prev,
                alpha=0.8,
                return_unprocessed=z_settings["return_unprocessed"]
            )

            # store results directly to the disk-backed arrays
            self.focused_frames[k] = focused_field.cpu().numpy() if torch.is_tensor(focused_field) else focused_field
            self.z_values[k] = best_z_value
            best_index_prev = best_index

        elapsed = time.time() - start_time
        self.status_label.setText(
            f"Focus search complete: {n_frames} frames processed in {elapsed:.1f} s\n"
            f"Results saved:\n{focused_path}\n{zvalues_path}"
        )

        # plot z-values over time and save
        self.save_focus_plot()

    def save_focus_plot(self):
        if self.z_values is None:
            self.status_label.setText("No z-values available yet")
            return

        plt.figure(figsize=(8, 4))
        plt.plot(self.z_values, marker='o', linestyle='-', color='blue', markersize=2.5, linewidth=2)
        plt.xlabel("Frame index")
        plt.ylabel("Estimated focus z-position (µm)")
        plt.title("Focus positions over time")
        plt.grid(True)

        # Save inside current run folder
        plot_path = os.path.join(self.current_run_folder, "focus_over_time.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()

        self.status_label.setText(f"Focus plot saved to {plot_path}")        

    def save_random_tiffs(self):
        if self.focused_frames is None:
            self.status_label.setText("Run focus first")
            return

        n_random = self.num_tiffs_spin.value()  # user-specified number of images
        n_frames = self.focused_frames.shape[0]
        selected = random.sample(range(n_frames), min(n_random, n_frames))

        # Determine which channels to save
        channels = []
        if self.save_real_checkbox.isChecked():
            channels.append("real")
        if self.save_imag_checkbox.isChecked():
            channels.append("imag")
        if self.save_abs_checkbox.isChecked():
            channels.append("abs")

        if len(channels) == 0:
            self.status_label.setText("No channels selected to save.")
            return

        # Ensure 'images' folder exists
        images_folder = os.path.join(self.current_run_folder, "images")
        os.makedirs(images_folder, exist_ok=True)

        for i, idx in enumerate(selected):
            frame = self.focused_frames[idx]
            # Build RGB array depending on selected channels
            rgb = np.zeros((frame.shape[0], frame.shape[1], len(channels)), dtype=np.float32)
            for ch_idx, ch in enumerate(channels):
                if ch == "real":
                    rgb[..., ch_idx] = np.real(frame)
                elif ch == "imag":
                    rgb[..., ch_idx] = np.imag(frame)
                elif ch == "abs":
                    rgb[..., ch_idx] = np.abs(frame)

            fname = os.path.join(images_folder, f"frame_{idx:04d}.tiff")
            tifffile.imwrite(fname, rgb)

        self.status_label.setText(
            f"Saved {len(selected)} TIFFs with channels {channels} to: {images_folder}"
        )









