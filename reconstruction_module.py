# reconstruction_module.py
import sys
import os
import time
import numpy as np
import glob
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
                             QLineEdit, QPushButton, QLabel, QSizePolicy, QFileDialog,
                             QDialog, QTextEdit, QSlider, QScrollArea)
from PyQt5.QtCore import Qt

import cfg as CFG

sys.path.append('code')
import read_video as rv
import reconstruction as rec
import image_utils as iu
import propagation as prop

class ImageWidget(QWidget):
    def __init__(self, data, is_histogram=False, title=""):
        super().__init__()
        self.data = data
        self.is_histogram = is_histogram
        self.title = title
        self.colorbar = None
        self.cmap = 'gray'
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        self.canvas = FigureCanvas(plt.Figure())
        
        # Make the canvas expand with the parent widget
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        
        layout.addWidget(self.canvas)
        
        self.ax = self.canvas.figure.add_subplot(111)
        self.update_plot()
        self.ax.set_title(self.title)

    def update_plot(self):
        if self.colorbar:
            self.colorbar.remove()
            self.colorbar = None
        self.ax.clear()
        if self.is_histogram:
            hist, bins = np.histogram(self.data, bins=np.arange(257))  # 0-255
            self.ax.bar(bins[:-1], hist, width=1, color='darkblue', alpha=0.6, align="center")
            smoothed = gaussian_filter1d(hist, sigma=2)
            self.ax.plot(bins[:-1], smoothed, color="red", linewidth=2, label="Smoothed")
            self.ax.set_xlim(0, 255)
            self.ax.set_ylabel("Frequency")
        else:
            im = self.ax.imshow(self.data, cmap=self.cmap, interpolation='none', aspect='auto')
            self.colorbar = self.ax.figure.colorbar(im, ax=self.ax)
        self.ax.set_title(self.title)
        self.canvas.draw()

    def update_data(self, data, is_histogram=False, title=""):
        self.data = data
        self.is_histogram = is_histogram
        self.title = title
        self.update_plot()


class ReconstructionModule(QWidget):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.R = None
        self.frame = np.zeros([2, 2])
        self.z_stack = None
        self.current_z_idx = 0
        self.focus_idx = None
        self.focus_um = None
        self.prev_filename = None
        self.prev_start_frame = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Reconstruction')
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # ---------------- Parameter Section ----------------
        param_layout = QVBoxLayout()
        self.param_inputs = {}

        # Help button
        self.help_button = QPushButton("Help")
        self.help_button.clicked.connect(self.show_help)
        param_layout.addWidget(self.help_button)

        # First form layout (Precalculate)
        form_layout = QFormLayout()
        for param, default in CFG.rec_params.items():
            inp = QLineEdit(str(default))
            inp.setMaximumWidth(400)
            self.param_inputs[param] = inp

            # Filename browse
            if param == 'filename':
                hbox = QHBoxLayout()
                hbox.addWidget(inp)
                browse_btn = QPushButton("Browse File")
                browse_btn.clicked.connect(lambda _, le=inp: self.select_file(le))
                hbox.addWidget(browse_btn)
                form_layout.addRow(param, hbox)
            else:
                form_layout.addRow(param, inp)

        param_layout.addLayout(form_layout)

        # Precalculate button
        self.calc_button = QPushButton('Precalculate')
        self.calc_button.setStyleSheet("background-color: #4CAF50")
        self.calc_button.setMaximumWidth(400)
        self.calc_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.calc_button.clicked.connect(self.precalculate)
        param_layout.addWidget(self.calc_button)

        self.precalc_info = QLabel("Press Button to precalculate")
        self.precalc_info.setAlignment(Qt.AlignCenter)
        param_layout.addWidget(self.precalc_info)

        # ---------------- Z-propagation Section ----------------
        z_layout = QFormLayout()
        self.z_min_input = QLineEdit()
        self.z_max_input = QLineEdit()
        self.z_steps_input = QLineEdit()
        self.wavelength_input = QLineEdit()

        self.z_min_input.setText(str(CFG.zprop_defaults["z_min"]))
        self.z_max_input.setText(str(CFG.zprop_defaults["z_max"]))
        self.z_steps_input.setText(str(CFG.zprop_defaults["z_steps"]))
        self.wavelength_input.setText(str(CFG.zprop_defaults["wavelength"]))

        z_layout.addRow("Z min (μm):", self.z_min_input)
        z_layout.addRow("Z max (μm):", self.z_max_input)
        z_layout.addRow("Steps:", self.z_steps_input)
        z_layout.addRow("Wavelength (μm):", self.wavelength_input)

        # Compute z-stack button
        self.z_compute_btn = QPushButton("Compute Z-Propagation")
        self.z_compute_btn.setStyleSheet("background-color: #FFA500")
        self.z_compute_btn.clicked.connect(self.compute_z_stack)
        z_layout.addRow(self.z_compute_btn)

        # Feedback label for z-propagation
        self.z_info = QLabel("Press button to compute z-propagation")
        self.z_info.setAlignment(Qt.AlignCenter)
        z_layout.addRow(self.z_info)

        # Slider
        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setMinimum(0)
        self.z_slider.setMaximum(0)
        self.z_slider.setValue(0)
        self.z_slider.setEnabled(False)
        self.z_slider.valueChanged.connect(self.update_z_view)
        z_layout.addRow("Focus:", self.z_slider)

        # Set focus input/button
        focus_hbox = QHBoxLayout()
        self.focus_input = QLineEdit("0")
        self.focus_btn = QPushButton("Set Focus")
        self.focus_btn.clicked.connect(self.set_focus)
        focus_hbox.addWidget(self.focus_input)
        focus_hbox.addWidget(self.focus_btn)
        z_layout.addRow(focus_hbox)

        # Add spacing around Z-propagation box
        param_layout.addSpacing(20)  # space between Precalculate and Z-propagation
        param_layout.addLayout(z_layout)
        param_layout.addSpacing(20)  # space between Z-propagation and Reconstruction

        # Second form layout (Reconstruction)
        form_layout2 = QFormLayout()
        for param, default in CFG.rec_params_full.items():
            inp = QLineEdit(str(default))
            inp.setMaximumWidth(400)
            self.param_inputs[param] = inp

            # Save folder browse
            if param == 'save_folder':
                hbox = QHBoxLayout()
                hbox.addWidget(inp)
                browse_btn = QPushButton("Select Folder")
                browse_btn.clicked.connect(lambda _, le=inp: self.select_folder(le))
                hbox.addWidget(browse_btn)
                form_layout2.addRow(param, hbox)
            else:
                form_layout2.addRow(param, inp)

        param_layout.addLayout(form_layout2)

        # Reconstruction button
        self.recon_button = QPushButton('Reconstruction')
        self.recon_button.setStyleSheet("background-color: #FF5733")
        self.recon_button.setMaximumWidth(400)
        self.recon_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.recon_button.clicked.connect(self.reconstruction)
        param_layout.addWidget(self.recon_button)

        self.recon_info = QLabel("Press Button to reconstruct the hologram")
        self.recon_info.setAlignment(Qt.AlignCenter)
        param_layout.addWidget(self.recon_info)

        # ---- wrap the parameter panel in a scroll area ----
        param_container = QWidget()
        param_container.setLayout(param_layout)

        param_scroll = QScrollArea()
        param_scroll.setWidgetResizable(True)
        param_scroll.setWidget(param_container)
        param_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        param_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        main_layout.addWidget(param_scroll)

        # ---------------- Image Section ----------------
        self.image_layout = QGridLayout()
        main_layout.addLayout(self.image_layout)

        self.titles = ["Fft centered", "Histogram Hologram", "First phase", 
                       "First phase background", "Imaginary part", "Real part"]
        self.image_widgets = []

        for i in range(6):
            if i == 1:
                data = np.random.rand(CFG.rec_params['height']*CFG.rec_params['width'])
                self.image_widgets.append(ImageWidget(data, is_histogram=True, title=self.titles[i]))
            else:
                data = np.random.rand(CFG.rec_params['height'], CFG.rec_params['width'])
                self.image_widgets.append(ImageWidget(data, title=self.titles[i]))
            self.image_layout.addWidget(self.image_widgets[-1], i//2, i%2)

        # Set column/row stretches
        self.image_layout.setColumnStretch(0, 1)
        self.image_layout.setColumnStretch(1, 1)
        self.image_layout.setRowStretch(0, 1)
        self.image_layout.setRowStretch(1, 1)
        self.image_layout.setRowStretch(2, 1)

        #self.show()

    # ---------------- Helper functions ----------------
    def select_file(self, line_edit):
        fname, _ = QFileDialog.getOpenFileName(self, "Select File")
        if fname:
            line_edit.setText(fname)

    def select_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            line_edit.setText(folder)

    def show_help(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Help: Parameter Explanations")
        layout = QVBoxLayout()
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_content = ""
        for param, desc in CFG.rec_param_descriptions.items():
            help_content += f"{param}: {desc}\n\n"
        help_text.setText(help_content)
        layout.addWidget(help_text)
        dlg.setLayout(layout)
        dlg.resize(600, 500)
        dlg.exec_()

    # ---------------- Handlers ----------------
    def precalculate(self):
            # Retrieve parameter values
            param_values = {param: input_field.text() for param, input_field in self.param_inputs.items()}
            
            #Transform values to correct type
            param_values['filename'] = param_values['filename']
            param_values['frame_idx'] = int(param_values['frame_idx'])
            param_values['height'] = int(param_values['height'])
            param_values['width'] = int(param_values['width'])
            param_values['corner'] = int(param_values['corner'])
            param_values['lowpass_filtered_phase'] = int(param_values['lowpass_filtered_phase'])
            param_values['filter_radius'] = int(param_values['filter_radius']) if not param_values['filter_radius'] == 'None' else None
            #If mask_radiis is a list of integers, split it by comma and convert to list of integers
            param_values['mask_radiis'] = [int(r) for r in param_values['mask_radiis'].split(',')] if not param_values['mask_radiis'] == 'None' else None
            param_values['mask_case'] = param_values['mask_case']
            param_values['mask_out'] = int(param_values['mask_out'])
            param_values['phase_corrections'] = int(param_values['phase_corrections'])
            param_values['skip_background_correction'] = int(param_values['skip_background_correction'])
            param_values['correct_field'] = int(param_values['correct_field'])
            param_values['lowpass_kernel_end'] = int(param_values['lowpass_kernel_end'])
            param_values['kernel_size'] = int(param_values['kernel_size'])
            param_values['sigma'] = int(param_values['sigma'])

            #Read 1 frame from the video
            try:
                self.frame = rv.read_video(
                    param_values['filename'], 
                    start_frame=param_values['frame_idx'], 
                    max_frames=param_values['frame_idx']+1
                    )[0]
            except:
                self.recon_info.setText("No frames in the video or wrong filename")
                return

            #Check so that the frame is not empty
            if self.frame.size == 0:
                self.recon_info.setText("No frames in the video")
                return
            
            #Check if height and width are the same as the frame size else crop the frame
            if param_values['height'] <= self.frame.shape[0] and param_values['width'] <= self.frame.shape[1]:
                #Crop the frame to the correct size
                self.frame = iu.cropping_image(self.frame, h=param_values['height'], w=param_values['width'], corner=param_values['corner'])


            # Initialize reconstruction object
            self.R = rec.HolographicReconstruction(
                image_size=(param_values['height'], param_values['width']),
                first_image=self.frame,
                #crop=param_values['crop'],
                lowpass_filtered_phase=param_values['lowpass_filtered_phase'],
                filter_radius=param_values['filter_radius'],
                mask_radiis=param_values['mask_radiis'],
                mask_case=param_values['mask_case'],
                mask_out=param_values['mask_out'],
                phase_corrections=param_values['phase_corrections'],
                skip_background_correction=param_values['skip_background_correction'],
                correct_field=param_values['correct_field'],
                lowpass_kernel_end=param_values['lowpass_kernel_end'],
                kernel_size=param_values['kernel_size'],
                sigma=param_values['sigma']
                )

            # Perform precalculation
            start_time = time.time()
            self.R.precalculations()
            end_time = time.time()

            #Change color on the button to show that the precalculation is done
            self.calc_button.setStyleSheet("background-color: #00FF00")
            #Print information about the precalculation
            self.precalc_info.setText(f"Precalculation done in {end_time-start_time:.2f} seconds")

            #Images to visualize: FFT, Phase, and Field
            images = [
                torch.log10(self.R.fftIm2.abs()).cpu().numpy(),
                self.R.first_phase.cpu().numpy(),
                self.R.phase_img_smooth.squeeze(0).squeeze(0).cpu().numpy(),
                self.R.first_field_corrected.squeeze(0).squeeze(0).imag.cpu().numpy(),
                self.R.first_field_corrected.squeeze(0).squeeze(0).real.cpu().numpy()
                ]

            #Histogram of the hologram
            histogram_data = self.frame.flatten()

            #10000 random numbers from histogram_data
            histogram_data = np.random.choice(histogram_data, 25000) if histogram_data.size > 25000 else histogram_data

            self.image_widgets[0].update_data(images[0], title=self.titles[0])
            self.image_widgets[1].update_data(histogram_data, is_histogram=True, title=self.titles[1])

            #Update the rest of the images
            for idx, image in enumerate(images[1:]):
                self.image_widgets[idx+2].update_data(image, title=self.titles[idx+2])

            # After precalc, enable z-slider and reset stack
            self.z_stack = None
            self.z_slider.setEnabled(False)
            self.current_z_idx = 0

    def compute_z_stack(self):
        try:
            # Read z-propagation parameters
            z_min = float(self.z_min_input.text())   # umeters
            z_max = float(self.z_max_input.text())   # umeters
            n_steps = int(self.z_steps_input.text())

            # Generate z values
            self.z_values = np.linspace(z_min, z_max, n_steps)

            # Make sure 0 is included and sorted.
            if 0 not in self.z_values:
                self.z_values = np.insert(self.z_values, 0, 0)
                self.z_values.sort()
                self.z_values = np.unique(self.z_values)


            # Wavelength (µm → m)
            wavelength = float(self.wavelength_input.text())

            # Field from reconstruction
            field = self.R.first_field_corrected.squeeze()
            h, w = field.shape

            # Init propagator
            self.propagator = prop.Propagator(
                image_size=(h, w),
                wavelength=wavelength,
                padding=256
            )

            # Compute stack
            self.propagated_stack = self.propagator.forward(field, self.z_values)

            # Enable slider
            self.z_slider.setEnabled(True)
            self.z_slider.setMaximum(n_steps)
            self.z_slider.setValue(0)

            # Feedback text in propagation box
            self.z_info.setText(
                f"Z-propagation done: {n_steps} steps\nrange {z_min} – {z_max} um"
            )

            # Show first slice
            self.update_z_view()

        except Exception as e:
            self.z_info.setText(f"Error in z-propagation: {e}")

    def update_z_view(self):
        if self.propagated_stack is None:
            return

        idx = self.z_slider.value()
        self.current_focus_idx = idx
        # Show both index and µm value
        self.focus_input.setText(f"{idx}  ({self.z_values[idx-1]:.2f} µm)")

        # Safely extract real/imag
        current_field = self.propagated_stack[idx]
        if isinstance(current_field, torch.Tensor):
            current_field = current_field.cpu().numpy()

        self.image_widgets[4].update_data(current_field.imag, title="Imaginary part")
        self.image_widgets[5].update_data(current_field.real, title="Real part")

    def set_focus(self):
        if self.propagated_stack is None:
            return
        try:
            text = self.focus_input.text().strip()
            if text == "":
                self.z_info.setText("Empty focus input")
                return

            # Try interpreting as index first
            # Split by space and take first part
            text = text.split()[0]

            if text.isdigit():
                idx = int(text)
                idx = max(0, min(idx, self.z_slider.maximum()))
            else:
                # Interpret as z-value in µm
                z_um = float(text)
                # Find the closest index
                idx = int(np.argmin(np.abs(self.z_values - z_um)))

            # Update slider
            self.z_slider.setValue(idx)

            # Store focus
            self.focus_idx = idx
            self.focus_um = self.z_values[idx]

            # Update display
            self.focus_input.setText(f"{self.focus_um:.3f}")
            self.z_info.setText(f"Focus set to index {idx} (z = {self.focus_um:.3f} µm)")

        except ValueError:
            self.z_info.setText("Invalid focus input")


    def reconstruction(self):
        
        start_time = time.time()

        # ---------------- Retrieve parameters ----------------
        param_values = {param: input_field.text() for param, input_field in self.param_inputs.items()}
        param_values['save_folder'] = param_values['save_folder']
        param_values['n_frames'] = int(param_values['n_frames'])
        param_values['n_frames_max_mem'] = int(param_values['n_frames_max_mem'])
        param_values['start_frame'] = int(param_values['start_frame'])
        param_values['n_frames_step'] = int(param_values['n_frames_step'])
        param_values['fft_save'] = int(param_values['fft_save'])
        param_values['recalculate_offset'] = int(param_values['recalculate_offset'])
        param_values['save_movie_gif'] = int(param_values['save_movie_gif'])
        param_values['colormap'] = param_values['colormap']
        param_values['cornerf'] = int(param_values['cornerf'])

        # ---------------- Create folders ----------------
        os.makedirs(f"{param_values['save_folder']}/field/", exist_ok=True)
        os.makedirs(f"{param_values['save_folder']}/images/", exist_ok=True)
        os.makedirs(f"{param_values['save_folder']}/frames/", exist_ok=True)

        # ---------------- Check precalculation ----------------
        if self.R is None:
            self.recon_info.setText("No precalculation done")
            return

        # ---------------- Update reconstruction params ----------------
        self.R.fft_save = param_values['fft_save']
        self.R.recalculate_offset = param_values['recalculate_offset']

        # ---------------- Prepare frame indices ----------------

        # for valididty check that the the number of frames to reconstruct does not exceed the total number of frames in the video
        n_frames_in_video = int(rv.get_video_properties(param_values['filename'])['frame_count'])
        if param_values['start_frame'] + (param_values['n_frames'] - 1) * param_values['n_frames_step'] >= n_frames_in_video:
            self.recon_info.setText("Error: Number of frames to reconstruct exceeds total frames in video, video has only " + str(n_frames_in_video) + " frames.")
            return

        frame_indices = list(range(
            param_values['start_frame'],
            param_values['start_frame'] + param_values['n_frames'] * param_values['n_frames_step'],
            param_values['n_frames_step']
        ))

        n_total_frames = len(frame_indices)

        # ---------------- Create memmap for reconstructed fields ----------------
        H, W = self.R.image_size
        dtype = np.complex64 if self.R.fft_save else np.float32
        n_points = self.R.get_fft_number_of_points() if self.R.fft_save else 0

        memmap_file = f"{param_values['save_folder']}/field/field.npy"
        field_memmap = np.lib.format.open_memmap(
            memmap_file, 
            mode='w+', 
            shape=(n_total_frames, n_points) if self.R.fft_save else (n_total_frames,H, W), 
            dtype=dtype
            )

        # ---------------- Batch processing ----------------
        all_results = []
        frames_processed = 0

        # Print that reconstruction has started
        self.recon_info.setText(f"Reconstruction started for {n_total_frames} frames...")

        while frames_processed < len(frame_indices):
            batch_indices = frame_indices[frames_processed:frames_processed + param_values['n_frames_max_mem']]
            frames_batch = rv.read_video_by_indices(param_values['filename'], batch_indices)

            if len(frames_batch) == 0:
                break  # No more frames

            # Crop frames
            frames_batch = np.stack([
                iu.cropping_image(f, h=self.R.image_size[0], w=self.R.image_size[1],
                                corner=param_values['cornerf']) for f in frames_batch
            ])

            # Convert to tensor and move to device
            data_batch = torch.tensor(frames_batch).to(self.device)

            # Forward reconstruction
            data_batch = self.R.forward(data_batch)

            # Refocus if needed
            if self.focus_idx is not None:
                if self.R.fft_save:
                    data_batch = self.R.load_fft(data_batch)
                data_batch = self.propagator.forward_fields(data_batch, Z=self.focus_um)
                if self.R.fft_save:
                    data_batch = self.R.save_fft(data_batch)

            # Write batch to memmap and flush
            batch_start = frames_processed
            batch_end = frames_processed + len(batch_indices)
            field_memmap[batch_start:batch_end] = data_batch.cpu().numpy()
            field_memmap.flush()

            frames_processed += len(batch_indices)

            if frames_processed % 50 == 0 or frames_processed == n_total_frames:
                self.recon_info.setText(f"Reconstructed {frames_processed}/{n_total_frames} frames...")
            

        end_time = time.time()
        self.recon_info.setText(f"Reconstruction done in {end_time-start_time:.2f} seconds, "
                                f"{(end_time-start_time)/field_memmap.shape[0]:.2f} seconds per frame")

        # ---------------- Save first frame images ----------------
        first_frame = field_memmap[0:1]
        if self.R.fft_save:
            first_frame = self.R.load_fft(torch.tensor(first_frame, device=self.device)).cpu().numpy()

        plt.imsave(f"{param_values['save_folder']}/images/abs.png", np.abs(first_frame.squeeze()), cmap=param_values['colormap'])
        plt.imsave(f"{param_values['save_folder']}/images/real.png", np.real(first_frame.squeeze()), cmap=param_values['colormap'])
        plt.imsave(f"{param_values['save_folder']}/images/imag.png", np.imag(first_frame.squeeze()), cmap=param_values['colormap'])
        plt.imsave(f"{param_values['save_folder']}/images/phase.png", np.angle(first_frame.squeeze()), cmap=param_values['colormap'])
        plt.imsave(f"{param_values['save_folder']}/images/fft.png",
                np.log10(np.abs(np.fft.fftshift(np.fft.fft2(first_frame.squeeze())))),
                cmap=param_values['colormap'])
        
        # ---------------- Save GIF if requested ----------------
        movie_cap = param_values['n_frames'] if param_values['n_frames'] < 400 else 400  # Max frames for GIF

        if param_values['save_movie_gif']:
            
            # Load frames and convert if needed
            if self.R.fft_save:
                field_batch = torch.tensor(field_memmap[:movie_cap]).to(self.device)
                field = self.R.load_fft(field_batch).cpu().numpy()
            else:
                field = field_memmap[:movie_cap]

            for i, f in enumerate(field):
                iu.save_frame(f.imag, f"{param_values['save_folder']}/frames/", name=f"imag_{i}",
                            annotate=True, annotatename=f"Frame {i}", dpi=250)
            iu.save_gif(f"{param_values['save_folder']}/frames/",
                        f"{param_values['save_folder']}/images/imag_gif.gif", duration=100, loop=0)
            for f in glob.glob(f"{param_values['save_folder']}/frames/*.png"):
                os.remove(f)

        # ---------------- Save parameters ----------------
        with open(f"{param_values['save_folder']}/parameters.txt", "w") as f:
            for k, v in param_values.items():
                f.write(f"{k}: {v}\n")
            f.write(f"Time per frame: {(time.time() - start_time)/field_memmap.shape[0]:.2f} seconds\n")
            if self.focus_um is not None:
                f.write(f"Focus (µm): {self.focus_um}\n")

        with open(f"{param_values['save_folder']}/fft_loading.txt", "w") as f:
            f.write(f"radius: {self.R.rad}\n")
            f.write(f"mask: {self.R.mask_list[0]}\n")
            f.write(f"xsize: {self.R.xrc}\n")
            f.write(f"ysize: {self.R.yrc}\n")

        # ---------------- Update UI ----------------
        self.recon_button.setStyleSheet("background-color: #00FF00")
        


