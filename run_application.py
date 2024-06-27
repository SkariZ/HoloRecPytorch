import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
                             QFormLayout, QGridLayout, QSizePolicy)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import torch

import cfg as CFG

#import read_video as rv from folder code
sys.path.append('code')
import read_video as rv
import reconstruction as rec

class ImageWidget(QWidget):
    def __init__(self, data, is_histogram=False, title=""):
        super().__init__()
        self.data = data
        self.is_histogram = is_histogram
        self.title = title
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        self.canvas = FigureCanvas(plt.Figure())
        layout.addWidget(self.canvas)
        
        self.ax = self.canvas.figure.add_subplot(111)
        if self.is_histogram:
            self.ax.hist(self.data, bins=255, color='darkblue', alpha=0.7, density=True)
        else:
            self.ax.imshow(self.data, cmap='gray')

        self.ax.set_title(self.title)
        self.ax.axis('off')

    def update_data(self, data, is_histogram=False, title=""):
        self.data = data
        self.is_histogram = is_histogram
        self.title = title
        self.ax.clear()
        if self.is_histogram:
            self.ax.hist(self.data, bins=255, color='darkblue', alpha=0.7, density=True)
        else:
            self.ax.imshow(self.data)
            #self.ax.figure.colorbar(self.ax.imshow(self.data), ax=self.ax)
        self.ax.set_title(self.title)
        self.ax.axis('off')
        self.canvas.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Parameter Input and Image Display')
        
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        
        # Create main horizontal layout
        main_layout = QHBoxLayout()
        
        # Create layout for parameters
        param_layout = QVBoxLayout()
        
        self.param_inputs = {}
        form_layout = QFormLayout()
        
        # Set default values for parameters
        default_values = CFG.rec_params
  
        for param, default_value in default_values.items():
            input_field = QLineEdit(self)
            input_field.setText(str(default_value))
            input_field.setMaximumWidth(400)  # Set maximum width for input fields
            form_layout.addRow(param, input_field)
            self.param_inputs[param] = input_field
        param_layout.addLayout(form_layout)
        
        # Create calculate button. Darkgreen color is used for the button.
        self.calc_button = QPushButton('Precalculate', self)
        self.calc_button.setStyleSheet("background-color: #4CAF50")
        self.calc_button.setMaximumWidth(400)
        self.calc_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.calc_button.clicked.connect(self.precalculate)
        param_layout.addWidget(self.calc_button)
        #main_layout.addLayout(param_layout)
        #Make the button come right under the input fields
        param_layout.addStretch(1)

        #Below the precalculate button, a new set of default values is displayed
        form_layout2 = QFormLayout()
        default_values2 = CFG.rec_params_full
        for param, default_value in default_values2.items():
            input_field = QLineEdit(self)
            input_field.setText(str(default_value))
            input_field.setMaximumWidth(400)
            form_layout2.addRow(param, input_field)
            self.param_inputs[param] = input_field
        param_layout.addLayout(form_layout2)

        # Create reconstruction button.
        self.recon_button = QPushButton('Reconstruction', self)
        self.recon_button.setStyleSheet("background-color: #FF5733")
        self.recon_button.setMaximumWidth(400)
        self.recon_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.recon_button.clicked.connect(self.reconstruction)
        param_layout.addWidget(self.recon_button)
        #main_layout.addLayout(param_layout)

        # Create a box that prints some text once the reconstruction is done with time and other information
        self.recon_info = QLabel("Reconstruction information")
        self.recon_info.setAlignment(Qt.AlignCenter)
        param_layout.addWidget(self.recon_info)
        main_layout.addLayout(param_layout)

        # Create layout for images
        self.image_layout = QGridLayout()
        main_layout.addLayout(self.image_layout)
        
        self.centralWidget.setLayout(main_layout)
        self.show()
        
        # Initialize image widgets with titles
        self.titles = ["Fft centered", "Histogram Hologram", "Phase", "First phase background", "Imaginary part", "Real part"]
        self.image_widgets = []
        for i in range(6):
            if i == 1:
                self.image_widgets.append(ImageWidget(np.random.rand(CFG.rec_params['height']*CFG.rec_params['width']), is_histogram=True, title=self.titles[i]))
            else:
                self.image_widgets.append(ImageWidget(np.random.rand(CFG.rec_params['height'], CFG.rec_params['width']), title=self.titles[i]))

        for idx, image_widget in enumerate(self.image_widgets):
            self.image_layout.addWidget(image_widget, idx // 2, idx % 2)

        # Initialize some variables
        self.frame = None
        self.prev_filename = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def precalculate(self):
        # Retrieve parameter values
        param_values = {param: input_field.text() for param, input_field in self.param_inputs.items()}
        
        #Transform values to correct type
        param_values['filename'] = param_values['filename']
        param_values['height'] = int(param_values['height'])
        param_values['width'] = int(param_values['width'])
        param_values['crop'] = int(param_values['crop'])
        param_values['lowpass_filtered_phase'] = bool(param_values['lowpass_filtered_phase'])
        param_values['filter_radius'] = int(param_values['filter_radius']) if not param_values['filter_radius'] == 'None' else None
        #If mask_radiis is a list of integers, split it by comma and convert to list of integers
        param_values['mask_radiis'] = [int(r) for r in param_values['mask_radiis'].split(',')] if not param_values['mask_radiis'] == 'None' else None
        param_values['mask_case'] = param_values['mask_case']
        param_values['phase_corrections'] = int(param_values['phase_corrections'])
        param_values['kernel_size'] = int(param_values['kernel_size'])
        param_values['sigma'] = int(param_values['sigma'])

        #Read 1 frame from the video
        if param_values['filename'] != self.prev_filename and self.frame == None:
            self.frame = rv.read_video(param_values['filename'], start_frame=0, max_frames=1)[0]

            #Check so that the frame is not empty
            if self.frame.size == 0:
                self.recon_info.setText("No frames in the video")
                return
            self.prev_filename = param_values['filename']
        
        #Perform reconstruction
        R = rec.HolographicReconstruction(
            image_size=(param_values['height'], param_values['width']),
            first_image=self.frame,
            crop=param_values['crop'],
            lowpass_filtered_phase=param_values['lowpass_filtered_phase'],
            filter_radius=param_values['filter_radius'],
            mask_radiis=param_values['mask_radiis'],
            mask_case=param_values['mask_case'],
            phase_corrections=param_values['phase_corrections'],
            kernel_size=param_values['kernel_size'],
            sigma=param_values['sigma']
            )

        # Perform precalculation
        R.precalculations()

        self.titles = ["Fft centered", "Histogram Hologram", "Phase", "First phase background", "Imaginary part", "Real part"]
        
        # Perform precalculation (dummy data used here for demonstration)
        images = [
            torch.log10(R.fftIm2.abs()).cpu().numpy(),
            R.first_phase.cpu().numpy(),
            R.phase_img_smooth.squeeze(0).squeeze(0).cpu().numpy(),
            R.first_field_corrected.squeeze(0).squeeze(0).real.cpu().numpy(),
            R.first_field_corrected.squeeze(0).squeeze(0).imag.cpu().numpy()
        ]
        
        histogram_data = self.frame[::4, ::4].flatten()

        # Update images and histogram
        for idx, image in enumerate(images):
            if idx==1:
                self.image_widgets[idx].update_data(histogram_data, is_histogram=True, title=self.titles[idx])
            else:
                self.image_widgets[idx].update_data(image, title=self.titles[idx])
        
    def reconstruction(self):
        pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    sys.exit(app.exec_())
