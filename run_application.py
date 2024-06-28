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
        self.colorbar = None  # Attribute to store the colorbar
        self.cmap = 'viridis'
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        self.canvas = FigureCanvas(plt.Figure())
        layout.addWidget(self.canvas)
        
        self.ax = self.canvas.figure.add_subplot(111)

        self.update_plot()
            
        self.ax.set_title(self.title)
        #self.ax.figure.tight_layout()

    def update_plot(self):
        # Clear the colorbar if it exists
        if self.colorbar:
            self.colorbar.remove()
            self.colorbar = None

        self.ax.clear()

        if self.is_histogram:
            self.ax.hist(self.data, bins=255, color='darkblue', alpha=0.7, density=True)
        else:
            im = self.ax.imshow(self.data, cmap=self.cmap, interpolation='none')
            self.colorbar = self.ax.figure.colorbar(im, ax=self.ax)
            #self.ax.axis('off')

        self.ax.set_title(self.title)
        self.canvas.draw()

    def update_data(self, data, is_histogram=False, title=""):
        self.data = data
        self.is_histogram = is_histogram
        self.title = title

        self.update_plot()
        #self.ax.figure.tight_layout()


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
        self.titles = ["Fft centered", "Histogram Hologram", "First phase", "First phase background", "Imaginary part", "Real part"]
        self.image_widgets = []
        for i in range(6):
            if i == 1:
                self.image_widgets.append(ImageWidget(np.random.rand(CFG.rec_params['height']*CFG.rec_params['width']), is_histogram=True, title=self.titles[i]))
            else:
                self.image_widgets.append(ImageWidget(np.random.rand(CFG.rec_params['height'], CFG.rec_params['width']), title=self.titles[i]))

        for idx, image_widget in enumerate(self.image_widgets):
            self.image_layout.addWidget(image_widget, idx // 2, idx % 2)

        # Initialize some variables
        self.frame = np.zeros([2, 2])
        self.prev_filename = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.R = None

    def precalculate(self):
        # Retrieve parameter values
        param_values = {param: input_field.text() for param, input_field in self.param_inputs.items()}
        
        #Transform values to correct type
        param_values['filename'] = param_values['filename']
        param_values['frame_idx'] = int(param_values['frame_idx'])
        param_values['height'] = int(param_values['height'])
        param_values['width'] = int(param_values['width'])
        param_values['crop'] = int(param_values['crop'])
        param_values['lowpass_filtered_phase'] = bool(param_values['lowpass_filtered_phase'])
        param_values['filter_radius'] = int(param_values['filter_radius']) if not param_values['filter_radius'] == 'None' else None
        #If mask_radiis is a list of integers, split it by comma and convert to list of integers
        param_values['mask_radiis'] = [int(r) for r in param_values['mask_radiis'].split(',')] if not param_values['mask_radiis'] == 'None' else None
        param_values['mask_case'] = param_values['mask_case']
        param_values['phase_corrections'] = int(param_values['phase_corrections'])
        param_values['skip_background_correction'] = int(param_values['skip_background_correction'])
        param_values['kernel_size'] = int(param_values['kernel_size'])
        param_values['sigma'] = int(param_values['sigma'])

        #Read 1 frame from the video
        if param_values['filename'] != self.prev_filename or self.prev_start_frame != param_values['frame_idx'] or self.frame.shape[0]<param_values['height'] or self.frame.shape[1]<param_values['width']:

            self.frame = rv.read_video(
                param_values['filename'], 
                start_frame=param_values['frame_idx'], 
                max_frames=param_values['frame_idx']+1)[0]

            #Check so that the frame is not empty
            if self.frame.size == 0:
                self.recon_info.setText("No frames in the video")
                return
            
            #Save previous filename so that the frame is not read again if the filename is the same
            self.prev_filename = param_values['filename']
            self.prev_start_frame = param_values['frame_idx']
        
        #Check if height and width are the same as the frame size else crop the frame
        if param_values['height'] < self.frame.shape[0] or param_values['width'] < self.frame.shape[1]:
            #Crop the frame to the correct size
            self.frame = self.frame[:param_values['height'], :param_values['width']]

        #Perform reconstruction
        self.R = rec.HolographicReconstruction(
            image_size=(param_values['height'], param_values['width']),
            first_image=self.frame,
            crop=param_values['crop'],
            lowpass_filtered_phase=param_values['lowpass_filtered_phase'],
            filter_radius=param_values['filter_radius'],
            mask_radiis=param_values['mask_radiis'],
            mask_case=param_values['mask_case'],
            phase_corrections=param_values['phase_corrections'],
            skip_background_correction=param_values['skip_background_correction'],
            kernel_size=param_values['kernel_size'],
            sigma=param_values['sigma']
            )

        # Perform precalculation
        self.R.precalculations()

        xc, yc = self.R.image_size[0]//2, self.R.image_size[1]//2

        #Images to visualize
        images = [
            torch.log10(self.R.fftIm2.abs()).cpu().numpy()[
                xc - self.R.filter_radius:xc + self.R.filter_radius, yc - self.R.filter_radius:yc + self.R.filter_radius
                ],
            self.R.first_phase.cpu().numpy(),
            self.R.phase_img_smooth.squeeze(0).squeeze(0).cpu().numpy(),
            self.R.first_field_corrected.squeeze(0).squeeze(0).imag.cpu().numpy(),
            self.R.first_field_corrected.squeeze(0).squeeze(0).real.cpu().numpy()
            ]

        #Histogram of the hologram
        histogram_data = self.frame[::4, ::4].flatten()

        self.image_widgets[0].update_data(images[0], title=self.titles[0])
        self.image_widgets[1].update_data(histogram_data, is_histogram=True, title=self.titles[1])

        #Update the rest of the images
        for idx, image in enumerate(images[1:]):
            self.image_widgets[idx+2].update_data(image, title=self.titles[idx+2])

        
    def reconstruction(self):
        pass



if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    sys.exit(app.exec_())
