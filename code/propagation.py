import torch

class Propagator():
    def __init__(self, image_size, wavelength=0.532, pixel_size=0.114, device='cpu'):
        
        self.image_size = image_size
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.device = device

        # Padding to avoid edge effects
        self.padding = image_size // 2

    def forward(self, x):
        pass

    def precalculate_kernel(self):
        pass

    def find_focus(self, x):
        pass
