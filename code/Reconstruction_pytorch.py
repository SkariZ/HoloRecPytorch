import torch
import torch.nn as nn
import torch.nn.functional as F

import other_utils as ou
import reconstruction_utils as ru

class HolographicReconstruction(nn.Module):
    def __init__(
            self,
            image_size=(1450, 1930),
            do_precalculations=False,
            first_image=None,
            crop=0,
            lowpass_filtered_phase=None,
            filter_radius=200,
            correct_fourier_peak=(0, 0),
            ):

        super(HolographicReconstruction, self).__init__()

        # Parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.image_size = image_size
        self.first_image = first_image.to(self.device) if first_image is not None else None
        self.crop = crop
        self.lowpass_filtered_phase = lowpass_filtered_phase
        self.filter_radius = filter_radius
        self.correct_fourier_peak = correct_fourier_peak


        # Do precalculations
        if do_precalculations:
            self.precalculations()



    def precalculations(self):
        """
        Precalculations for the reconstruction.
        """

        #Get the size of the image
        yr, xr = self.first_image.shape
        
        #Set the filter radius if not set
        if not isinstance(filter_radius, (int, torch.uint8)):
            self.filter_radius = int(min(torch.tensor([xr, yr])).item() / 7)

        #Generate coordinates for full frame
        x = torch.arange(-(xr/2-1/2), (xr/2 + 1/2), 1, device=self.first_image.device)
        y = torch.arange(-(yr/2-1/2), (yr/2 + 1/2), 1, device=self.first_image.device)

        self.X, self.Y = torch.meshgrid(x, y, indexing='ij')
        self.position_matrix = torch.sqrt(X**2 + Y**2)#Distance from center
        
        #Cropping the field to avoid edge effects
        if self.crop > 0:
            xrc = xr - self.crop*2
            yrc = yr - self.crop*2
            
            xc = torch.arange(-(xrc/2-1/2), (xrc/2 + 1/2), 1, device=self.first_image.device)
            yc = torch.arange(-(yrc/2-1/2), (yrc/2 + 1/2), 1, device=self.first_image.device)
            Y_c, X_c = torch.meshgrid(xc, yc, indexing='ij')
        else:
            self.X_c = X
            self.Y_c = Y
        
        # kx and ky is the wave vector that defines the direction of propagation. Used to calculate the fourier shift.
        kx = torch.linspace(torch.pi, torch.pi, xr) 
        ky = torch.linspace(torch.pi, torch.pi, yr)
        self.KX, self.KY = torch.meshgrid(kx, ky, indexing='ij')
    

    def forward(self, holograms):
        """
        Forward pass of the model.

        Parameters:
        - holograms (torch.Tensor): Holograms to reconstruct.
        """

        #Subtract the mean
        holograms = holograms - holograms.mean()

        # Fourier transform


        pass

