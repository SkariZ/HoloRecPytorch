import torch
import torch.nn as nn
import torch.nn.functional as F

import other_utils as OU
import fft_loader as FL

from reconstruction_utils import FourierPeakFinder #PolynomialFitter, PolynomialFitterV2, PhaseFrequencyFilter,


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
            poly_dims=4,
            mask_radiis=None,
            mask_case="ellipse",
            recalculate_offset=True,
            phase_corrections=1,
            fft_save=False
            ):

        super(HolographicReconstruction, self).__init__()

        # Parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.image_size = image_size
        self.first_image = torch.tensor(first_image).to(self.device) if first_image is not None else None
        self.crop = crop
        self.lowpass_filtered_phase = lowpass_filtered_phase
        self.filter_radius = filter_radius
        self.correct_fourier_peak = correct_fourier_peak
        self.poly_dims = poly_dims
        self.mask_radiis = mask_radiis
        self.mask_case = mask_case
        self.recalculate_offset = recalculate_offset
        self.phase_corrections = phase_corrections
        self.fft_save = fft_save

        if self.mask_radiis is not None:
            self.mask_radiis = [self.filter_radius, 10, 5]

        # Do precalculations
        if do_precalculations:
            self.precalculations() if first_image is not None else print("No first image given. Cannot do precalculations.")

    def precalculations(self):
        """
        Precalculations for the reconstruction.
        """
        if self.first_image is None:
            raise ValueError("No first image given. Cannot do precalculations.")

        # Move the first image to the device
        self.first_image = self.first_image.to(self.device)

        # Subtract the mean of the first image
        self.first_image = self.first_image - self.first_image.mean()

        #Get the size of the image
        self.xr, self.yr = self.first_image.shape
        
        #Set the filter radius if not set
        if not isinstance(self.filter_radius, (int, torch.uint8)):
            self.filter_radius = int(min(torch.tensor([self.xr, self.yr])).item() / 5)

        #Generate coordinates for full frame
        x = torch.arange(self.xr, device=self.first_image.device) - self.xr // 2
        y = torch.arange(self.yr, device=self.first_image.device) - self.yr // 2

        self.X, self.Y = torch.meshgrid(x, y, indexing='ij')
        self.position_matrix = torch.sqrt(self.X**2 + self.Y**2) #Distance from center
        
        #Cropping the field to avoid edge effects
        self.xrc = self.xr - self.crop*2
        self.yrc = self.yr - self.crop*2

        if self.crop > 0:
            xc = torch.arange(self.xrc, device=self.first_image.device) - self.xrc // 2
            yc = torch.arange(self.yrc, device=self.first_image.device) - self.yrc // 2
            self.Y_c, self.X_c = torch.meshgrid(xc, yc, indexing='ij')
        else:
            self.X_c = self.X
            self.Y_c = self.Y
        
        # kx and ky is the wave vector that defines the direction of propagation. Used to calculate the fourier shift.
        kx = torch.linspace(-torch.pi, torch.pi, self.xr) 
        ky = torch.linspace(-torch.pi, torch.pi, self.yr)
        self.KX, self.KY = torch.meshgrid(kx, ky, indexing='ij')

        #Create the masks
        self.masks_precalculate()

        # Initialize FourierPeakFinder with the required arguments. This class is used to find the peak coordinates.
        self.FPF = FourierPeakFinder(
            self.position_matrix, 
            self.filter_radius, 
            self.correct_fourier_peak, 
            self.KX, 
            self.KY, 
            self.X, 
            self.Y
            )

        #Find the peak coordinates
        self.kx_add_ky, self.dist_peak = self.FPF.find_peak_coordinates(self.first_image)
        self.kx_add_ky.to(self.device)
        
        #Calculate the fft of the first image
        self.fftIm2 = torch.fft.fftshift(
            torch.fft.fft2(
            (self.first_image) * torch.exp(1j*(self.kx_add_ky)))
            ) * self.mask_list[0]
        
        # Initialize the PolynomialFitter class. This class is used to fit a polynomial to the phase.
        #self.PF = PolynomialFitter(order=self.poly_dims, shape=self.first_image.shape, device=self.device)
        #self.PF = PolynomialFitterV2(input_shape=self.first_image.shape)

        # Calculate the first field and phase
        self.first_field = torch.fft.ifft2(torch.fft.fftshift(self.fftIm2)).to(self.device)
        self.first_phase = torch.angle(self.first_field).to(self.device)

        #First phase background
        #self.phase_background, self.first_phase_sub = self.PF.fit_and_subtract_phase_background_torch(self.first_phase)
        self.phase_background = torch.mean(self.first_phase)
        #self.phase_background = self.PF.correct_phase_4order(self.first_phase)

        #First corrected field
        self.first_field_corrected = torch.exp(-1j * self.phase_background) * self.first_field

    def masks_precalculate(self):
        """
        Precalculate the masks for the reconstruction. The first mask is for the area of which we extract information from the hologram.
        Is set in the filter_radius. The 2, 3 are for lowpass filtering the phase.
        The 2, 3 are for lowpass filtering the phase.
        """

        self.mask_list = []
        if self.mask_radiis is not None:
            for i, rad_curr in enumerate(self.mask_radiis):

                if i == 0:
                    if self.mask_case == 'ellipse':
                        m = OU.create_ellipse_mask(self.xr, self.yr, percent=self.filter_radius/self.xr)
                    elif self.mask_case == 'circular':
                        m = OU.create_circular_mask(self.xr, self.yr, radius=self.filter_radius)
                else:
                    if self.mask_case == 'ellipse':
                        m = OU.create_ellipse_mask(self.xrc, self.yrc, percent=rad_curr/self.xrc)
                    elif self.mask_case == 'circular':
                        m = OU.create_circular_mask(self.xrc, self.yrc, radius=rad_curr)

                self.mask_list.append(m.to(self.device))

            #Set the rad for the mask. Important for storing the fft later on.
            self.rad = self.mask_radiis[0]

        else:
            self.rad = self.filter_radius
            if self.mask_case == 'ellipse':
                m = OU.create_ellipse_mask(self.xr, self.yr, percent=self.rad/self.yr)

            elif self.mask_case == 'circular':
                m = OU.create_circular_mask(self.xr, self.yr, radius=self.rad)

            self.mask_list.append(m.to(self.device))



    def forward(self, holograms):
        """
        Forward pass of the model.

        Parameters:
        - holograms (torch.Tensor): Holograms to reconstruct.
        """

        #Subtract the mean
        #holograms = holograms - holograms.mean()

        #Matrix to store the field in
        reconstructed_fields = torch.zeros(
            holograms.shape[0], self.xrc, self.yrc, device=self.device
            ).type(torch.complex64)

        print(f"Reconstructing {holograms.shape[0]} holograms.")
        for i, holo in enumerate(holograms):

            #Subtract the mean
            holo = holo - holo.mean()

            #Compute the 2-dimensional discrete Fourier Transform with offset image.
            if not self.recalculate_offset:
                fftImage = torch.fft.fftshift(
                    torch.fft.fft2(
                        holo * torch.exp(1j*(self.kx_add_ky))
                        )) * self.mask_list[0]
            else:
                #Find the peak coordinates
                self.kx_add_ky, _ = self.FPF.find_peak_coordinates(holo)
                fftImage = torch.fft.fftshift(
                    torch.fft.fft2(
                        holo * torch.exp(1j*(self.kx_add_ky))
                        )) * self.mask_list[0]

            #Inverse 2-dimensional discrete Fourier Transform
            fftImage2 = torch.fft.ifft2(torch.fft.fftshift(fftImage))

            #If we use the previous phase background to correct the phase first.
            #if self.phase_background is not None:
            #    reconstructed_fields[i] = reconstructed_fields[i] * torch.exp(-1j * self.phase_background)

            #Removes edges in x and y. Some edge effects
            if self.crop > 0:
                fftImage2 = fftImage2[self.crop:-self.crop, self.crop:-self.crop]
            reconstructed_fields[i] = fftImage2

            #Lowpass filtered phase
            if self.lowpass_filtered_phase is not None:
                phase_img = OU.phase_frequencefilter(reconstructed_fields[i], mask=self.mask_list[1], is_field=True)
            else:
                phase_img = torch.angle(reconstructed_fields[i])

            #Correct the field with the phase background
            reconstructed_fields[i] = reconstructed_fields[i] * torch.exp(-1j * phase_img)

            if self.phase_corrections > 0:
                for _ in range(self.phase_corrections):
                    if self.lowpass_filtered_phase is not None:
                        phase_img = OU.phase_frequencefilter(reconstructed_fields[i], mask=self.mask_list[2], is_field=True)

                        reconstructed_fields[i] = reconstructed_fields[i] * torch.exp(-1j * phase_img)

            #Correct reconstructed_fields[i]
            reconstructed_fields[i] = reconstructed_fields[i] * torch.exp(-1j * torch.mean(torch.angle(reconstructed_fields[i])))

            #Save the fft instead of the field if fft_save is True 
            if self.fft_save:
                reconstructed_fields = FL.field_to_vec_multi(reconstructed_fields, self.rad, mask=self.mask_list[0])

        return reconstructed_fields
