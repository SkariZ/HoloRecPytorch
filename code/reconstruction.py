import torch
import torch.nn as nn
import torch.nn.functional as F

import other_utils as OU
import fft_loader as FL

from reconstruction_utils import FourierPeakFinder, PolynomialFitterV2 #PolynomialFitter, PhaseFrequencyFilter,


class HolographicReconstruction(nn.Module):
    def __init__(
            self,
            image_size=(1450, 1930),
            do_precalculations=False,
            first_image=None,
            crop=0,
            lowpass_filtered_phase=None,
            filter_radius=None,
            correct_fourier_peak=(0, 0),
            mask_radiis=None,
            mask_case="ellipse",
            recalculate_offset=True,
            phase_corrections=3,
            skip_background_correction=False,
            fft_save=False,
            correct_field=True,
            kernel_size=27,
            sigma=9
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
        self.mask_radiis = mask_radiis
        self.mask_case = mask_case
        self.recalculate_offset = recalculate_offset
        self.phase_corrections = phase_corrections
        self.skip_background_correction = skip_background_correction
        self.fft_save = fft_save
        self.correct_field = correct_field
        self.kernel_size = kernel_size
        self.sigma = sigma

        # Do precalculations
        if do_precalculations:
            self.precalculations() if first_image is not None else print("No first image given. Cannot do precalculations.")

    def get_settings(self):
        "Get the settings of the reconstruction, in a dictionary."

        settings = {
            "image_size": self.image_size,
            "crop": self.crop,
            "lowpass_filtered_phase": self.lowpass_filtered_phase,
            "filter_radius": self.filter_radius,
            "correct_fourier_peak": self.correct_fourier_peak,
            "mask_radiis": self.mask_radiis,
            "mask_case": self.mask_case,
            "recalculate_offset": self.recalculate_offset,
            "phase_corrections": self.phase_corrections,
            "skip_background_correction": self.skip_background_correction,
            "fft_save": self.fft_save,
            "correct_field": self.correct_field,
            "kernel_size": self.kernel_size,
            "sigma": self.sigma
            }

        return settings

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
        if not isinstance(self.filter_radius, (int, float)):
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

        self.PF = PolynomialFitterV2(
            shape=(self.xrc, self.yrc), 
            device=self.device
            )

        #Find the peak coordinates
        self.kx_add_ky, self.dist_peak = self.FPF.find_peak_coordinates(self.first_image)
        self.kx_add_ky.to(self.device)
        
        #Calculate the fft of the first image
        self.fftIm2 = torch.fft.fftshift(
            torch.fft.fft2(
            (self.first_image) * torch.exp(1j*(self.kx_add_ky)))
            ) * self.mask_list[0]
        
        # Calculate the first field and phase
        self.first_field = torch.fft.ifft2(torch.fft.fftshift(self.fftIm2)).to(self.device)
        self.first_phase = torch.angle(self.first_field).to(self.device)

        if not self.skip_background_correction:
            #Lowpass filtered phase
            if self.lowpass_filtered_phase and len(self.mask_list) > 1:
                field = OU.phase_frequencefilter(self.first_field, mask=self.mask_list[1], is_field=True, return_phase=False)
            else:
                field = self.first_field
            
            self.phase_img_smooth = self.PF.correct_phase_4order(torch.angle(field))

            # Apply Gaussian smoothing
            #real_smoothed = F.conv2d(field.real.unsqueeze(0).unsqueeze(0), self.kernel, padding=self.padding)
            #imag_smoothed = F.conv2d(field.imag.unsqueeze(0).unsqueeze(0), self.kernel, padding=self.padding)

            # Combine the smoothed real and imaginary parts 
            #self.phase_img_smooth = torch.angle(real_smoothed + 1j * imag_smoothed)

            #Correct the field with the phase background
            self.first_field_corrected = self.first_field * torch.exp(-1j * self.phase_img_smooth)

            if self.phase_corrections > 0:
                for _ in range(self.phase_corrections):
                    if self.lowpass_filtered_phase and len(self.mask_list) > 2:
                        field = OU.phase_frequencefilter(self.first_field_corrected, mask=self.mask_list[2], is_field=True, return_phase=False)

                        # Apply Gaussian smoothing
                        #real_smoothed = F.conv2d(field.real, self.kernel_lowpass, padding=self.padding)
                        #imag_smoothed = F.conv2d(field.imag, self.kernel_lowpass, padding=self.padding)

                        #Combine the smoothed real and imaginary parts
                        #phase_img_smooth = torch.angle(real_smoothed + 1j * imag_smoothed)

                        phase_img_smooth = self.PF.correct_phase_4order(torch.angle(field))

                        #Correct the field with the phase background
                        self.first_field_corrected = self.first_field_corrected * torch.exp(-1j * phase_img_smooth)

            #Squeeze the field
            self.first_field_corrected = self.first_field_corrected.squeeze(0).squeeze(0)

        else:
            self.phase_img_smooth = self.first_phase
            self.first_field_corrected = self.first_field

        #Correct reconstructed_fields[i] with the mean of the phase
        self.first_field_corrected = self.first_field_corrected * torch.exp(-1j * torch.mean(torch.angle(self.first_field_corrected)))

    def masks_precalculate(self):
        """
        Precalculate the masks for the reconstruction. The first mask is for the area of which we extract information from the hologram.
        Is set in the filter_radius. The 2, 3 are for lowpass filtering the phase.
        The 2, 3 are for lowpass filtering the phase.
        """

        self.mask_list = []

        if self.mask_case == 'ellipse':
            m = OU.create_ellipse_mask(self.xr, self.yr, percent=self.filter_radius/self.xr)
        elif self.mask_case == 'circular':
            m = OU.create_circular_mask(self.xr, self.yr, radius=self.filter_radius)

        #Append the mask to the list
        self.mask_list.append(m.to(self.device))

        #Set the rad for the mask. Important for storing the fft later on.
        self.rad = self.filter_radius

        #Lowpass filter the phase-masks
        if self.mask_radiis is not None and self.lowpass_filtered_phase:
            for _, rad_curr in enumerate(self.mask_radiis):
                if self.mask_case == 'ellipse':
                    m = OU.create_ellipse_mask(self.xrc, self.yrc, percent=rad_curr/self.xrc)
                elif self.mask_case == 'circular':
                    m = OU.create_circular_mask(self.xrc, self.yrc, radius=rad_curr)

                #Append the mask to the list
                self.mask_list.append(m.to(self.device))

        #Lowpass filter the phase-masks later on
        self.kernel = self.gaussian_kernel(self.kernel_size, self.sigma).to(self.device)
        self.kernel_lowpass = self.gaussian_kernel(self.kernel_size, self.sigma*2).to(self.device)
        self.padding = (self.kernel_size - 1) // 2


    def gaussian_kernel(self, size, sigma):
        """Function to create a Gaussian kernel."""

        coords = torch.arange(size) - size // 2
        x, y = torch.meshgrid(coords, coords, indexing='ij')
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        # Add a channel dimension
        kernel = kernel.view(1, 1, size, size)
        return kernel


    def forward(self, holograms):
        """
        Forward pass of the reconstruction

        Parameters:
        - holograms (torch.Tensor): Holograms to reconstruct.
        """

        #Check if holograms need to be cropped
        if holograms.shape[1] != self.xr or holograms.shape[2] != self.yr:
            holograms = holograms[:, :self.xr, :self.yr]

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

            #Removes edges in x and y. Some edge effects
            if self.crop > 0:
                fftImage2 = fftImage2[self.crop:-self.crop, self.crop:-self.crop]
            reconstructed_fields[i] = fftImage2

            if not self.skip_background_correction:
                #Lowpass filtered phase
                if self.lowpass_filtered_phase and len(self.mask_list) > 1:
                    field = OU.phase_frequencefilter(reconstructed_fields[i], mask=self.mask_list[1], is_field=True, return_phase=False)
                else:
                    field = reconstructed_fields[i]

                # Apply Gaussian smoothing
                #real_smoothed = F.conv2d(field.real.unsqueeze(0).unsqueeze(0), self.kernel, padding=self.padding)
                #imag_smoothed = F.conv2d(field.imag.unsqueeze(0).unsqueeze(0), self.kernel, padding=self.padding)

                # Combine the smoothed real and imaginary parts
                #phase_img_smooth = torch.angle(real_smoothed + 1j * imag_smoothed)
                phase_img_smooth = self.PF.correct_phase_4order(torch.angle(field))

                #Correct the field with the phase background
                reconstructed_fields[i] = reconstructed_fields[i] * torch.exp(-1j * phase_img_smooth)

                if self.lowpass_filtered_phase and len(self.mask_list) > 2:
                    for _ in range(self.phase_corrections):
                            field = OU.phase_frequencefilter(reconstructed_fields[i], mask=self.mask_list[2], is_field=True, return_phase=False)

                            # Apply Gaussian smoothing
                            #real_smoothed = F.conv2d(field.real.unsqueeze(0).unsqueeze(0), self.kernel_lowpass, padding=self.padding)
                            #imag_smoothed = F.conv2d(field.imag.unsqueeze(0).unsqueeze(0), self.kernel_lowpass, padding=self.padding)

                            # Combine the smoothed real and imaginary parts
                            #phase_img_smooth = torch.angle(real_smoothed + 1j * imag_smoothed)
                            phase_img_smooth = self.PF.correct_phase_4order(torch.angle(field))

                            #Correct the field with the phase background
                            reconstructed_fields[i] = reconstructed_fields[i] * torch.exp(-1j * phase_img_smooth)

            #Correct reconstructed_fields[i] with the mean of the phase
            reconstructed_fields[i] = reconstructed_fields[i] * torch.exp(-1j * torch.mean(torch.angle(reconstructed_fields[i])))

        if self.correct_field:
            for i in range(reconstructed_fields.shape[0]):
                reconstructed_fields[i] = OU.correctfield(reconstructed_fields[i], n_iter=3)
        
        #Save the fft instead of the field if fft_save is True 
        if self.fft_save:
            reconstructed_fields = FL.field_to_vec_multi(reconstructed_fields, self.rad, mask=self.mask_list[0])

        return reconstructed_fields
    
    def load_fft(self, ffts):
        """
        Load ffts and reconstruct the fields.

        Parameters:
        - ffts (torch.Tensor): FFTs to reconstruct.
        """
        return FL.vec_to_field_multi(ffts, self.rad, shape=(self.xrc, self.yrc), mask=self.mask_list[0])

