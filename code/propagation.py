import torch
import torch.nn.functional as F

class Propagator():
    def __init__(
            self, 
            image_size,
            padding=None,
            wavelength=0.532, 
            pixel_size=0.114,
            ri_medium=1.33,
            zv=None
            ):
        
        # Parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.image_size = image_size
        self.padding = padding
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.ri_medium = ri_medium
        self.zv = zv

        # Check if zv is a list
        if isinstance(zv, int):
            self.zv = [zv]

        # Padding to avoid edge effects
        if self.padding is None:
            self.padding = image_size[0] // 2

        #Wavevector
        self.k = 2 * torch.pi / self.wavelength * self.ri_medium

        # Precalculate K and C
        self.precalculate_KC()

        if zv is not None:
            self.precalculate_Tz()

    def forward(self, x, Z=None):
        """
        Propagate the field x

        Input:
            x : Field to propagate
            Z : Propagation distance. If None, the default propagation distance is used.
        Output:
            Propagated field.

        """
        #Check if Z is defined. This allows for changing the propagation distance.
        if Z is not None:
            self.zv = Z
            self.precalculate_Tz()

        #Check if x is a tensor
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        #Check if x has two channels
        if x.shape[-1] == 2:
            x = x[..., 0] + 1j*x[..., 1]

        #Check so x is complex
        elif not torch.is_complex(x):
            x = x.to(torch.complex64)

        #Check so zv is defined
        if self.zv is None or isinstance(self.zv, int):
            raise ValueError('zv is not defined. Please define zv to propagate the field.')

        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))

        #Fourier transform of field
        f1 = torch.fft.fft2(x)

        propagations = torch.zeros((len(self.zv), *self.image_size), dtype=torch.complex64, device=self.device)

        for i, z_prop in enumerate(self.zv):
            #Propagate f1 by z_prop
            refocused = torch.fft.ifft2(self.Tz[i]*f1)
        
            if self.padding > 0:
                refocused = refocused[self.padding:-self.padding, self.padding:-self.padding]

            propagations[i] = refocused

        return propagations
    
    def forward_fields(self, x, Z=None):
        """
        Propagate one or more fields x by a single Z value.

        Inputs:
            x : torch.Tensor
                Field(s) to propagate. Shape can be (H, W) for a single field
                or (N, H, W) for multiple fields.
            Z : float
                Propagation distance. If None, uses self.zv.

        Returns:
            torch.Tensor
                Propagated field(s) with same batch dimensions as input.
        """
        
        if Z is not None:
            self.zv = [Z]  # Store as a list for compatibility
            self.precalculate_Tz()

        # Ensure x is a complex torch tensor
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.complex64, device=self.device)
        elif not torch.is_complex(x):
            x = x.to(torch.complex64)

        # Add batch dimension if single field
        single_field = False
        if x.ndim == 2:
            x = x.unsqueeze(0)  # shape (1, H, W)
            single_field = True

        N, H, W = x.shape

        # Pad each field if needed
        if self.padding > 0:
            x = torch.nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding))

        # FFT of each field
        f1 = torch.fft.fft2(x)

        # Prepare output tensor
        propagated = torch.zeros_like(x, dtype=torch.complex64)

        # Propagate using the first (and only) Z value
        refocused = torch.fft.ifft2(self.Tz[0] * f1)

        # Remove padding
        if self.padding > 0:
            refocused = refocused[..., self.padding:-self.padding, self.padding:-self.padding]

        propagated = refocused

        # Remove batch dim if single field
        if single_field:
            propagated = propagated.squeeze(0)

        return propagated

    def precalculate_Tz(self):
        """
        Precalculate the Tz matrix for the propagator.
        """

        self.Tz = torch.stack([self.C*torch.fft.fftshift(torch.exp(self.k * 1j*z*(self.K-1))) for z in self.zv])
        
        #set nan values to 0
        self.Tz[torch.isnan(self.Tz)] = 0

    def precalculate_KC(self):
        """
        Precalculate the K and C matrix for the propagator.
        """

        xr, yr = self.image_size

        if self.padding > 0:
            xr += 2*self.padding
            yr += 2*self.padding

        x = 2 * torch.pi / self.pixel_size * torch.arange(-(xr/2 - 0.5), (xr/2 + 0.5), 1, device=self.device) / xr
        y = 2 * torch.pi / self.pixel_size * torch.arange(-(yr/2 - 0.5), (yr/2 + 0.5), 1, device=self.device) / yr

        KXk, KYk = torch.meshgrid(x, y, indexing='ij')

        #Calculate K matrix. Make it complex for multiplication later
        self.K = torch.sqrt(1 - (KXk / self.k)**2 - (KYk / self.k)**2).real.to(torch.complex64)

        # Create a circular disk here.
        self.C = torch.fft.fftshift(((KXk / self.k)**2 + (KYk / self.k)**2 < 1).float())


    def find_focus_field(self, x, Z=None, crop_size=None, criterion='max', criterion_pre="abs", crit_max=True, return_crit=False, moving_avg=None):

        #Check if Z is defined. This allows for changing the propagation distance.
        if Z is not None:
            self.zv = Z
            self.precalculate_Tz()

        #Check if x is a tensor
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        #Check so zv is defined
        if self.zv is None or isinstance(self.zv, int):
            raise ValueError('zv is not defined. Please define zv to propagate the field.')
        
        #Propagate the field
        propagations = self.forward(x)

        #Save the propagated fields
        propagations_pre = propagations.clone()

        #Find the focus field
        if crop_size is not None:
            crop_size = crop_size // 2
            propagations = propagations[:,
                self.image_size[0]//2-crop_size:self.image_size[0]//2+crop_size, 
                self.image_size[1]//2-crop_size:self.image_size[1]//2+crop_size
                ]
        
        #Criterion preprocessing
        if criterion_pre == 'abs':
            propagations = torch.abs(propagations)
        elif criterion_pre == 'real':
            propagations = propagations.real
        elif criterion_pre == 'imag':
            propagations = propagations.imag
        elif criterion_pre == 'phase':
            propagations = torch.angle(propagations)
            propagations = torch.remainder(propagations, 2*torch.pi)
        elif criterion_pre == 'sobel':
            kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
            propagations = torch.stack([F.conv2d(torch.abs(prop.unsqueeze(0)), kernel, padding=1) for prop in propagations]).squeeze(1)
        elif criterion_pre == 'laplace':
            kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
            propagations = torch.stack([F.conv2d(torch.abs(prop.unsqueeze(0)), kernel, padding=1) for prop in propagations]).squeeze(1)

        #Criterion to find the best focus
        if criterion == 'max':
            crit_vals = torch.stack([prop.max() for prop in propagations])
        elif criterion == 'sum':
            crit_vals = torch.stack([prop.sum() for prop in propagations])
        elif criterion == 'mean':
            crit_vals = propagations.mean(dim=(1, 2))
        elif criterion == 'std':
            crit_vals = propagations.std(dim=(1, 2))
        elif criterion == 'fftmax' and criterion_pre is None:
            crit_vals = torch.stack([torch.abs(torch.fft.fftshift(torch.fft.fft2(prop))).max() for prop in propagations])
        else:
            raise ValueError('Criterion not implemented.')

        crit_vals = crit_vals.real

        #Moving average
        if moving_avg is not None:
            crit_vals = torch.nn.functional.avg_pool1d(crit_vals.unsqueeze(0), moving_avg, stride=1).squeeze(0)

        if crit_max:
            best_focus = torch.argmax(crit_vals)
        else:
            best_focus = torch.argmin(crit_vals)
    	
        #Return the best focus field
        best_field = propagations_pre[best_focus]

        if return_crit:
            return best_field, crit_vals
        else:
            return best_field
