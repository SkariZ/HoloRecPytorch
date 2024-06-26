import torch

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
            self.padding = image_size // 2

        #Wavevector
        self.k = 2 * torch.pi / self.wavelength * self.ri_medium

        # Precalculate K and C
        self.precalculate_KC()

        if zv is not None:
            self.precalculate_Tz()

    def forward(self, x):

        #Check if x is a tensor
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        #Check so zv is defined
        if self.zv is None or isinstance(self.zv, int):
            raise ValueError('zv is not defined. Please define zv to propagate the field.')

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

    def precalculate_Tz(self):
        """
        Precalculate the Tz matrix for the propagator.
        """

        self.Tz = torch.tensor([self.C*torch.fft.fftshift(torch.exp(self.k * 1j*z*(self.K-1))) for z in self.zv], device=self.device)

    def precalculate_KC(self):
        """
        Precalculate the K and C matrix for the propagator.
        """

        xr, yr = self.image_size

        x = 2 * torch.pi/(self.pixel_size) * torch.arange(-(xr/2-1/2), (xr/2 + 1/2), 1) / xr
        y = 2 * torch.pi/(self.pixel_size) * torch.arange(-(yr/2-1/2), (yr/2 + 1/2), 1) / yr

        KXk, KYk = torch.meshgrid(x, y, indexing='ij', device=self.device)

        self.K = torch.real(torch.sqrt(torch.tensor(1 -(KXk/self.k)**2 - (KYk/self.k)**2 , dtype = torch.complex64, device=self.device)))
        self.C = ((KXk/self.k)**2 + (KYk/self.k)**2 < 1).float().to(self.device)

    def find_focus(self, x):
        pass
