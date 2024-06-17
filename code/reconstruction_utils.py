from itertools import product
import torch
import torch.nn.functional as F

class PolynomialFitter:
    def __init__(
        self, 
        order=4,
        shape=(256, 256),
        device='cuda'
        ):

        self.device = device
        self.order = order
        self.shape = shape  # Shape of the phase data (replace with your actual shape)
        self.G = self.prepare_G_matrix()
        
    def prepare_G_matrix(self):
        """Prepare the G matrix for polynomial fitting."""
        ncols = (self.order + 1) * (self.order + 2) // 2
        G = torch.zeros((self.shape[0] * self.shape[1], ncols), dtype=torch.float32, device=self.device)
        
        x, y = torch.meshgrid(torch.arange(self.shape[0], dtype=torch.float32, device=self.device), 
                              torch.arange(self.shape[1], dtype=torch.float32, device=self.device), 
                              indexing='ij')
        x = x.flatten()
        y = y.flatten()

        index = 0
        for i in range(self.order + 1):
            for j in range(self.order + 1):
                if i + j <= self.order:
                    G[:, index] = (x**i) * (y**j)
                    index += 1
        return G

    def polyfit2d_torch(self, x, y, z):
        """Fit a 2D polynomial of given order to the data using PyTorch."""
        x, y, z = x.flatten(), y.flatten(), z.flatten()
        coeffs = torch.linalg.lstsq(self.G[:], z.unsqueeze(1)).solution.flatten()
        return coeffs

    def polyval2d_torch(self, x, y, coeffs):
        """Evaluate a 2D polynomial of given order with coefficients using PyTorch."""
        ij = list(product(range(self.order + 1), range(self.order + 1)))
        z = torch.zeros_like(x, dtype=coeffs.dtype, device=x.device)
        index = 0
        for i, j in ij:
            if i + j <= self.order:
                z += coeffs[index] * (x**i) * (y**j)
                index += 1
        return z

    def fit_and_subtract_phase_background_torch(self, phase_data, poly_background=None):
        """Fit a polynomial to the phase background and subtract it using PyTorch."""
        device = phase_data.device
        x, y = torch.meshgrid(torch.arange(phase_data.size(0), dtype=torch.float32, device=device), 
                              torch.arange(phase_data.size(1), dtype=torch.float32, device=device), 
                              indexing='ij')
        
        # Fit a 2D polynomial
        if poly_background is None:
            coeffs = self.polyfit2d_torch(x, y, phase_data)
        
            # Evaluate the polynomial on the grid
            poly_background = self.polyval2d_torch(x, y, coeffs)
        
        # Subtract the polynomial background
        corrected_phase = phase_data - poly_background
        
        return corrected_phase, poly_background

    def generate_phase_pattern(self, order=4, freq=0.1, noise_level=0.05):
        """Generate a synthetic phase pattern with polynomial background and sinusoidal components."""
        
        x, y = torch.meshgrid(
            torch.arange(self.shape[0], dtype=torch.float32), 
            torch.arange(self.shape[1], dtype=torch.float32),
            indexing='ij'
            )
        
        # Polynomial components
        poly_background = (
            0.1 * x +
            0.05 * y +
            0.01 * x * y +
            0.001 * x**2 +
            0.002 * y**2 -
            0.005 * x**3 +
            0.003 * x**2 * y -
            0.002 * x * y**2 +
            0.001 * y**3
        )
        
        # Sinusoidal components with varying frequencies
        sinusoidal_component1 = torch.sin(freq * x) + torch.cos(freq * y)
        sinusoidal_component2 = torch.sin(2 * freq * x) + torch.cos(2 * freq * y)
        
        # Combine components with noise
        phase_pattern = poly_background + sinusoidal_component1 + sinusoidal_component2 + noise_level * torch.randn(self.shape, dtype=torch.float32)
        phase_pattern/=phase_pattern.max()

        return phase_pattern

class PolynomialFitterV2:
    def __init__(self, input_shape, device='cuda'):
        self.input_shape = input_shape
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.polynomial = self.get_4th_polynomial(input_shape).to(self.device)
        self.G_matrix = self.get_G_matrix(input_shape).to(self.device)

    def get_4th_polynomial(self, input_shape):
        """
        Function that retrieves the 4th-order polynomial

        Input:
            input_shape : Shape of matrix
        Output :
            Polynomial matrix. 
        """
        yrc, xrc = input_shape

        xc = torch.arange(-xrc / 2 + 0.5, xrc / 2 + 0.5, 1, device=self.device)
        yc = torch.arange(-yrc / 2 + 0.5, yrc / 2 + 0.5, 1, device=self.device)
        Y_c, X_c = torch.meshgrid(yc, xc)

        # 4th order polynomial.
        polynomial = [
            X_c**2,
            X_c * Y_c,
            Y_c**2,
            X_c,
            Y_c,
            X_c**3,
            X_c**2 * Y_c,
            X_c * Y_c**2,
            Y_c**3,
            X_c**4,
            X_c**3 * Y_c,
            X_c**2 * Y_c**2,
            X_c * Y_c**3,
            Y_c**4
        ]

        return torch.stack(polynomial)

    def get_G_matrix(self, input_shape):
        """
        Input:
            input_shape : Shape of matrix
        Output:
            Matrix to store 4-order polynomial. (Not including constant)
        """
        yrc, xrc = input_shape

        xc = torch.arange(-xrc / 2 + 0.5, xrc / 2 + 0.5, 1, device=self.device)
        yc = torch.arange(-yrc / 2 + 0.5, yrc / 2 + 0.5, 1, device=self.device)
        Y_c, X_c = torch.meshgrid(yc, xc)

        # Vectors of equal size. x1 and y1 spatial coordinates
        x1 = 1 / 2 * ((X_c[1:, 1:] + X_c[:-1, :-1])).flatten()
        y1 = 1 / 2 * ((Y_c[1:, 1:] + Y_c[:-1, :-1])).flatten()

        G = torch.zeros((2 * (xrc - 1) * (yrc - 1), 14), device=self.device)  # Matrix to store 4-order polynomial. (Not including constant)
        G_end = G.shape[0]
        # (derivate w.r.t to X and Y respectively.)
        uneven_range = torch.arange(1, G_end + 1, 2, device=self.device)
        even_range = torch.arange(0, G_end, 2, device=self.device)

        G[uneven_range, 4] = 1
        G[even_range, 3] = 1
        G[even_range, 1] = y1
        G[even_range, 0] = 2 * x1
        G[uneven_range, 1] = x1
        G[uneven_range, 2] = 2 * y1

        G[even_range, 5] = 3 * x1**2
        G[even_range, 6] = 2 * x1 * y1
        G[uneven_range, 6] = x1**2
        G[uneven_range, 7] = 2 * x1 * y1
        G[even_range, 7] = y1**2
        G[uneven_range, 8] = 3 * y1**2

        G[even_range, 9] = 4 * x1**3
        G[even_range, 10] = 3 * x1**2 * y1
        G[uneven_range, 10] = x1**3
        G[even_range, 11] = 2 * x1 * y1**2
        G[uneven_range, 11] = 2 * x1**2 * y1
        G[even_range, 12] = y1**3
        G[uneven_range, 12] = 3 * x1 * y1**2
        G[uneven_range, 13] = 4 * y1**3

        return G

    def correct_phase_4order(self, phase_img):
        """ 
        Calculates the coefficients (4th order) by taking the derivative of phase image to fit a phase background. 

        Input: 
            phase_img : phase image
        Output: 
            Phase background fit
        """
        An0 = phase_img.clone()

        #Move to device
        An0 = An0.to(self.device)

        yr, xr = An0.shape

        An0 = An0 - An0[0, 0]  # Set phase to "0"

        # Derivative the phase to handle the modulus
        dx = -torch.pi + (torch.pi + torch.diff(An0, dim=0)) % (2 * torch.pi)
        dy = -torch.pi + torch.transpose((torch.pi + torch.diff(torch.transpose(An0, 0, 1), dim=0)) % (2 * torch.pi), 0, 1)

        # dx1, dy1 the derivatives
        dx1 = 1 / 2 * ((dx[:, 1:] + dx[:, :-1])).flatten()
        dy1 = 1 / 2 * ((dy[1:, :] + dy[:-1, :])).flatten()

        # (derivate w.r.t to X and Y respectively.) Each factor have a constant b_i to be fitted later on.
        G_end = self.G_matrix.shape[0]
        uneven_range = torch.arange(1, G_end + 1, 2, device=self.device)
        even_range = torch.arange(0, G_end, 2, device=self.device)

        dt = torch.zeros(2 * (xr - 1) * (yr - 1), device=self.device)
        dt[even_range] = dx1
        dt[uneven_range] = dy1

        # Here the coefficients to the polynomial are calculated. Note that np.linalg.lstsq(b,B)[0] is equivalent to \ in MATLAB              
        R = torch.linalg.cholesky(torch.matmul(torch.transpose(self.G_matrix, 0, 1), self.G_matrix))

        # Equivalent to R\(R'\(G'*dt))
        b = torch.linalg.solve(R, torch.linalg.solve(R.t(), torch.matmul(self.G_matrix.t(), dt)))

        # Phase background is defined by the 4th order polynomial with the fitted parameters.
        phase_background = torch.zeros_like(self.polynomial[0], device=self.device)
        for i, factor in enumerate(self.polynomial):
            phase_background += b[i] * factor

        return phase_background

class PhaseFrequencyFilter:
    def __init__(
        self,
        crop=0,
        ):
        self.crop = crop
    
    def filter(self, field, mask, is_field=True):
        """
        Lowpass filter the image with mask defined in input.

        Input:
            field : Complex valued field (torch.Tensor).
            is_field : if input is a field, or if it needs to be Fourier transformed before.
        Output:
            phase_img : phase of optical field (torch.Tensor).
        """
        if not isinstance(field, torch.Tensor):
            raise TypeError("Input 'field' must be a torch.Tensor.")
        
        if not isinstance(mask, torch.Tensor):
            raise TypeError("Input 'mask' must be a torch.Tensor.")
        
        if field.shape != mask.shape:
            raise ValueError("Inputs 'field' and 'mask' must have the same shape.")
        
        if is_field:
            freq = torch.fft.fftshift(torch.fft.fft2(field))
        else:
            freq = field
            
        #Construct low-pass mask
        freq_low = freq * mask
        
        #Shift the zero-frequency component to the center of the spectrum and compute inverse FFT
        E_field = torch.fft.ifft2(torch.fft.ifftshift(freq_low))
        
        if self.crop > 0:
            phase_img = torch.angle(E_field)[self.crop:-self.crop, self.crop:-self.crop]
        else:
            phase_img = torch.angle(E_field)
        
        return phase_img

class FourierPeakFinder:
    def __init__(
        self, 
        position_matrix, 
        filter_radius, 
        correct_fourier_peak, 
        KX, 
        KY, 
        X, 
        Y
        ):
        """
        Initialize FourierPeakFinder class.

        Args:
        - position_matrix (torch.Tensor): Distance matrix from the center.
        - filter_radius (int): Radius for filtering.
        - correct_fourier_peak (tuple): Correction values for Fourier peak.
        - KX (torch.Tensor): KX matrix (assumed to be defined similarly to X).
        - KY (torch.Tensor): KY matrix (assumed to be defined similarly to Y).
        - X (torch.Tensor): X matrix.
        - Y (torch.Tensor): Y matrix.
        """
        self.position_matrix = position_matrix
        self.filter_radius = filter_radius
        self.correct_fourier_peak = correct_fourier_peak
        self.KX = KX
        self.KY = KY
        self.X = X
        self.Y = Y

    def find_peak_coordinates(self, frame):
        """
        Find peak coordinates in Fourier space for a given frame.

        Args:
        - frame (torch.Tensor): Input tensor (2D) representing the frame.

        Returns:
        - Tuple: (kx_add_ky, dist_peak) where:
          - kx_add_ky (torch.Tensor): Resulting tensor from computing kx_pos * X + ky_pos * Y.
          - dist_peak (torch.Tensor): Distance of the peak from the center.
        """
        xr, yr = frame.shape

        # Compute 2D Fourier transform
        fftImage = torch.fft.fft2(frame)

        # Shift zero-frequency component to the center of the spectrum
        fftImage = torch.fft.fftshift(fftImage)

        # Apply filters to fftImage
        fftImage = torch.where(self.position_matrix < self.filter_radius, torch.tensor(0, dtype=fftImage.dtype, device=fftImage.device), fftImage)
        
        #Set fftImage to zero from 0 to middle of the image
        fftImage[:, :xr//2] = 0

        # Compute magnitude of the fftImage
        fftImage = torch.abs(fftImage)

        # Apply Gaussian filter to the fftImage
        fftImage = F.conv2d(fftImage.real.unsqueeze(0).unsqueeze(0), torch.ones(1, 1, 7, 7, device=fftImage.device) / 49, padding=1).squeeze()

        #Find the coordinates of the maximum values in the fftImage
        max_index = torch.argmax(fftImage)
        max_coords = torch.unravel_index(max_index, fftImage.shape)
        max_coords = (max_coords[0], max_coords[1])

        # Extract coordinates from X and Y matrices
        x_pos = self.position_matrix[max_coords]
        y_pos = self.position_matrix[max_coords]
        dist_peak = torch.sqrt(x_pos**2 + y_pos**2)

        # Assuming KX and KY are defined similarly to X and Y
        kx_pos = self.KX[max_coords]
        ky_pos = self.KY[max_coords]

        # Compute kx_add_ky
        kx_add_ky = kx_pos * self.X + ky_pos * self.Y

        return kx_add_ky, dist_peak

# Example usage:
if __name__ == "__main__":
    # Example data (replace with your actual data)
    
    filter_radius = 20
    correct_fourier_peak = (0, 0)
    X = torch.randn(256, 256)  # Example X matrix (replace with actual X and Y matrices)
    Y = torch.randn(256, 256)  # Example Y matrix
    position_matrix = torch.sqrt(X**2 + Y**2)  # Example position_matrix (computed previously)
    KX = torch.randn(X.shape)  # Example KX (replace with actual KX and KY)
    KY = torch.randn(Y.shape)  # Example KY

    # Initialize FourierPeakFinder
    finder = FourierPeakFinder(position_matrix, filter_radius, correct_fourier_peak, KX, KY, X, Y)

    # Example frame (replace with actual frame)
    frame = torch.randn(256, 256)*5

    # Find peak coordinates in Fourier space
    kx_add_ky, dist_peak = finder.find_peak_coordinates(frame)

    # Use kx_add_ky and dist_peak as needed in your further processing
    print(kx_add_ky.shape)



# Example usage
if __name__ == "__main__":
    # Create an instance of PolynomialFitter
    fitter = PolynomialFitter(order=5)
    
    # Generate example phase data (replace with your actual data)
    phase_data = fitter.generate_phase_pattern(order=5, freq=0.1, noise_level=0.05)
    phase_data.requires_grad = False  # Ensure phase_data does not require gradients
    
    # Fit and subtract the polynomial background
    corrected_phase, poly_background = fitter.fit_and_subtract_phase_background_torch(phase_data)
    
    # Visualize the results
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(phase_data.numpy(), cmap='viridis')
    plt.title('Original Phase Data')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(poly_background.numpy(), cmap='viridis')
    plt.title('Fitted Polynomial Background')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(corrected_phase.numpy(), cmap='viridis')
    plt.title('Corrected Phase Data')
    plt.colorbar()

    plt.tight_layout()
    plt.show()
