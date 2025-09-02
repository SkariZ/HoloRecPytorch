from itertools import product
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class PolynomialFitter:
    def __init__(self, order=4, shape=(256, 256), device='cuda'):
        self.device = device
        self.order = order
        self.shape = shape  # Shape of the phase data (replace with your actual shape)

        # Prepare the spatial coordinates
        self.x, self.y = torch.meshgrid(
            torch.arange(self.shape[0], dtype=torch.float32, device=self.device),
            torch.arange(self.shape[1], dtype=torch.float32, device=self.device),
            indexing='ij'
        )
        self.x = (self.x - self.shape[0] // 2) / (self.shape[0] // 2)
        self.y = (self.y - self.shape[1] // 2) / (self.shape[1] // 2)

        # Prepare the G matrix for polynomial fitting
        self.G = self.prepare_G_matrix()

        # Coefficients of the polynomial
        self.coeffs = None

    def prepare_G_matrix_old(self):
        """Prepare the G matrix for polynomial fitting."""
        ncols = (self.order + 1) * (self.order + 2) // 2
        G = torch.zeros((self.shape[0] * self.shape[1], ncols), dtype=torch.float32, device=self.device)

        x = self.x.flatten()
        y = self.y.flatten()

        index = 0
        for i in range(self.order + 1):
            for j in range(self.order + 1):
                if i + j <= self.order:
                    G[:, index] = (x**i) * (y**j)
                    index += 1
        return G
    
    def polyfit2d_torch(self, z):
        """Fit a 2D polynomial of given order to the data using PyTorch."""
        z = z.flatten().unsqueeze(1)
        self.coeffs = torch.linalg.lstsq(self.G, z).solution.flatten()
        return self.coeffs

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

    def compute_2d_gradients(self, phase):
        """Computes the gradients in x and y directions for the phase image."""
        # Derivative the phase to handle the modulus
        dx = -torch.pi + (torch.pi + torch.diff(phase, dim=0)) % (2 * torch.pi)
        dy = -torch.pi + torch.transpose((torch.pi + torch.diff(torch.transpose(phase, 0, 1), dim=0)) % (2 * torch.pi), 0, 1)

        # Add 1st row and column
        grad_x = torch.zeros_like(phase)
        grad_x[1:, :] = dx
        grad_x[0, :] = dx[0, :]
        grad_y = torch.zeros_like(phase)
        grad_y[:, 1:] = dy
        grad_y[:, 0] = dy[:, 0]

        return grad_x, grad_y

    def integrate_2d_gradients(self, grad_x, grad_y):
        """Integrates 2D gradients to obtain the phase image."""
        integrated_phase = torch.cumsum(grad_x, dim=1) + torch.cumsum(grad_y, dim=0)
        return integrated_phase

    def fit_background(self, phase_data, subtract_mean=True, derivate_phase=True, poly_background=None):
        """Fit a polynomial to the phase background and subtract it using PyTorch."""
        # Subtract the mean
        if subtract_mean:
            phase_data -= phase_data.mean()

        # Derivate the phase to handle the modulus
        if derivate_phase:
            grad_x, grad_y = self.compute_2d_gradients(phase_data)
            grad_x_flat = grad_x.flatten()
            grad_y_flat = grad_y.flatten()

            # Fit a 2D polynomial to the gradients
            coeffs_x = self.polyfit2d_torch(grad_x_flat)
            coeffs_y = self.polyfit2d_torch(grad_y_flat)

            # Evaluate the polynomial gradients
            grad_x_background = self.polyval2d_torch(self.x, self.y, coeffs_x)
            grad_y_background = self.polyval2d_torch(self.x, self.y, coeffs_y)

            # Integrate the gradients to get the polynomial background
            poly_background = self.integrate_2d_gradients(grad_x_background, grad_y_background)
        else:
            if poly_background is None:
                # Fit a 2D polynomial to the phase data
                coeffs = self.polyfit2d_torch(phase_data)
                # Evaluate the polynomial background
                poly_background = self.polyval2d_torch(self.x, self.y, coeffs)

        return poly_background

    def generate_phase_pattern(self, num_waves=3, freq_range=(0.025, 0.1), noise_level=0.05):
        """
        Generate a phase pattern with a few waves and noise.

        Parameters
        ----------
        num_waves : int, optional
            The number of sine waves to combine. Default is 3.
        freq_range : tuple, optional
            The range of frequencies for the sine waves. Default is (0.05, 0.15).
        noise_level : float, optional
            The standard deviation of the random noise. Default is 0.05.

        Returns
        -------
        phase_data : torch.Tensor
            The generated phase pattern.
        """

        phase_data = torch.zeros(self.shape, dtype=torch.float32, device=self.device)
        for _ in range(num_waves):
            freq_x = torch.rand(1).item() * (freq_range[1] - freq_range[0]) + freq_range[0]
            freq_y = torch.rand(1).item() * (freq_range[1] - freq_range[0]) + freq_range[0]
            phase_shift_x = torch.rand(1).item() * 2 * torch.pi
            phase_shift_y = torch.rand(1).item() * 2 * torch.pi
            phase_data += torch.sin(2 * torch.pi * (freq_x * self.x + freq_y * self.y) + phase_shift_x + phase_shift_y)

        # Add noise
        noise = noise_level * torch.randn_like(phase_data, device=self.device)
        phase_data += noise

        #Scale to -pi to pi
        phase_data = (phase_data - phase_data.min()) / (phase_data.max() - phase_data.min()) * 2 * torch.pi - torch.pi

        return phase_data


class PolynomialFitterV2:
    def __init__(self, shape, device='cuda'):
        self.shape = shape
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.x, self.y = torch.meshgrid(
            torch.arange(self.shape[0], dtype=torch.float32, device=self.device),
            torch.arange(self.shape[1], dtype=torch.float32, device=self.device),
            indexing='ij'
        )
        self.x = (self.x-1/2 - self.shape[0] // 2+1/2) #/ (self.shape[0] // 2)
        self.y = (self.y-1/2 - self.shape[1] // 2+1/2) #/ (self.shape[1] // 2)

        self.polynomial = self.get_4th_polynomial().to(self.device)
        self.G_matrix = self.get_G_matrix().to(self.device)

    def get_4th_polynomial(self):
        """
        Function that retrieves the 4th-order polynomial

        Output :
            Polynomial matrix. 
        """

        # 4th order polynomial.
        polynomial = [
            self.x**2,
            self.x * self.y,
            self.y**2,
            self.x,
            self.y,
            self.x**3,
            self.x**2 * self.y,
            self.x * self.y**2,
            self.y**3,
            self.x**4,
            self.x**3 * self.y,
            self.x**2 * self.y**2,
            self.x * self.y**3,
            self.y**4
        ]

        return torch.stack(polynomial)

    def get_G_matrix(self):
        """
        Output:
            Matrix to store 4-order polynomial. (Not including constant)
        """

        # Vectors of equal size. x1 and y1 spatial coordinates
        x1 = 1 / 2 * ((self.x[1:, 1:] + self.x[:-1, :-1])).flatten()
        y1 = 1 / 2 * ((self.y[1:, 1:] + self.y[:-1, :-1])).flatten()

        #G = torch.zeros((2 * (xrc - 1) * (yrc - 1), 14), device=self.device)  # Matrix to store 4-order polynomial. (Not including constant)
        G = torch.zeros((2 * (self.shape[0] - 1) * (self.shape[1] - 1), 14), device=self.device)

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

        dt = torch.zeros(2 * (An0.shape[0] - 1) * (An0.shape[1] - 1), device=self.device)
        dt[even_range] = dx1
        dt[uneven_range] = dy1

        # Here the coefficients to the polynomial are calculated. Note that np.linalg.lstsq(b,B)[0] is equivalent to \ in MATLAB              
        R = torch.linalg.cholesky(torch.matmul(torch.transpose(self.G_matrix, 0, 1), self.G_matrix))
        R = torch.transpose(R, 0, 1)
        Rt = torch.transpose(R, 0, 1)

        # Equivalent to R\(R'\(G'*dt))
        b = torch.linalg.solve(R,
                               torch.linalg.solve(
                                   Rt, torch.matmul(self.G_matrix.t(), dt))
                                )
        self.b=b
        # Phase background is defined by the 4th order polynomial with the fitted parameters.
        phase_background = torch.zeros_like(self.polynomial[0], device=self.device)
        for i, factor in enumerate(self.polynomial):
            phase_background = phase_background + b[i] * factor
        
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
        fftImage = F.conv2d(fftImage.real.unsqueeze(0).unsqueeze(0), torch.ones(1, 1, 7, 7, device=fftImage.device) / 49, padding=3).squeeze()

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

class Polynomial2DModel(nn.Module):
    def __init__(self, 
                 degree_x, 
                 degree_y,
                 shape=None, 
                 device='cuda', 
                 num_epochs=10000, 
                 loss_tolerance=5e-3, 
                 lr=5e-3,
                 patience=5000
                 ):
        super(Polynomial2DModel, self).__init__()

        self.degree_x = degree_x
        self.degree_y = degree_y
        self.device = device

        if shape is not None:
            self.x, self.y = torch.meshgrid(
                torch.arange(shape[0], dtype=torch.float32, device=self.device),
                torch.arange(shape[1], dtype=torch.float32, device=self.device),
                indexing='ij'
            )
            self.x = (self.x - shape[0] // 2) / (shape[0] // 2)
            self.y = (self.y - shape[1] // 2) / (shape[1] // 2)

        self.num_epochs = num_epochs
        self.loss_tolerance = loss_tolerance

        # Create parameters for each polynomial coefficient
        self.coefficients = nn.ParameterList(
            [nn.Parameter(torch.randn(1).to(self.device)) for _ in range((degree_x + 1) * (degree_y + 1))]
        )

        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.75, patience=patience // 2
            )
        self.patience = patience

    def forward(self, x=None, y=None):

        if x is None or y is None:
            x, y = self.x, self.y
        
        out = torch.zeros_like(x, device=self.device)
        idx = 0
        for i in range(self.degree_x + 1):
            for j in range(self.degree_y + 1):
                out += self.coefficients[idx] * (x ** i) * (y ** j)
                idx += 1
        return out

    def fit(self, z, x=None, y=None, n_init=3):

        if x is None or y is None:
            x, y = self.x, self.y

        x, y, z = x.flatten().unsqueeze(1), y.flatten().unsqueeze(1), z.flatten().unsqueeze(1)
        z = z.clone().detach().to(self.device)

        print(f'Fitting a {self.degree_x}x{self.degree_y} polynomial to the data...')

        # Initialize the coefficients a few times and keep the best one
        best_loss = np.inf
        best_coeffs = None
        if n_init > 1:
            print('Initializing coefficients...')
            for _ in range(n_init):
                for param in self.parameters():
                    param.data = torch.randn(1).to(self.device)
                for _ in range(10):
                    self.optimizer.zero_grad()
                    outputs = self(x, y)
                    loss = self.criterion(outputs, z)
                    loss.backward()
                    self.optimizer.step()
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_coeffs = [param.data.clone() for param in self.parameters()]

            # Set the best coefficients
            for i, param in enumerate(self.parameters()):
                param.data = best_coeffs[i]

        # Train the model
        patience_counter = 0
        for epoch in range(self.num_epochs):
            outputs = self(x, y)
            loss = self.criterion(outputs, z)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.learning_rate_scheduler.step(loss)

            if loss.item() < best_loss - self.loss_tolerance:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter > self.patience:
                print(f'Early stopping at epoch {epoch + 1} with best loss: {best_loss:.4f}')
                break

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}')

class Polynomial2DModelNN(nn.Module):
    def __init__(self,
                 input_dim=2,
                 output_dim=1,
                 n_hidden_layers=6,
                 hidden_dim=64, 
                 device='cuda', 
                 num_epochs=100000, 
                 loss_tolerance=1e-3, 
                 lr=8e-3,
                 patience=2500
                 ):
        
        super(Polynomial2DModelNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden_layers = n_hidden_layers
        self.hidden_dim = hidden_dim
        self.device = device
        self.num_epochs = num_epochs
        self.loss_tolerance = loss_tolerance

        # Define the neural network
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        for _ in range(n_hidden_layers - 1):
            self.network.add_module('hidden', nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2)
            ))
        #Downsample
        self.network.add_module('downsample', nn.Linear(hidden_dim, hidden_dim // 2))
        self.network.add_module('leaky_relu', nn.LeakyReLU(0.2))
        self.network.add_module('output', nn.Linear(hidden_dim // 2, output_dim))

        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.75, patience=patience // 2
            )
        self.patience = patience

    def forward(self, x):
        return self.network(x)

    def fit(self, x, y, z):
        x, y, z = x.flatten(), y.flatten(), z.flatten()
        
        x = x.clone().detach().unsqueeze(1).to(self.device)
        y = y.clone().detach().unsqueeze(1).to(self.device)
        z = z.clone().detach().unsqueeze(1).to(self.device)

        input_data = torch.cat((x, y), dim=1)

        best_loss = np.inf
        patience_counter = 0
        for epoch in range(self.num_epochs):
            self.train()
            outputs = self(input_data)
            loss = self.criterion(outputs, z)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.learning_rate_scheduler.step(loss)

            if loss.item() < best_loss - self.loss_tolerance:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter > self.patience:
                print(f'Early stopping at epoch {epoch + 1} with best loss: {best_loss:.4f}')
                break

            if (epoch + 1) % 1000 == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}')

def phaseunwrap_skimage(field, norm_phase_after=False):
    """
    Unwrap the phase of the field using skimage.

    Parameters:
    field : torch.Tensor
        Complex field whose phase needs to be unwrapped.
    norm_phase_after : bool, optional
        If True, normalize the phase to be between -pi and pi after unwrapping.

    Returns:
    torch.Tensor
        The complex field with the unwrapped (and optionally normalized) phase.
    """
    from skimage.restoration import unwrap_phase

    # Check if the field is complex
    if torch.is_complex(field):
        phase = torch.angle(field)
    else:
        raise ValueError("The field is not complex")

    # Unwrap the phase using skimage's unwrap_phase function
    phase_unwrapped = unwrap_phase(phase.cpu().numpy())
    phase_unwrapped = torch.from_numpy(phase_unwrapped).to(field.device)

    # Normalize the phase to be between -pi and pi after unwrapping if requested
    if norm_phase_after:
        phase_unwrapped = phase_unwrapped - phase_unwrapped.min()
        phase_unwrapped = phase_unwrapped / phase_unwrapped.max()
        phase_unwrapped = phase_unwrapped * 2 * torch.pi - torch.pi

    # Return the corrected field
    return torch.abs(field) * torch.exp(1j * phase_unwrapped)
