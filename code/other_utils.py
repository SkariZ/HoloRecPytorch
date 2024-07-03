import torch
import torch.nn.functional as F
import numpy as np

import scipy.ndimage as ndi


def create_circular_mask(h, w, center=None, radius=None):
    """
    Creates a circular mask.

    Input:
        h : height
        w : width
        center : Define center of image. If None -> middle is used.
        radius : radius of circle.
    Output:
        Circular mask.

    """
    
    if center is None:  # use the middle of the image
        center = (int(h/2), int(w/2))
        
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    
    X, Y = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    dist_from_center = torch.sqrt((X - center[0])**2 + (Y - center[1])**2)
    
    mask = dist_from_center <= radius
    return mask


def create_ellipse_mask(h, w, center=None, radius_h=None, radius_w=None, percent=0.05):
    """
    Creates an ellipsoid mask.

    Input:
        h : height
        w : width
        center : Define center of the image. If None, the middle is used.
        radius_h : Radius in height
        radius_w : Radius in width
        percent : If radius_h or radius_w is not defined, use this percentage factor instead.
    Output:
        Ellipsoid mask.
    """

    if center is None:
        center_w, center_h = int(w/2), int(h/2)
    else:
        center_w, center_h = center[0], center[1]

    if radius_h is None and radius_w is None:
        if percent is not None:
            radius_w, radius_h = int(percent*w), int(percent*h)
        else:
            radius_w, radius_h = int(0.25*w/2), int(0.25*h/2)  # Ellipsoid of this size. To get some output

    x_indices, y_indices = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    
    # Calculate the equation of the ellipse
    ellipse_equation = (
        ((x_indices - center_h) / radius_h) ** 2 +
        ((y_indices - center_w) / radius_w) ** 2
    )

    # Set pixels within the ellipse to 1
    mask = ellipse_equation <= 1
    return mask

def phase_frequencefilter(field, mask, is_field=True, return_phase=True, crop=0):
    """
    Lowpass filter the image with mask defined in inpu.

    Input:
        field : Complex valued field.
        mask : Mask (binary 2D image)
        is_field : if input is a field, or if needs to be forier transformed before.
        crop : if we shall crop the ouput slightly
    Output:
        phase_img : phase of optical field.
    """

    if is_field:
        freq = torch.fft.fftshift(torch.fft.fft2(field))
    else:
        freq = field
        
    #construct low-pass mask
    freq_low = freq * mask
    
    E_field = torch.fft.ifft2(torch.fft.fftshift(freq_low)) #Shift the zero-frequency component to the center of the spectrum. and compute inverse fft
    
    if crop > 0:
        E_field = E_field[crop:-crop, crop:-crop]

    if return_phase:
        return torch.angle(E_field)
    else:
        return E_field


def correctfield(field, n_iter=5):
    """
    Correct field
    """

    if field.dtype == torch.float32:
        field = field.to(torch.complex64)

    f_new = field.clone()

    # Normalize with mean of absolute value.
    f_new = f_new / torch.mean(torch.abs(f_new))

    for _ in range(n_iter):
        f_new = f_new * torch.exp(-1j * torch.median(torch.angle(f_new)))

    return f_new


def correctfield_sign(field, pos=True):
    """
    Force mean of the real part to be positive.
    """

    if field.dtype == torch.float32:
        field = field.to(torch.complex64)

    if pos:
        for i, f in enumerate(field):
            if torch.mean(torch.real(f)) < 0:
                field[i] = -f
    else:
        for i, f in enumerate(field):
            if torch.mean(torch.real(f)) > 0:
                field[i] = -f
                
    return field


def phase_corrections(phase, phase_corrections=3):
    """
    Corrects the phase by removing the phase jumps.

    Input:
        phase : phase image
        phase_corrections : number of phase corrections.
    Output:
        phase : corrected phase image.
    """

    for i in range(phase_corrections):
        phase = phase - torch.mean(phase)
        phase = ndi.median_filter(phase, size=5)
        phase = phase - torch.mean(phase)
    return phase


def phase_corrections_derivative(phase, phase_corrections=3):
    """
    Corrects the phase by removing the phase jumps.

    Input:
        phase : phase image
        phase_corrections : number of phase corrections.
    Output:
        phase : corrected phase image.
    """

    #Take the derivative of the phase
    phase = torch.tensor(phase, dtype=torch.float32)
    phase = phase - torch.mean(phase)
    phase = torch.cat((phase[1:], phase[-1:])) - phase
    phase = phase - torch.mean(phase)
    for i in range(phase_corrections):
        phase = ndi.median_filter(phase, size=5)
        phase = phase - torch.mean(phase)
    return phase