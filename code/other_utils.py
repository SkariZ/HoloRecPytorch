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

def phase_frequencefilter(field, mask, is_field=True, crop=0):
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
        phase_img = torch.angle(E_field[crop:-crop, crop:-crop])
    else:
        phase_img = torch.angle(E_field)
    
    return phase_img