import torch
import torch.fft as fft

from other_utils import create_circular_mask

def data_to_real(img):
    """
    Transforms a complex image to a real image.

    Input:
        img : complex image (torch complex tensor)
    Output:
        Real image (torch float tensor with 2 channels).
    """
    image = torch.zeros((*img.shape, 2), dtype=torch.float32).to(img.device)
    image[..., 0] = img.real
    image[..., 1] = img.imag
    return image

def real_to_imag(img):
    """
    Transforms a real image with 2 channels to a complex image.

    Input:
        img : real image (torch float tensor with 2 channels)
    Output:
        Complex image (torch complex tensor).
    """
    return img[..., 0] + 1j * img[..., 1]

def field_to_vec(field, pupil_radius, mask=None):
    """
    Transforms a field to a vector given a pupil radius.

    Input:
        field : complex tensor representing the field
        pupil_radius : radius of the pupil
        mask : optional circular mask
    Output:
        Vector of complex numbers (torch complex tensor).
    """
    if not torch.is_complex(field):
        field = field[..., 0] + 1j * field[..., 1]

    h, w = field.shape
    fft_image = fft.fftshift(fft.fft2(field))

    if mask is None:
        mask = create_circular_mask(h, w, radius=pupil_radius)
    
    return fft_image[mask]

def field_to_vec_multi(fields, pupil_radius, mask=None):
    """
    Transforms multiple fields to vectors given a pupil radius.

    Input:
        fields : complex tensor representing multiple fields
        pupil_radius : radius of the pupil
        mask : optional circular mask
    Output:
        List of vectors of complex numbers (list of torch complex tensors).
    """
    if not torch.is_complex(fields):
        fields = fields[..., 0] + 1j * fields[..., 1]

    _, h, w = fields.shape

    if mask is None:
        mask = create_circular_mask(h, w, radius=pupil_radius).to(fields.device)

    fvec = torch.tensor([], dtype=torch.complex64, device=fields.device)
    for field in fields:
        fft_image = fft.fftshift(fft.fft2(field))
        vec = fft_image[mask]
        fvec = torch.cat((fvec, vec.unsqueeze(0)), dim=0)
    
    return fvec

def vec_to_field(vec, pupil_radius, shape, mask=None, to_real=False):
    """
    Transforms a vector to a field given pupil radius and shape.

    Input:
        vec : vector of complex numbers
        pupil_radius : radius of the pupil
        shape : shape of the resulting field
        mask : optional circular mask
        to_real : flag indicating whether to convert to real image
    Output:
        Complex tensor representing the field.
    """
    if mask is None:
        mask = create_circular_mask(shape[0], shape[1], radius=pupil_radius)
    mask = mask.type(torch.complex64).to(vec.device)
    mask[mask == 1] = vec

    field = fft.ifft2(fft.ifftshift(mask))
    if to_real:
        field = data_to_real(field)

    return field

def vec_to_field_multi(vecs, pupil_radius, shape, mask=None, to_real=False):
    """
    Transforms multiple vectors to fields given pupil radius and shape.

    Input:
        vecs : list of vectors of complex numbers
        pupil_radius : radius of the pupil
        shape : shape of the resulting fields
        mask : optional circular mask
        to_real : flag indicating whether to convert to real images
    Output:
        List of complex tensors representing the fields.
    """
    if mask is None:
        mask = create_circular_mask(shape[0], shape[1], radius=pupil_radius)
    mask = mask.type(torch.complex64).to(vecs.device)

    fields = torch.tensor([], dtype=torch.complex64, device=vecs.device)
    for vec in vecs:
        mm = mask.clone()
        mm[mm == 1] = vec
        field = fft.ifft2(fft.ifftshift(mm))
        fields = torch.cat((fields, field.unsqueeze(0)), dim=0)

    if to_real:
        fields = torch.stack([data_to_real(f) for f in fields], dim=0)

    return fields
