import numpy as np

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
    
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
        
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

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
        center_w, center_h = int(w / 2), int(h / 2)
    else:
        center_w, center_h = center[0], center[1]

    if radius_h is None and radius_w is None:
        if percent is not None:
            radius_w, radius_h = int(percent * w), int(percent * h)
        else:
            radius_w, radius_h = int(0.25 * w / 2), int(0.25 * h / 2)  # Ellipsoid of this size. To get some output

    img = np.zeros((h, w), dtype=np.uint8)
    y_indices, x_indices = np.indices(img.shape[:2])

    # Calculate the equation of the ellipse
    ellipse_equation = (
        ((x_indices - center_w) / radius_w) ** 2 +
        ((y_indices - center_h) / radius_h) ** 2
    )

    # Set pixels within the ellipse to 1
    mask = np.where(ellipse_equation <= 1, 1, 0)
    return mask