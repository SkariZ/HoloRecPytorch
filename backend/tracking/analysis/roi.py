import numpy as np

def extract_complex_patch(field: np.ndarray, x: float, y: float, r: int):
    x0 = int(round(float(x)))
    y0 = int(round(float(y)))
    x1, x2 = x0 - r, x0 + r + 1
    y1, y2 = y0 - r, y0 + r + 1
    H, W = field.shape
    if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
        return None
    return field[y1:y2, x1:x2]


def load_roi_from_detection(provider, t, x, y, roi_radius):
    field = provider.get_complex_frame_np(int(t))
    return extract_complex_patch(field, float(x), float(y), int(roi_radius))


def load_rois_for_track(provider, tt, xx, yy, roi_radius):
    rois = []
    for t, x, y in zip(tt, xx, yy):
        patch = load_roi_from_detection(provider, t, x, y, roi_radius)
        if patch is not None:
            rois.append(patch)
    return rois


def build_stack_no_recenter(provider, tt, xx, yy, roi_radius, intensity_mode="abs2"):
    stack = []
    for t, x, y in zip(tt, xx, yy):
        field = provider.get_complex_frame_np(int(t))
        patch = extract_complex_patch(field, float(x), float(y), int(roi_radius))
        if patch is None:
            continue

        if intensity_mode == "abs2":
            img = (np.abs(patch) ** 2).astype(np.float32)
        else:
            img = np.abs(patch).astype(np.float32)

        stack.append(img)

    if not stack:
        return None
    return np.stack(stack, axis=0)


def build_complex_stack_no_recenter(provider, tt, xx, yy, roi_radius):
    stack = []
    for t, x, y in zip(tt, xx, yy):
        field = provider.get_complex_frame_np(int(t))
        patch = extract_complex_patch(field, float(x), float(y), int(roi_radius))
        if patch is None:
            continue
        stack.append(patch.astype(np.complex64))

    if not stack:
        return None
    return np.stack(stack, axis=0)
