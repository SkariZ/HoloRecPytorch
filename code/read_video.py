import cv2
import numpy as np


def get_video_properties(filename):
    """
    Get properties of a video file.

    Args:
        filename (str): Path to video file.

    Returns:
        dict: Dictionary containing properties like frame count, width, height, fps.
    """
    video = cv2.VideoCapture(filename)
    
    if not video.isOpened():
        raise ValueError(f"Could not open video file: {filename}")

    properties = {
        'frame_count': int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': video.get(cv2.CAP_PROP_FPS),
        'fourcc': int(video.get(cv2.CAP_PROP_FOURCC))
    }

    video.release()
    return properties


def read_video(filename, start_frame=0, max_frames=None, step=1):
    """
    Read a video file and return the frames as a numpy array.
    """

    video = cv2.VideoCapture(filename) # videobject
    
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) # Number of frames

    frames = []
    for i in range(start_frame, n_frames, step):
        ret, frame = video.read()

        if not ret:
            break

        if max_frames is not None and i >= max_frames:
            break
        frames.append(frame[...,0]) # Assuming grayscale video...

    frames = np.array(frames, dtype=np.float32)

    video.release()

    return frames


def read_video_by_indices(filename, indices):
    """
    Read specific frames from a video file and return them as a numpy array.

    Args:
        filename (str): Path to video file.
        indices (list[int]): Frame indices to read (0-based).

    Returns:
        np.ndarray: Array of frames (grayscale, float32), shape (len(indices), H, W)
    """
    video = cv2.VideoCapture(filename)
    frames = []

    # Sort indices just in case
    indices = sorted(indices)
    next_idx_ptr = 0  # Pointer to next frame index in the list

    current_frame = 0
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    while next_idx_ptr < len(indices) and current_frame < total_frames:
        ret, frame = video.read()
        if not ret:
            break

        if current_frame == indices[next_idx_ptr]:
            # Extract grayscale if needed
            frames.append(frame[..., 0] if frame.ndim == 3 else frame)
            next_idx_ptr += 1

        current_frame += 1

    video.release()
    return np.array(frames, dtype=np.float32)
