import cv2
import numpy as np

def read_video(filename, start_frame=0, max_frames=None):
    """
    Read a video file and return the frames as a numpy array.
    """

    video = cv2.VideoCapture(filename) # videobject
    
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) # Number of frames

    frames = []
    for i in range(start_frame, n_frames):
        ret, frame = video.read()

        if not ret:
            break

        if max_frames is not None and i >= max_frames:
            break
        frames.append(frame[...,0])

    frames = np.array(frames, dtype=np.float32)

    video.release()

    return frames
