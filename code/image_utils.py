import glob
import os
import re
import numpy as np
from PIL import Image
import imageio.v2 as imageio

import matplotlib.pyplot as plt

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text) ]

#Save frame in native resolution. Can change colormap if necessary.
def save_frame(frame, folder, name, cmap='gray', annotate='False', annotatename='', dpi=300):

    if annotate:
        fig, ax = plt.subplots()
        ax.imshow(frame, cmap=cmap)
        ax.annotate(
            annotatename, 
            xy=(0.015, 0.985),
            xycoords='axes fraction', 
            fontsize=14, 
            horizontalalignment='left', 
            verticalalignment='top',
            color='white'
            )
        ax.axis('off')
        fig.savefig(f"{folder}/{name}.png", bbox_inches="tight", pad_inches=0, dpi=dpi)
        plt.close(fig)
    
    else:
        plt.imsave(f"{folder}/{name}.png", frame, cmap=cmap, dpi=dpi)


def cropping_image(image, h, w, corner):
    """
    Crops the image
    """
    
    hi, wi = image.shape[:2]
    if hi<h or wi<w:
        raise Exception("Cropping size larger than actual image size.")

    if corner == 1:
        image = image[:h, :w] #Top left
    elif corner == 2:
        image = image[:h, -w:] #Top right
    elif corner == 3:
        image = image[-h:, :w] #Bottom left
    elif corner == 4:
        image = image[-h:, -w:] #Bottom right

    return image
    

# -------------------- Save individual frames --------------------
def save_frame(frame, folder, name, cmap='gray', annotate=False, annotatename='', dpi=300):
    """
    Save a single frame to disk with optional annotation.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.png")

    if annotate:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.imshow(frame, cmap=cmap)
        ax.annotate(
            annotatename,
            xy=(0.015, 0.985),
            xycoords='axes fraction',
            fontsize=14,
            ha='left',
            va='top',
            color='white'
        )
        ax.axis('off')
        fig.savefig(path, bbox_inches="tight", pad_inches=0, dpi=dpi)
        plt.close(fig)
    else:
        import matplotlib.pyplot as plt
        plt.imsave(path, frame, cmap=cmap, dpi=dpi)

# -------------------- Save video (.avi or .mp4) --------------------
def save_video(folder, savefile, fps=12, codec='MJPG', quality=10):
    """
    Save a folder of PNG frames as a high-quality video (.avi or .mp4).

    Requires ffmpeg installed for mp4 output.

    Args:
        folder (str): Path to folder containing PNG frames.
        savefile (str): Output video file path (.avi or .mp4 recommended).
        fps (int): Frames per second.
        codec (str): Codec, e.g., 'MJPG', 'XVID', 'H264'.
        quality (int): For MJPG codec (1-10).
    """
    if not os.path.exists(folder):
        raise ValueError(f"Folder does not exist: {folder}")

    imgs = glob.glob(os.path.join(folder, "*.png"))
    if not imgs:
        raise ValueError(f"No PNG files found in folder: {folder}")

    imgs.sort(key=natural_keys)

    # Force the ffmpeg plugin
    writer = imageio.get_writer(savefile, format='FFMPEG', mode='I', fps=fps, codec=codec, quality=quality)

    for file in imgs:
        img = imageio.imread(file)

        # Ensure uint8
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        # Ensure 3-channel RGB for video
        if len(img.shape) == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.shape[2] == 4:  # RGBA -> RGB
            img = img[:, :, :3]

        writer.append_data(img)

    writer.close()
    print(f"Video saved: {savefile}")

# -------------------- Save GIF --------------------
def save_gif(folder, savefile, duration=100, loop=0, resize=None):
    """
    Save frames from folder as GIF with better quality.
    resize: tuple (width, height) if resizing is needed, otherwise None
    """
    imgs = glob.glob(os.path.join(folder, "*.png"))
    if not imgs:
        raise ValueError("No PNG files found in folder")
    imgs.sort(key=natural_keys)

    frames = [Image.open(f).convert("RGBA") for f in imgs]

    if resize:
        frames = [f.resize(resize, Image.LANCZOS) for f in frames]

    frames[0].save(
        savefile,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
        optimize=True
    )
    print(f"GIF saved: {savefile}")