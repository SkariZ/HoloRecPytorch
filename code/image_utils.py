import glob
import re

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
    plt.imsave(f"{folder}/{name}.png", frame, cmap=cmap)

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


def cropping_image(image, h, w, corner):
    """
    Crops the image
    """
    print(corner)
    hi, wi = image.shape[:2]
    if hi<=h or wi<=w:
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
    

def save_video(folder, savefolder, fps=12, quality=10):
    """
    Saves a video to a folder. Uses maximal settings for imageio writer. fps is defined in config.
    
    savefolder = Where and name of the video.
    folder = Directory containing n_frames .png files.
    """

    #Check if package exists
    try:
        #import imageio
        import imageio.v2 as imageio
    except ImportError:
        print("Package imageio not installed. Please install with 'pip install imageio'.")
        return

    writer = imageio.get_writer(savefolder, mode='I', codec='mjpeg', fps=fps, quality=quality, pixelformat='yuvj444p', macro_block_size=1)
    #writer = imageio.get_writer(savefolder, mode = 'I')

    imgs = glob.glob(folder + "*.png")
    imgs.sort(key=natural_keys)
    for file in imgs:
        im = imageio.imread(file)
        writer.append_data(im)
    writer.close()
    
def gif(folder, savefolder, duration=100, loop=0):
    """
    Save frames to a gif. 
    
    folder = Directory containing .png files.
    """
    try:
        from PIL import Image
    except ImportError:
        print("Package PIL not installed. Please install with 'pip install Pillow'.")
        return

    # Create the frames
    frames = []
    imgs = glob.glob(folder + "*.png")
    imgs.sort(key=natural_keys)
    for file_name in imgs:
        new_frame = Image.open(file_name)
        new_frame = new_frame.convert("P", palette=Image.ADAPTIVE)

        frames.append(new_frame)
    # Save into a GIF file that loops forever
    frames[0].save(savefolder, format='GIF', append_images=frames[1:] , save_all=True, duration=duration, loop=loop)