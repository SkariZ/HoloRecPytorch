# Dummy config for parameters in the form of a dictionary for application

# Parameters for the reconstruction
rec_params = {
    'filename': 'C:/MyFolder/xxx.avi',
    'frame_idx': 0,
    'height': 1450,
    'width': 1930,
    'corner': 1,
    'lowpass_filtered_phase': 0,
    'filter_radius': None,
    'mask_radiis': None,
    'mask_case': "ellipse",
    'phase_corrections': 3,
    'skip_background_correction': 0,
    'correct_field': 1,
    'lowpass_kernel_end': 0,
    'kernel_size': 27,
    'sigma': 9,
    }

# Parameters for full field reconstruction
rec_params_full = {
    'save_folder': 'C:/myFolder/xxx',
    'n_frames': 100,
    'start_frame': 0,
    'n_frames_step': 1,
    'fft_save': 1,
    'recalculate_offset': 1,
    'save_movie_gif': 0,
    'colormap': 'viridis',
    'cornerf': 1,
    }

# Tooltips for reconstruction parameters
rec_param_descriptions = {
    'filename': 'Path to your video file',
    'frame_idx': 'Index of the first frame to process',
    'height': 'Height of cropped image in pixels',
    'width': 'Width of cropped image in pixels',
    'corner': 'Corner to start cropping from (0 = top-left)',
    'lowpass_filtered_phase': 'Whether to apply lowpass filter on phase (1=yes, 0=no)',
    'filter_radius': 'Radius for FFT filter (pixels)',
    'mask_radiis': 'Comma-separated list of mask radii',
    'mask_case': 'Mask type (e.g., circle, square)',
    'phase_corrections': 'Number of phase corrections to apply',
    'skip_background_correction': 'Skip background correction (1=yes, 0=no)',
    'correct_field': 'Apply field correction (1=yes, 0=no)',
    'lowpass_kernel_end': 'End value for lowpass kernel',
    'kernel_size': 'Kernel size for smoothing',
    'sigma': 'Sigma for smoothing kernel',
    'save_folder': 'Folder to save reconstructed images and fields',
    'n_frames': 'Number of frames to reconstruct',
    'start_frame': 'Starting frame index for reconstruction',
    'n_frames_step': 'Step between frames',
    'fft_save': 'Save FFT images (1=yes, 0=no)',
    'recalculate_offset': 'Recalculate offset (1=yes, 0=no)',
    'save_movie_gif': 'Save GIF movie of reconstructed frames (1=yes, 0=no)',
    'colormap': 'Colormap for saved images (e.g., gray, viridis)',
    'cornerf': 'Corner for cropping during reconstruction',
}