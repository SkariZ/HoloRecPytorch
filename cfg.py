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

