# Dummy config for parameters in the form of a dictionary for application

# Parameters for the reconstruction
rec_params = {
    'filename': 'C:/MyFolder/xxx.avi',
    'height': 1450,
    'width': 1930,
    'crop': 0,
    'lowpass_filtered_phase': None,
    'filter_radius': None,
    'mask_radiis': None,
    'mask_case': "ellipse",
    'phase_corrections': 3,
    'kernel_size': 27,
    'sigma': 9
    }

# Parameters for full field reconstruction
rec_params_full = {
    'save_folder': 'C:/myFolder/xxx',
    'n_frames': 100,
    'n_frames_start': 0,
    'n_frames_step': 1,
    'fft_save': True,
    'recalculate_offset': True,
    'save_movie_gif': False,
    }

