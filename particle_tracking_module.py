# particle_tracking_module.py
import os
import time
import json
import numpy as np
import torch

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QScrollArea,
    QLabel, QLineEdit, QPushButton, QFileDialog, QSpinBox, QDoubleSpinBox,
    QCheckBox, QComboBox, QTextEdit
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from backend.tracking.frame_provider import FrameProvider
from backend.tracking.detection import UnetDetector, channel_from_complex, PeakDetector
from backend.tracking.linking import (
    link_detections, flatten_detections, save_detections_csv, save_tracks_csv
)
from backend.tracking.preprocess import PreprocessConfig, spatial_preprocess, temporal_subtract_pair
from backend.tracking.detection import PeakDetector, LoGDetector, UnetDetector
from backend.tracking.models.registry import UNET_MODEL_REGISTRY

from matplotlib.widgets import RectangleSelector

class ParticleTrackingModule(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Particle Tracking")

        self.full_data = None          # memmap
        self.provider = None           # FrameProvider
        self.preview_img = None        # np float32 (H,W)
        self.preview_dets = []         # list of PeakDetection
        self.fov_coords = None   # (x1, y1, x2, y2) in full-image coordinates

        self.save_folder = None
        self.data_path = None

        self._build_ui()

    def _build_ui(self):
        main_layout = QHBoxLayout(self)

        # ---------------- Left: controls (scrollable) ----------------
        controls_layout = QVBoxLayout()

        # Load box
        controls_layout.addWidget(QLabel("Input data (.npy):"))
        load_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select field.npy / focused_frames.npy / etc.")
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self._browse_npy)
        load_row.addWidget(self.path_edit, 1)
        load_row.addWidget(btn_browse)
        controls_layout.addLayout(load_row)

        self.btn_load = QPushButton("Load data")
        self.btn_load.clicked.connect(self._load_data)
        controls_layout.addWidget(self.btn_load)

        self.status = QLabel("Ready.")
        self.status.setWordWrap(True)
        controls_layout.addWidget(self.status)

        controls_layout.addSpacing(10)

        # FFT decode options (only needed for 1D frames)
        controls_layout.addWidget(QLabel("FFT decode (only if frames are saved as vectors):"))
        self.fft_checkbox = QCheckBox("Decode FFT-compressed frames")
        self.fft_checkbox.setChecked(False)
        controls_layout.addWidget(self.fft_checkbox)

        size_row = QHBoxLayout()
        self.height_input = QLineEdit("1024")
        self.width_input = QLineEdit("1024")
        size_row.addWidget(QLabel("Orig H:"))
        size_row.addWidget(self.height_input)
        size_row.addWidget(QLabel("Orig W:"))
        size_row.addWidget(self.width_input)
        controls_layout.addLayout(size_row)

        pr_row = QHBoxLayout()
        self.pupil_radius_input = QLineEdit("200")
        pr_row.addWidget(QLabel("Pupil radius:"))
        pr_row.addWidget(self.pupil_radius_input)
        controls_layout.addLayout(pr_row)

        self.mask_shape_combo = QComboBox()
        self.mask_shape_combo.addItems(["ellipse", "circle"])
        controls_layout.addWidget(QLabel("Mask shape:"))
        controls_layout.addWidget(self.mask_shape_combo)

        controls_layout.addSpacing(10)

        # Preview controls
        controls_layout.addWidget(QLabel("Preview frame index:"))
        self.frame_spin = QSpinBox()
        self.frame_spin.setMinimum(0)
        self.frame_spin.setMaximum(0)
        controls_layout.addWidget(self.frame_spin)

        controls_layout.addWidget(QLabel("Channel:"))
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["imag", "real", "abs", "phase"])
        self.channel_combo.setCurrentText("imag")
        controls_layout.addWidget(self.channel_combo)

        # -------- Preprocessing controls (your requested set) --------
        controls_layout.addWidget(QLabel("Preprocessing:"))

        form_pre = QFormLayout()

        self.bg_sigma = QDoubleSpinBox()
        self.bg_sigma.setRange(0.0, 1e6)
        self.bg_sigma.setDecimals(3)
        self.bg_sigma.setValue(0.0)  # 0 disables
        form_pre.addRow("Gaussian subtraction σ:", self.bg_sigma)

        self.smooth_sigma = QDoubleSpinBox()
        self.smooth_sigma.setRange(0.0, 1e6)
        self.smooth_sigma.setDecimals(3)
        self.smooth_sigma.setValue(0.0)  # 0 disables
        form_pre.addRow("Gaussian filter σ:", self.smooth_sigma)

        self.normalize_checkbox = QCheckBox("Normalize to 0–1")
        self.normalize_checkbox.setChecked(True)
        form_pre.addRow(self.normalize_checkbox)

        self.subsize_spin = QSpinBox()
        self.subsize_spin.setRange(0, 10**6)
        self.subsize_spin.setValue(0)  # 0 disables temporal subtraction
        form_pre.addRow("Window subtraction subsize:", self.subsize_spin)

        controls_layout.addLayout(form_pre)

        controls_layout.addWidget(QLabel("Field of view (ROI):"))

        # For roi selector
        roi_row = QHBoxLayout()
        self.btn_apply_roi = QPushButton("Use drawn ROI")
        self.btn_apply_roi.clicked.connect(self._apply_roi_from_selector)
        roi_row.addWidget(self.btn_apply_roi)

        self.btn_reset_roi = QPushButton("Reset ROI")
        self.btn_reset_roi.clicked.connect(self._reset_roi)
        roi_row.addWidget(self.btn_reset_roi)

        controls_layout.addLayout(roi_row)

        self.btn_preview = QPushButton("Update preview")
        self.btn_preview.clicked.connect(self._update_preview)
        controls_layout.addWidget(self.btn_preview)

        controls_layout.addSpacing(10)        

        # Detection controls
        controls_layout.addWidget(QLabel("Detection (on preprocessed image):"))
        form_det = QFormLayout()

        self.detector_combo = QComboBox()
        self.detector_combo.addItems([
            "Peaks (local max)",
            "LoG (blobs)",
            "U-Net (score map)",
        ])
        form_det.addRow("Detector:", self.detector_combo)

        # U-Net model row container (hidden unless U-Net selected)
        self.unet_row_widget = QWidget()
        unet_row_layout = QHBoxLayout(self.unet_row_widget)
        unet_row_layout.setContentsMargins(0, 0, 0, 0)

        self.unet_model_combo = QComboBox()
        self.unet_model_combo.addItems(list(UNET_MODEL_REGISTRY.keys()))
        unet_row_layout.addWidget(self.unet_model_combo, 1)

        form_det.addRow("U-Net model:", self.unet_row_widget)

        # start hidden
        self.unet_row_widget.setVisible(False)

        # Peaks parameter (only relevant for Peaks)
        self.det_win = QSpinBox()
        self.det_win.setRange(1, 999)
        self.det_win.setValue(9)
        form_det.addRow("Local max window:", self.det_win)

        # LoG parameter (only relevant for LoG)
        self.log_sigma = QDoubleSpinBox()
        self.log_sigma.setRange(0.1, 50.0)
        self.log_sigma.setDecimals(3)
        self.log_sigma.setValue(2.0)
        form_det.addRow("LoG sigma:", self.log_sigma)

        # Universal controls
        self.det_thresh = QDoubleSpinBox()
        self.det_thresh.setRange(-1e9, 1e9)
        self.det_thresh.setDecimals(6)
        self.det_thresh.setValue(0.25)
        form_det.addRow("Threshold:", self.det_thresh)

        self.det_topk = QSpinBox()
        self.det_topk.setRange(0, 10000)
        self.det_topk.setValue(0)  # 0 = off
        form_det.addRow("Top-K (0=off):", self.det_topk)

        self.border_skip = QSpinBox()
        self.border_skip.setRange(0, 10000)
        self.border_skip.setValue(16)
        form_det.addRow("Skip border (px):", self.border_skip)

        controls_layout.addLayout(form_det)

        # Hide/show relevant rows automatically (keeps UI clean)
        def _update_detector_params():
            det = self.detector_combo.currentText()
            is_peaks = det.startswith("Peaks")
            is_log   = det.startswith("LoG")
            is_unet  = det.startswith("UNet") or det.startswith("U-Net")

            # Peaks params
            self.det_win.setVisible(is_peaks)
            form_det.labelForField(self.det_win).setVisible(is_peaks)

            # LoG params
            self.log_sigma.setVisible(is_log)
            form_det.labelForField(self.log_sigma).setVisible(is_log)

            # U-Net model selector row
            self.unet_row_widget.setVisible(is_unet)
            # (optional) also hide/show the label:
            form_det.labelForField(self.unet_row_widget).setVisible(is_unet)


        self.detector_combo.currentTextChanged.connect(_update_detector_params)
        _update_detector_params()
    
        self.btn_detect = QPushButton("Detect on preview + overlay")
        self.btn_detect.clicked.connect(self._detect_preview)
        controls_layout.addWidget(self.btn_detect)

        controls_layout.addSpacing(10)

        # Tracking controls
        controls_layout.addWidget(QLabel("Tracking (all processed frames):"))
        form_tr = QFormLayout()

        self.max_link_dist = QDoubleSpinBox()
        self.max_link_dist.setRange(0.0, 1e9)
        self.max_link_dist.setValue(10.0)
        form_tr.addRow("Max link dist:", self.max_link_dist)

        self.max_gap = QSpinBox()
        self.max_gap.setRange(0, 10000)
        self.max_gap.setValue(0)
        form_tr.addRow("Max frame gap:", self.max_gap)

        self.min_len = QSpinBox()
        self.min_len.setRange(1, 1000000)
        self.min_len.setValue(3)
        form_tr.addRow("Min track length:", self.min_len)

        controls_layout.addLayout(form_tr)

        # Save folder
        controls_layout.addWidget(QLabel("Save folder:"))
        save_row = QHBoxLayout()
        self.save_edit = QLineEdit()
        self.save_edit.setPlaceholderText("Choose where to save tracks/detections")
        btn_save_browse = QPushButton("Browse")
        btn_save_browse.clicked.connect(self._browse_save_folder)
        save_row.addWidget(self.save_edit, 1)
        save_row.addWidget(btn_save_browse)
        controls_layout.addLayout(save_row)

        self.btn_run = QPushButton("Run tracking + save CSV")
        self.btn_run.clicked.connect(self._run_tracking_all)
        controls_layout.addWidget(self.btn_run)

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        controls_layout.addWidget(self.log, 1)

        # Wrap controls in scroll area
        container = QWidget()
        container.setLayout(controls_layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        main_layout.addWidget(scroll, 1)

        # ---------------- Right: preview canvas ----------------
        self.canvas = FigureCanvas(plt.Figure())
        self.ax = self.canvas.figure.add_subplot(111)
        self.ax.set_title("Preview")
        main_layout.addWidget(self.canvas, 3)

        self._roi_selector = RectangleSelector(
        self.ax,
        onselect=self._on_roi_select,
        useblit=True,
        button=[1],           # left click
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True
        )
        self._roi_last = None  # temporary selection (x1,y1,x2,y2)

    # ---------- Helpers ----------
    def _log(self, msg: str):
        self.log.append(msg)

    def _browse_npy(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Numpy File", "", "Numpy Files (*.npy)")
        if fname:
            self.path_edit.setText(fname)

    def _browse_save_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Folder", self.save_folder or "")
        if folder:
            self.save_folder = folder
            self.save_edit.setText(folder)

    def _parse_orig_size(self):
        try:
            h = int(self.height_input.text().strip())
            w = int(self.width_input.text().strip())
            return (h, w)
        except Exception:
            return None

    def _parse_pupil_radius(self):
        txt = self.pupil_radius_input.text().strip()
        if txt == "":
            return None
        try:
            return int(txt)
        except Exception:
            return None

    def _try_apply_field_meta(self, npy_path: str):
        folder = os.path.dirname(npy_path)
        meta_path = os.path.join(folder, "field_meta.npz")
        if not os.path.exists(meta_path):
            self._log("No field_meta.npz found (manual FFT settings if needed).")
            return
        try:
            meta = np.load(meta_path, allow_pickle=True)
            orig_h = int(meta["orig_height"])
            orig_w = int(meta["orig_width"])
            pupil_radius = int(meta["pupil_radius"])
            mask_shape = str(meta["mask_shape"])

            self.height_input.setText(str(orig_h))
            self.width_input.setText(str(orig_w))
            self.pupil_radius_input.setText(str(pupil_radius))
            if mask_shape in ["ellipse", "circle"]:
                self.mask_shape_combo.setCurrentText(mask_shape)

            if "fft_save" in meta.files and int(meta["fft_save"]) == 1:
                self.fft_checkbox.setChecked(True)

            self._log(f"Applied meta: orig=({orig_h},{orig_w}), radius={pupil_radius}, mask={mask_shape}")
        except Exception as e:
            self._log(f"Warning: failed to read/apply field_meta.npz: {e}")

    def _make_pre_cfg(self) -> PreprocessConfig:
        return PreprocessConfig(
            gaussian_subtract_sigma=float(self.bg_sigma.value()),
            gaussian_filter_sigma=float(self.smooth_sigma.value()),
            normalize_01=bool(self.normalize_checkbox.isChecked()),
            temporal_subsize=int(self.subsize_spin.value()),
        )

    def _rebuild_provider(self):
        if self.full_data is None:
            return
        self.provider = FrameProvider(
            self.full_data,
            fft_enabled=self.fft_checkbox.isChecked(),
            orig_size=self._parse_orig_size(),
            pupil_radius=self._parse_pupil_radius(),
            mask_shape=self.mask_shape_combo.currentText(),
        )

    def _build_detector(self):
        name = self.detector_combo.currentText()
        topk_val = int(self.det_topk.value())
        topk = None if topk_val == 0 else topk_val
        threshold = float(self.det_thresh.value())
        border_skip = int(self.border_skip.value())


        if name.startswith("Peaks"):
            return PeakDetector(win_size=int(self.det_win.value()), threshold=threshold, top_k=topk, border_skip=border_skip)
        
        elif name.startswith("LoG"):
            return LoGDetector(sigma=float(self.log_sigma.value()), threshold=threshold, top_k=topk, border_skip=border_skip)
        
        elif name.startswith("U-Net"):
            model_name = self.unet_model_combo.currentText()
            spec = UNET_MODEL_REGISTRY[model_name]
            return UnetDetector(
                model_spec=spec,
                threshold=threshold,
                top_k=topk,
                border_skip=border_skip,
            )
        else:
            raise ValueError(f"Unknown detector: {name}")


    # ---------- Step 1: Load ----------
    def _load_data(self):
        path = self.path_edit.text().strip()
        if not path:
            self.status.setText("Select a .npy file first.")
            return
        if not os.path.exists(path):
            self.status.setText("File not found.")
            return

        try:
            self.full_data = np.load(path, mmap_mode="r")
        except Exception as e:
            self.status.setText(f"Load failed: {e}")
            return

        self.data_path = path
        n_frames = int(self.full_data.shape[0])
        self.frame_spin.setMaximum(max(0, n_frames - 1))

        # Default save folder: sibling to the folder containing the npy (like your other modules)
        file_folder = os.path.dirname(path)
        parent = os.path.dirname(file_folder)
        default_save = os.path.join(parent, "tracking")
        os.makedirs(default_save, exist_ok=True)
        self.save_folder = default_save
        self.save_edit.setText(default_save)

        self.status.setText(f"Loaded {path} | shape={self.full_data.shape} dtype={self.full_data.dtype}")
        self._log(f"Loaded data: shape={self.full_data.shape}, ndim={self.full_data.ndim}, dtype={self.full_data.dtype}")

        # Apply meta if present + auto-enable FFT if shape looks vectorized (T,N)
        self._try_apply_field_meta(path)
        if self.full_data.ndim == 2:
            self.fft_checkbox.setChecked(True)
            self._log("Data looks FFT-compressed (T,N). Enabled FFT decode automatically.")

        # Build provider
        try:
            self._rebuild_provider()
        except Exception as e:
            self.provider = None
            self._log(f"Provider init failed: {e}")
            self.status.setText(f"Provider init failed: {e}")
            return

        # auto preview
        self._update_preview()

    # ---------- Step 2: Preview + preprocess ----------
    def _update_preview(self):
        if self.provider is None:
            self._log("No provider. Load data first (and set FFT params if needed).")
            return

        idx = int(self.frame_spin.value())
        cfg = self._make_pre_cfg()
        channel = self.channel_combo.currentText()

        try:
            field = self.provider.get_complex_frame_np(idx)
            img = channel_from_complex(field, channel)
            img = self._apply_roi_to_image(img)

            # temporal subtraction: t - (t + subsize)
            if cfg.temporal_subsize > 0:
                j = idx + cfg.temporal_subsize
                if j >= len(self.provider):
                    self._log(f"Preview: idx+subsize out of range (idx={idx}, subsize={cfg.temporal_subsize}).")
                    self.preview_img = None
                    self.preview_dets = []
                    self._redraw_preview()
                    return
                field2 = self.provider.get_complex_frame_np(j)
                img2 = channel_from_complex(field2, channel)
                img2 = self._apply_roi_to_image(img2)
                pre = temporal_subtract_pair(img, img2, cfg)
            else:
                pre = spatial_preprocess(img, cfg)

        except Exception as e:
            self._log(f"Preview decode/preprocess failed: {e}")
            # try rebuilding provider once (in case FFT settings were edited)
            try:
                self._rebuild_provider()
            except Exception:
                return
            return

        self.preview_img = pre
        self.preview_dets = []  # clear overlay
        self._redraw_preview()

    def _on_roi_select(self, eclick, erelease):
        if eclick.xdata is None or eclick.ydata is None or erelease.xdata is None or erelease.ydata is None:
            return
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        self._roi_last = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        self._log(f"ROI drawn: {self._roi_last}")

    def _apply_roi_from_selector(self):
        if self._roi_last is None:
            self._log("Draw an ROI on the preview first (drag a rectangle).")
            return
        self.fov_coords = self._roi_last
        self._log(f"ROI applied: {self.fov_coords}")
        self._update_preview()

    def _reset_roi(self):
        self.fov_coords = None
        self._roi_last = None
        self._log("ROI reset (full frame).")
        self._update_preview()

    def _apply_roi_to_image(self, img2d: np.ndarray) -> np.ndarray:
        if self.fov_coords is None:
            return img2d
        x1, y1, x2, y2 = self.fov_coords
        # clip to bounds
        H, W = img2d.shape
        x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W))
        y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H))
        if x2 <= x1 or y2 <= y1:
            return img2d
        return img2d[y1:y2, x1:x2]

    # ---------- Step 3: Detect + visualize ----------
    def _detect_preview(self):
        if self.preview_img is None:
            self._log("No preview image. Click 'Update preview' first.")
            return

        topk_val = int(self.det_topk.value())
        topk = None if topk_val == 0 else topk_val

        detector = self._build_detector()

        dets = detector.detect(self.preview_img)
        self.preview_dets = dets
        self._log(f"Preview detections: {len(dets)}")
        self._redraw_preview()

    def _redraw_preview(self):
        self.ax.clear()
        if self.preview_img is not None:
            self.ax.imshow(self.preview_img, cmap="gray", interpolation="nearest")
        if self.preview_dets:
            xs = [d.x for d in self.preview_dets]
            ys = [d.y for d in self.preview_dets]
            self.ax.scatter(xs, ys, s=22, marker="o", facecolors="none", edgecolors="lime")
        self.ax.set_title(f"Preview (frame {self.frame_spin.value()})")
        self.canvas.draw()

    # ---------- Step 4: Full tracking + save ----------
    def _run_tracking_all(self):
        t0 = time.time()

        if self.provider is None:
            self._log("No provider. Load data first.")
            return
        if not self.save_folder:
            self._log("Select a save folder first.")
            return

        cfg = self._make_pre_cfg()
        channel = self.channel_combo.currentText()

        T = len(self.provider)
        T_eff = T - cfg.temporal_subsize if cfg.temporal_subsize > 0 else T
        if T_eff <= 0:
            self._log("Temporal subsize too large for this dataset.")
            return

        self._log(f"Running detection+tracking. Total frames={T}, processed frames={T_eff}, subsize={cfg.temporal_subsize}")

        # Detector from current UI params
        topk_val = int(self.det_topk.value())
        topk = None if topk_val == 0 else topk_val
        detector = self._build_detector()

        detections_by_frame = []

        for t in range(T_eff):
            if t % 50 == 0:
                self.status.setText(f"Processing frame {t}/{T_eff}...")

            try:
                field = self.provider.get_complex_frame_np(t)
                img = channel_from_complex(field, channel)
                img = self._apply_roi_to_image(img)

                if cfg.temporal_subsize > 0:
                    field2 = self.provider.get_complex_frame_np(t + cfg.temporal_subsize)
                    img2 = channel_from_complex(field2, channel)
                    img2 = self._apply_roi_to_image(img2)
                    pre = temporal_subtract_pair(img, img2, cfg)
                else:
                    pre = spatial_preprocess(img, cfg)

                dets = detector.detect(pre)
                detections_by_frame.append(dets)

            except Exception as e:
                self._log(f"Frame {t} failed: {e}")
                detections_by_frame.append([])

        # Link
        self.status.setText("Linking tracks...")
        tracks = link_detections(
            detections_by_frame,
            max_link_dist=float(self.max_link_dist.value()),
            max_frame_gap=int(self.max_gap.value()),
            min_track_len=int(self.min_len.value()),
        )

        flat = flatten_detections(detections_by_frame)

        # Save
        os.makedirs(self.save_folder, exist_ok=True)
        det_path = os.path.join(self.save_folder, "detections.csv")
        tr_path = os.path.join(self.save_folder, "tracks.csv")

        save_detections_csv(det_path, flat)
        save_tracks_csv(tr_path, tracks)

        elapsed = time.time() - t0

        settings = self._collect_run_settings(cfg)
        metrics = self._compute_tracking_metrics(detections_by_frame, tracks, T_eff)

        report = {
            "settings": settings,
            "metrics": metrics,
            "runtime_seconds": float(elapsed),
        }

        self._save_run_report(report, self.save_folder)


        self._log(f"Saved:\n  {det_path}\n  {tr_path}")
        self.status.setText(f"Done. Tracks={len(tracks)}  Detections={len(flat)}")

    def cleanup(self):
        self.full_data = None
        self.provider = None
        self.preview_img = None
        self.preview_dets = []

    def _collect_run_settings(self, cfg: PreprocessConfig) -> dict:
        # FFT settings
        settings = {
            "input_path": self.data_path,
            "data_shape": tuple(self.full_data.shape) if self.full_data is not None else None,
            "channel": self.channel_combo.currentText(),

            "fft": {
                "enabled": bool(self.fft_checkbox.isChecked()),
                "orig_size": self._parse_orig_size(),
                "pupil_radius": self._parse_pupil_radius(),
                "mask_shape": self.mask_shape_combo.currentText(),
            },

            "preprocess": {
                "gaussian_subtract_sigma": float(cfg.gaussian_subtract_sigma),
                "gaussian_filter_sigma": float(cfg.gaussian_filter_sigma),
                "normalize_01": bool(cfg.normalize_01),
                "temporal_subsize": int(cfg.temporal_subsize),
            },

            "detection": {
                "detector": self.detector_combo.currentText(),
                "threshold": float(self.det_thresh.value()),
                "top_k": int(self.det_topk.value()),   # 0 means off
                # detector-specific knobs (safe to include even if not used)
                "peaks_win_size": int(self.det_win.value()),
                "log_sigma": float(self.log_sigma.value()),
            },

            "linking": {
                "max_link_dist": float(self.max_link_dist.value()),
                "max_frame_gap": int(self.max_gap.value()),
                "min_track_len": int(self.min_len.value()),
            },
        }
        return settings


    def _compute_tracking_metrics(self, detections_by_frame, tracks, T_eff: int) -> dict:
        # detections per frame
        det_counts = np.array([len(d) for d in detections_by_frame], dtype=np.int32)
        total_dets = int(det_counts.sum())

        def _track_items(tr):
            """
            Return an iterable of points/detections for a track object or list.
            """
            # common patterns
            if hasattr(tr, "detections"):
                return tr.detections
            if hasattr(tr, "points"):
                return tr.points
            if hasattr(tr, "nodes"):
                return tr.nodes
            if hasattr(tr, "path"):
                return tr.path
            # if it's already a list/tuple
            if isinstance(tr, (list, tuple)):
                return tr
            # fallback: try iterating
            try:
                return list(tr)
            except TypeError:
                return []

        def _track_length(tr):
            items = _track_items(tr)
            try:
                return len(items)
            except Exception:
                return 0

        track_lengths = (
            np.array([_track_length(tr) for tr in tracks], dtype=np.int32)
            if tracks else np.array([], dtype=np.int32)
        )


        # per-step displacements across all tracks
        step_dists = []
        track_net_disp = []

        # tracks are typically a list of lists; each element usually contains at least (frame, x, y) or dicts.
        # We’ll try to handle the common cases robustly.
        def _xy_from_item(item):
            # supports Detection-like objects, dicts, tuples
            if hasattr(item, "x") and hasattr(item, "y"):
                return float(item.x), float(item.y)
            if isinstance(item, dict):
                return float(item.get("x")), float(item.get("y"))
            # tuple/list fallback: assume (..., x, y) or (x, y)
            if isinstance(item, (tuple, list)):
                if len(item) >= 3:
                    return float(item[-2]), float(item[-1])
                if len(item) == 2:
                    return float(item[0]), float(item[1])
            raise ValueError("Unknown track item format")

        for tr in tracks:
            items = _track_items(tr)
            if len(items) < 2:
                continue
            xs, ys = [], []
            for item in items:
                x, y = _xy_from_item(item)
                xs.append(x); ys.append(y)
            xs = np.asarray(xs, dtype=np.float32)
            ys = np.asarray(ys, dtype=np.float32)

            dx = np.diff(xs)
            dy = np.diff(ys)
            d = np.sqrt(dx*dx + dy*dy)
            step_dists.extend(d.tolist())

            net = float(np.sqrt((xs[-1]-xs[0])**2 + (ys[-1]-ys[0])**2))
            track_net_disp.append(net)

        step_dists = np.asarray(step_dists, dtype=np.float32)
        track_net_disp = np.asarray(track_net_disp, dtype=np.float32)

        # coverage: how many detections are in tracks?
        # If your tracks store the actual detections, then sum(track_lengths) approximates that.
        dets_in_tracks = int(track_lengths.sum()) if track_lengths.size else 0
        coverage = (dets_in_tracks / total_dets) if total_dets > 0 else 0.0

        metrics = {
            "frames_processed": int(T_eff),

            "detections": {
                "total": total_dets,
                "per_frame_mean": _np_float(det_counts.mean()) if det_counts.size else 0.0,
                "per_frame_median": _np_float(np.median(det_counts)) if det_counts.size else 0.0,
                "per_frame_std": _np_float(det_counts.std()) if det_counts.size else 0.0,
                "per_frame_min": _np_int(det_counts.min()) if det_counts.size else 0,
                "per_frame_max": _np_int(det_counts.max()) if det_counts.size else 0,
            },

            "tracks": {
                "count": int(len(tracks)),
                "length_mean": _np_float(track_lengths.mean()) if track_lengths.size else 0.0,
                "length_median": _np_float(np.median(track_lengths)) if track_lengths.size else 0.0,
                "length_std": _np_float(track_lengths.std()) if track_lengths.size else 0.0,
                "length_min": _np_int(track_lengths.min()) if track_lengths.size else 0,
                "length_max": _np_int(track_lengths.max()) if track_lengths.size else 0,
            },

            "coverage": {
                "detections_in_tracks": int(dets_in_tracks),
                "fraction_detections_linked": float(coverage),
            },

            "motion": {
                "step_dist_mean_px": _np_float(step_dists.mean()) if step_dists.size else 0.0,
                "step_dist_median_px": _np_float(np.median(step_dists)) if step_dists.size else 0.0,
                "step_dist_std_px": _np_float(step_dists.std()) if step_dists.size else 0.0,
                "step_dist_max_px": _np_float(step_dists.max()) if step_dists.size else 0.0,
                "track_net_disp_mean_px": _np_float(track_net_disp.mean()) if track_net_disp.size else 0.0,
                "track_net_disp_median_px": _np_float(np.median(track_net_disp)) if track_net_disp.size else 0.0,
                "track_net_disp_max_px": _np_float(track_net_disp.max()) if track_net_disp.size else 0.0,
            },
        }
        return metrics


    def _save_run_report(self, report: dict, folder: str):
        os.makedirs(folder, exist_ok=True)
        json_path = os.path.join(folder, "run_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        self._log(f"Saved run report: {json_path}")

        # Optional: also write a compact human-readable txt
        txt_path = os.path.join(folder, "run_report.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=== Particle Tracking Run Report ===\n\n")
            for k in ["input_path", "data_shape", "channel"]:
                f.write(f"{k}: {report['settings'].get(k)}\n")
            f.write("\n[fft]\n")
            for k, v in report["settings"]["fft"].items():
                f.write(f"{k}: {v}\n")
            f.write("\n[preprocess]\n")
            for k, v in report["settings"]["preprocess"].items():
                f.write(f"{k}: {v}\n")
            f.write("\n[detection]\n")
            for k, v in report["settings"]["detection"].items():
                f.write(f"{k}: {v}\n")
            f.write("\n[linking]\n")
            for k, v in report["settings"]["linking"].items():
                f.write(f"{k}: {v}\n")

            f.write("\n--- METRICS ---\n")
            for section, d in report["metrics"].items():
                f.write(f"\n[{section}]\n")
                if isinstance(d, dict):
                    for k, v in d.items():
                        f.write(f"{k}: {v}\n")
                else:
                    f.write(f"{d}\n")
        self._log(f"Saved run report: {txt_path}")




def _np_float(x):
    # make numpy scalars JSON safe
    try:
        return float(x)
    except Exception:
        return x

def _np_int(x):
    try:
        return int(x)
    except Exception:
        return x
