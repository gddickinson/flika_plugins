"""Hand-made grouping of the plugins by what they are for (the only curated field in the catalogue).

Keys are plugin folder names; the same folder name in EXPERIMENTAL/, OLD/ etc. gets the same topic. Anything not
listed falls into "other". Status groups (top level, EXPERIMENTAL, NOT_WORKING, ...) come from the folders themselves.
"""

TOPICS = [
    ("spt", "Single-particle tracking and localization",
     "Detect, localize and link single molecules (ThunderSTORM, U-Track, pynsight) and analyse the tracks."),
    ("calcium", "Calcium signals and puffs",
     "Detect, simulate, map and analyse local calcium events and noise in fluorescence movies."),
    ("lightsheet", "Light sheet and volumes",
     "Load light-sheet stacks, reshape them into 4D volumes and view them."),
    ("process", "Image processing",
     "Transforms, alignment, flash and frame removal, background subtraction and segmentation."),
    ("roi", "ROIs, profiles and annotation",
     "ROI tools, line scans, kymographs, masks and on-image annotation."),
    ("display", "Display and export",
     "Overlays, timestamps, scale bars and movie export."),
    ("other", "Other tools", "Loaders, package management and lab-specific tools."),
]

_BY_FOLDER = {
    "spt": ["spt_batch_analysis", "thunderstorm_flika", "locsAndTracksPlotter", "locsAndTracksPlotter_original",
            "pynsight", "pynsight_GDedit", "advanced_particle_tracking", "utrack_detection", "utrack_tracking",
            "tracking_results_plotter", "piezo1_cluster_analyzer", "translateAndScale", "overlayMultipleRecordings",
            "puncta_analyzer", "spt_batch_analysis_Feb12", "dipoleAnalysis", "tirf_analysis_suite"],
    "calcium": ["detect_puffs", "detect_puffs_GDedit", "simulatePuff", "puffMapper", "fft_Chunker",
                "ca_event_detector_flika", "ca_event_labeler_flika", "calcium_noise_analysis", "flicker_analyzer",
                "flicker_detector", "GlobalAnalysisPlugin", "synapse_plugin", "mepp_simulator", "scaledAverageSubtract"],
    "lightsheet": ["light_sheet_analyzer", "light_sheet_analyzer_GDedit", "lightSheet_tiffLoader", "volumeSlider",
                   "volumeSlider_2", "onTheFly_Reconstruction"],
    "process": ["image_transforms", "advanced_beam_splitter", "flash_remover", "frame_remover", "imageSegmentation",
                "cell_edge_movement_analysis", "quantimus"],
    "roi": ["centerSurroundROI", "roiExtras", "linescan", "kymograph_analyzer", "mask_editor", "annotator",
            "scribble"],
    "display": ["overlay", "timestamp", "videoExporter", "openCV"],
    "other": ["fileLoader", "packageManager", "neuroLab_207", "rodentTracker"],
}
TOPIC_OF = {folder: t for t, folders in _BY_FOLDER.items() for folder in folders}


def topic_of(folder: str) -> str:
    return TOPIC_OF.get(folder.strip(), "other")
