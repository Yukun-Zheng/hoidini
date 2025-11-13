import os

# Get the project root directory (where this file is located relative to hoidini/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default paths relative to the Hugging Face dataset clone
GRAB_DATA_PATH = os.environ.get(
    "GRAB_DATA_PATH",
    os.path.join(
        PROJECT_ROOT, "hoidini_data", "datasets", "GRAB_RETARGETED_compressed"
    ),
)
MANO_SMPLX_VERTEX_IDS_PATH = os.environ.get(
    "MANO_SMPLX_VERTEX_IDS_PATH",
    os.path.join(PROJECT_ROOT, "hoidini_data", "datasets", "MANO_SMPLX_vertex_ids.pkl"),
)
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(
        PROJECT_ROOT,
        "hoidini_data",
        "models",
        "cphoi_05011024_c15p100_v0",
        "model000120000.pt",
    ),
)
SMPL_MODELS_DATA = os.environ.get(
    "SMPL_MODELS_DATA",
    os.path.join(PROJECT_ROOT, "hoidini_data", "smpl_models"),
)

# Optional datasets (can be downloaded separately if needed)
GRAB_ORIG_DATA_PATH = os.environ.get("GRAB_ORIG_DATA_PATH", "/path/to/GRAB/original")
HUMANML3D_DATASET_PATH = os.environ.get("HUMANML3D_DATASET_PATH", "/path/to/HumanML3D")
OMOMO_DATA_PATH = os.environ.get("OMOMO_DATA_PATH", "/path/to/OMOMO")

# Additional dataset locations used by utilities and loaders
AMASS_EXTRACTED_PATH = os.environ.get(
    "AMASS_EXTRACTED_PATH",
    os.path.join(PROJECT_ROOT, "hoidini_data", "datasets", "AMASS_SMPL-X_extracted"),
)
AMASS_RIFKE_INPUTS_PATH = os.environ.get(
    "AMASS_RIFKE_INPUTS_PATH",
    os.path.join(PROJECT_ROOT, "hoidini_data", "datasets", "Data_AMASS_smplrifke_inputs"),
)
GRAB_RIFKE_INPUTS_PATH = os.environ.get(
    "GRAB_RIFKE_INPUTS_PATH",
    os.path.join(PROJECT_ROOT, "hoidini_data", "datasets", "Data_GRAB_smplrifke_inputs"),
)
HML3D_GRAB_SMPLRIFKE_INPUTS_PATH = os.environ.get(
    "HML3D_GRAB_SMPLRIFKE_INPUTS_PATH",
    os.path.join(
        PROJECT_ROOT,
        "hoidini_data",
        "datasets",
        "Data_SMPLX_HML3D_GRAB_smplrifke_inputs",
    ),
)
GRAB_RETARGETED_DATA_PATH = os.environ.get(
    "GRAB_RETARGETED_DATA_PATH",
    GRAB_DATA_PATH,
)

# Optional results references
EVAL_RESULTS_JSON_PATH = os.environ.get(
    "EVAL_RESULTS_JSON_PATH",
    os.path.join(
        PROJECT_ROOT,
        "hoidini_data",
        "results",
        "cphoi_paper_experiments_fixed_keep_object_static",
        "result_paths_per_experiment.json",
    ),
)
