import argparse
import sys
from pathlib import Path
from typing import List, Optional

# make project root importable (repo root, so `src` package is discoverable)
PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT))

from src.data_treatment.dataset_loader import DatasetLoader
from src.data_treatment.augmentation import augment_dataset


def default_pipelines() -> List[List[dict]]:
    rotation_intervals = [15 * i for i in range(1, 13)]  # 15, 30, ..., 180
    rotations = [[{"name": "rotate", "angle": angle}] for angle in rotation_intervals]
    
    scale_factors = [0.5, 0.75, 0.9, 1.1, 1.25, 1.5]
    scales = [[{"name": "scale", "sx": sf, "sy": sf}] for sf in scale_factors]
    
    noise_levels = [1.0, 2.0, 4.0, 8.0, 12.0]
    noises = [[{"name": "noise", "sigma": sigma}] for sigma in noise_levels]
    
    brightness_levels = [0.2, 0.4, 0.6, 0.8]
    contrast_levels = [0.2, 0.4, 0.6, 0.8]
    saturation_levels = [0.2, 0.4, 0.6, 0.8]
    color_jitters = [
        [{
            "name": "color_jitter",
            "brightness": b,
            "contrast": c,
            "saturation": s,
        }]
        for b in brightness_levels
        for c in contrast_levels
        for s in saturation_levels
    ]
    
    return [
        [{"name": "flip_h"}],
        [{"name": "flip_v"}],
        *rotations,
        *scales,
        *color_jitters,
        *noises,
    ]


def preprocess_and_get_loader(
    data_dir: Path | str,
    out_root: Path | str,
    pipelines: List[List[dict]],
    limit: Optional[int] = None,
) -> DatasetLoader:
    """Run augmentations defined in `pipelines` on `data_dir` and return a
    DatasetLoader pointed at `out_root` containing the augmented annotation files.

    This function prints per-image augmented counts and a total summary.
    """
    data_dir = Path(data_dir)
    out_root = Path(out_root)

    loader = DatasetLoader(data_dir, load_images=False)
    total = len(loader)
    print(f"Found {total} annotation files in {data_dir}")

    # delegate to augment_dataset which handles applying pipelines to all
    # JSONs. augment_dataset now scans recursively and uses each JSON's parent
    # folder as the image root so relative `imagePath` fields continue to work.
    augment_dataset(str(data_dir), str(data_dir), str(out_root), pipelines, limit=limit, max_combinations=1)

    # return a loader pointed at augmented folder
    augmented_loader = DatasetLoader(out_root, load_images=False)
    print(f"Processed {min(total, limit) if limit is not None else total} samples. Total augmented files generated: {len(augmented_loader)}.")
    return augmented_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Load dataset and run augmentations")
    parser.add_argument("--data_dir", default="data/dataset", help="root directory containing JSONs and images")
    parser.add_argument("--out_root", default="data/augmented", help="directory to write augmented images/jsons")
    parser.add_argument("--limit", type=int, default=None, help="limit number of samples to process")
    parser.add_argument("--pipelines", default=None, help="(optional) python file exporting `PIPELINES` list to override defaults")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # load pipelines
    if args.pipelines:
        pipelines_path = Path(args.pipelines)
        if not pipelines_path.exists():
            print(f"pipelines file not found: {pipelines_path}")
            return
        # import pipelines file as a module
        import importlib.util

        spec = importlib.util.spec_from_file_location("user_pipelines", str(pipelines_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        pipelines = getattr(mod, "PIPELINES", None)
        if pipelines is None:
            print("Pipelines file must define a top-level `PIPELINES` variable (list of op-lists)")
            return
    else:
        pipelines = default_pipelines()

    # run preprocessing/augmentation and get a DatasetLoader for augmented data
    augmented_loader = preprocess_and_get_loader(data_dir, out_root, pipelines, args.limit)

    print(f"Done. Augmented dataset ready: {len(augmented_loader)} annotation files in {out_root}")


if __name__ == "__main__":
    main()
