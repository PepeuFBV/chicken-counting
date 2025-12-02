from pathlib import Path
import sys

# allow running this script directly (adds repo root so `src` is importable)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_treatment.dataset_loader import DatasetLoader


def main() -> None:
    data_dir = Path(__file__).parents[1] / "data" / "dataset"
    loader = DatasetLoader(data_dir, load_images=False)

    print(f"Found {len(loader)} annotated samples")
    counts = loader.get_counts()
    print(f"Total annotated chickens (sum): {sum(counts)}")
    print(f"Per-image counts (first 10): {counts[:10]}")

    missing = loader.find_missing_images()
    if missing:
        print("Warning: some JSON files did not resolve to an image path:")
        for p in missing[:10]:
            print(" -", p)

    # example: iterate samples and show first sample points
    for i, s in enumerate(loader):
        print(f"Sample {i}: image={s['image_path']} count={s['count']}")
        print(" Points:", s["points"][:5])
        if i >= 2:
            break

    saveDirPath = "examples"
    try:
        loader.show_examples(save_dir=Path(saveDirPath), max_examples=5)
        print("Saved example images to: " + saveDirPath)
    except RuntimeError as exc:
        print("Skipping image display:", exc)


if __name__ == "__main__":
    main()
