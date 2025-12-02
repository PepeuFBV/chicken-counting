from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Optional, Iterator, Dict, Any

try:
    from PIL import Image
except Exception:  # pragma: no cover - pillow optional
    Image = None  # type: ignore


class DatasetLoader:
    """
    Load images and point annotations exported by LabelMe (JSON files).

    Behavior and assumptions:
    - Scans a provided directory for `.json` files (recursively) and treats each
      JSON as an annotation for one image.
    - Expects per-file annotations in LabelMe format: a top-level `shapes`
      list with items where `shape_type == 'point'` and `label == 'chicken'`.
    - Uses the JSON `imagePath` when present; otherwise infers image filename
      by replacing `.json` with common image extensions.
    - `load_images=True` will attempt to load image files as PIL `Image` objects.

    Example:
        loader = DatasetLoader('data/dataset', load_images=False)
        for sample in loader:
            image_or_path, points = sample['image'], sample['points']
    """

    def __init__(
        self,
        data_dir: str | Path,
        load_images: bool = False,
        allowed_labels: Optional[List[str]] = None,
        image_exts: Optional[List[str]] = None,
    ) -> None:
        self.root = Path(data_dir)
        if not self.root.exists():
            raise ValueError(f"data directory does not exist: {self.root}")

        self.load_images = load_images
        if load_images and Image is None:
            raise RuntimeError("Pillow is required to load images. Install with `pip install pillow`.")

        self.allowed_labels = allowed_labels or ["chicken"]
        self.image_exts = image_exts or ["jpg", "jpeg", "png"]

        # internal list of samples
        self.samples: List[Dict[str, Any]] = []
        self._scan()

    def _scan(self) -> None:
        json_files = sorted(self.root.rglob("*.json"))
        for j in json_files:
            try:
                with open(j, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                # skip unreadable files
                continue

            shapes = data.get("shapes", [])
            points: List[Tuple[float, float]] = []
            for s in shapes:
                if s.get("shape_type") != "point":
                    continue
                if s.get("label") not in self.allowed_labels:
                    continue
                pts = s.get("points")
                if not pts:
                    continue
                # LabelMe stores a list of points for the shape — for point shapes we
                # expect a single point; use the first pair.
                x, y = pts[0][0], pts[0][1]
                points.append((float(x), float(y)))

            # determine image path
            image_path = None
            image_path_field = data.get("imagePath")
            if image_path_field:
                candidate = j.parent / image_path_field
                if candidate.exists():
                    image_path = candidate
                else:
                    # try relative to root
                    candidate2 = self.root / image_path_field
                    if candidate2.exists():
                        image_path = candidate2

            if image_path is None:
                # try same base name with known extensions
                for ext in self.image_exts:
                    candidate = j.with_suffix(f".{ext}")
                    if candidate.exists():
                        image_path = candidate
                        break

            sample: Dict[str, Any] = {
                "json_path": j,
                "image_path": image_path,
                "points": points,
                "count": len(points),
            }

            if self.load_images and image_path is not None:
                try:
                    sample["image"] = Image.open(image_path).convert("RGB")
                except Exception:
                    sample["image"] = None

            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for s in self.samples:
            yield s

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

    def get_counts(self) -> List[int]:
        """Return list of counts per sample (number of annotated chickens)."""
        return [s["count"] for s in self.samples]

    def as_list(self) -> List[Dict[str, Any]]:
        """Return the internal list of samples for external usage."""
        return list(self.samples)

    def find_missing_images(self) -> List[Path]:
        """Return image paths that were not found when scanning (None results)."""
        missing: List[Path] = []
        for s in self.samples:
            if s["image_path"] is None:
                missing.append(s["json_path"])
        return missing
    
    def show_examples(self, save_dir: Optional[Path] = None, max_examples: int = 5) -> None:
        """Display or save first few images with annotated points (requires PIL).

        If `save_dir` is provided, generated plots are saved there instead of
        opening GUI windows. `save_dir` may be absolute or relative to the
        dataset root (self.root).
        """
        if Image is None:
            raise RuntimeError("Pillow is required to show images. Install with `pip install pillow`.")

        import matplotlib.pyplot as plt

        for i, s in enumerate(self.samples):
            if s["image_path"] is None:
                continue
            if self.load_images:
                img = s.get("image")
            else:
                try:
                    img = Image.open(s["image_path"]).convert("RGB")
                except Exception:
                    continue
            if img is None:
                continue
            plt.figure()
            plt.imshow(img)
            x_pts = [p[0] for p in s["points"]]
            y_pts = [p[1] for p in s["points"]]
            plt.scatter(x_pts, y_pts, c="red", s=10)
            plt.title(f"Sample {i}: {s['image_path'].name} - Count: {s['count']}")
            plt.axis("off")

            if save_dir is not None:
                out_dir = Path(save_dir)
                if not out_dir.is_absolute():
                    out_dir = self.root / out_dir
                out_dir.mkdir(parents=True, exist_ok=True)
                # use json basename + index to make filenames unique
                base = s.get("json_path") and s["json_path"].stem or f"sample_{i}"
                out_path = out_dir / f"{i:03d}_{base}.png"
                plt.savefig(out_path, bbox_inches="tight")
                plt.close()
            else:
                plt.show()

            if i >= (max_examples - 1):
                break


__all__ = ["DatasetLoader"]
