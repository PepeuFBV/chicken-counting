import os
import json
import math
import random
from typing import List, Tuple, Dict, Any

from PIL import Image, ImageEnhance
import numpy as np


def load_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], json_path: str):
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def save_image(img: Image.Image, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def _extract_points(shapes: List[Dict]) -> List[Tuple[float, float]]:
    pts = []
    for s in shapes:
        if s.get("shape_type") == "point":
            p = s.get("points", [])
            if p:
                x, y = p[0]
                pts.append((float(x), float(y)))
            else:
                pts.append((0.0, 0.0))
        else:
            # ignore other shapes for now
            pts.append(None)
    return pts


def _write_points_back(shapes: List[Dict], new_pts: List[Any]):
    out = []
    j = 0
    for s in shapes:
        if s.get("shape_type") == "point":
            coord = new_pts[j]
            j += 1
            s_copy = dict(s)
            s_copy["points"] = [[float(coord[0]), float(coord[1])]]
            out.append(s_copy)
        else:
            out.append(s)
    return out


def flip_horizontal(img: Image.Image, points: List[Tuple[float, float]]) -> Tuple[Image.Image, List[Tuple[float, float]]]:
    w, h = img.size
    img2 = img.transpose(Image.FLIP_LEFT_RIGHT)
    pts2 = [(w - x, y) for x, y in points]
    return img2, pts2


def flip_vertical(img: Image.Image, points: List[Tuple[float, float]]) -> Tuple[Image.Image, List[Tuple[float, float]]]:
    w, h = img.size
    img2 = img.transpose(Image.FLIP_TOP_BOTTOM)
    pts2 = [(x, h - y) for x, y in points]
    return img2, pts2


def rotate(img: Image.Image, points: List[Tuple[float, float]], angle: float) -> Tuple[Image.Image, List[Tuple[float, float]]]:
    # rotate around image center, use expand=True and adjust points accordingly
    w, h = img.size
    cx, cy = w / 2.0, h / 2.0
    rad = math.radians(angle)
    cos = math.cos(rad)
    sin = math.sin(rad)

    def rot_point(x, y):
        x_c = x - cx
        y_c = y - cy
        xr = cos * x_c - sin * y_c
        yr = sin * x_c + cos * y_c
        return xr + cx, yr + cy

    # rotated corners to find offset (expand=True behavior)
    corners = [rot_point(0, 0), rot_point(w, 0), rot_point(w, h), rot_point(0, h)]
    min_x = min(p[0] for p in corners)
    min_y = min(p[1] for p in corners)

    img2 = img.rotate(angle, expand=True)

    pts2 = []
    for x, y in points:
        xr, yr = rot_point(x, y)
        pts2.append((xr - min_x, yr - min_y))

    return img2, pts2


def scale(img: Image.Image, points: List[Tuple[float, float]], sx: float, sy: float = None) -> Tuple[Image.Image, List[Tuple[float, float]]]:
    if sy is None:
        sy = sx
    w, h = img.size
    nw, nh = max(1, int(round(w * sx))), max(1, int(round(h * sy)))
    img2 = img.resize((nw, nh), resample=Image.BILINEAR)
    pts2 = [(x * sx, y * sy) for x, y in points]
    return img2, pts2


def translate(img: Image.Image, points: List[Tuple[float, float]], tx: int, ty: int, bg_color=(0, 0, 0)) -> Tuple[Image.Image, List[Tuple[float, float]]]:
    w, h = img.size
    canvas = Image.new("RGB", (w, h), bg_color)
    canvas.paste(img, (tx, ty))
    pts2 = [(x + tx, y + ty) for x, y in points]
    # clip points to image area (optional)
    pts2 = [(max(0, min(w - 1, xx)), max(0, min(h - 1, yy))) for xx, yy in pts2]
    return canvas, pts2


def color_jitter(img: Image.Image, brightness=0.0, contrast=0.0, saturation=0.0) -> Image.Image:
    img2 = img
    if brightness != 0.0:
        factor = 1.0 + brightness * (random.uniform(-1, 1))
        img2 = ImageEnhance.Brightness(img2).enhance(factor)
    if contrast != 0.0:
        factor = 1.0 + contrast * (random.uniform(-1, 1))
        img2 = ImageEnhance.Contrast(img2).enhance(factor)
    if saturation != 0.0:
        factor = 1.0 + saturation * (random.uniform(-1, 1))
        img2 = ImageEnhance.Color(img2).enhance(factor)
    return img2


def add_noise(img: Image.Image, sigma: float = 5.0) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    arr += noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def augment_single(json_path: str, image_root: str, out_root: str, ops: List[Dict[str, Any]]):
    """
    Augment a single image+json pair.

    - `json_path`: path to label JSON
    - `image_root`: directory where image files referenced by JSON live
    - `out_root`: directory where augmented images/jsons will be written
    - `ops`: list of operations to apply in order. Each op is a dict with `name` and params.

    Supported ops: flip_h, flip_v, rotate(angle), scale(sx, sy), translate(tx,ty), color_jitter(...), noise(sigma)
    """
    data = load_json(json_path)
    image_name = data.get("imagePath") or os.path.splitext(os.path.basename(json_path))[0] + ".jpg"
    image_path = os.path.join(image_root, image_name)
    img = load_image(image_path)
    shapes = data.get("shapes", [])

    points = [p for p in _extract_points(shapes) if p is not None]

    # store mapping of original shapes order for writing back
    pts_index = 0

    img_out = img
    pts_out = points

    for op in ops:
        name = op.get("name")
        if name == "flip_h":
            img_out, pts_out = flip_horizontal(img_out, pts_out)
        elif name == "flip_v":
            img_out, pts_out = flip_vertical(img_out, pts_out)
        elif name == "rotate":
            angle = float(op.get("angle", 0.0))
            img_out, pts_out = rotate(img_out, pts_out, angle)
        elif name == "scale":
            sx = float(op.get("sx", 1.0))
            sy = op.get("sy")
            sy = float(sy) if sy is not None else None
            img_out, pts_out = scale(img_out, pts_out, sx, sy)
        elif name == "translate":
            tx = int(op.get("tx", 0))
            ty = int(op.get("ty", 0))
            img_out, pts_out = translate(img_out, pts_out, tx, ty)
        elif name == "color_jitter":
            img_out = color_jitter(img_out, brightness=op.get("brightness", 0.2), contrast=op.get("contrast", 0.2), saturation=op.get("saturation", 0.2))
        elif name == "noise":
            sigma = float(op.get("sigma", 5.0))
            img_out = add_noise(img_out, sigma=sigma)
        else:
            # unknown op
            continue

    # write outputs
    base = os.path.splitext(os.path.basename(image_name))[0]

    def _format_val(v: Any) -> str:
        # produce a compact, filename-safe representation for param values
        if isinstance(v, float):
            # use short representation and replace dot with 'p' to avoid extra dots
            s = f"{v:.6g}"
            return s.replace('.', 'p').replace('-', 'neg')
        if isinstance(v, (int, str)):
            return str(v).replace(' ', '_')
        return str(v)

    def _op_suffix(op: Dict[str, Any]) -> str:
        name = op.get('name')
        if not name:
            return ""
        parts = [name]
        # include sorted params (except name) so ops with different params yield different suffixes
        for k in sorted(op.keys()):
            if k == 'name':
                continue
            parts.append(f"{k}{_format_val(op[k])}")
        return "_" + "_".join(parts)

    ops_suffix = "".join([_op_suffix(o) for o in ops])
    new_image_name = f"{base}{ops_suffix}.jpg"
    new_json_name = f"{base}{ops_suffix}.json"

    out_image_path = os.path.join(out_root, new_image_name)
    out_json_path = os.path.join(out_root, new_json_name)

    save_image(img_out, out_image_path)

    # write json: keep same structure but update imagePath and shapes
    new_data = dict(data)
    new_data["imagePath"] = new_image_name
    new_data["shapes"] = _write_points_back(shapes, pts_out)
    save_json(new_data, out_json_path)


def augment_dataset(json_dir: str, image_root: str, out_root: str, ops_list: List[List[Dict[str, Any]]], limit: int = None):
    """
    Apply a list of augmentation pipelines to every json in `json_dir`.
    `ops_list` is a list of op sequences; for each json, each ops sequence will be applied producing one augmented sample.
    """
    os.makedirs(out_root, exist_ok=True)
    files = [f for f in os.listdir(json_dir) if f.lower().endswith('.json')]
    if limit is not None:
        files = files[:limit]
    for fname in files:
        json_path = os.path.join(json_dir, fname)
        for ops in ops_list:
            try:
                augment_single(json_path, image_root, out_root, ops)
            except Exception as e:
                print(f"Failed augment {fname} with {ops}: {e}")


if __name__ == "__main__":
    # simple example usage
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", default="./data/dataset")
    parser.add_argument("--image_root", default="./data/dataset")
    parser.add_argument("--out_root", default="./data/augmented")
    args = parser.parse_args()

    # define simple pipelines
    pipelines = [
        [{"name": "flip_h"}],
        [{"name": "flip_v"}],
        [{"name": "rotate", "angle": 90}],
        [{"name": "rotate", "angle": 180}],
        [{"name": "scale", "sx": 0.9}],
        [{"name": "scale", "sx": 1.1}],
        [{"name": "color_jitter", "brightness": 0.2, "contrast": 0.2, "saturation": 0.2}],
        [{"name": "noise", "sigma": 8.0}],
    ]

    augment_dataset(args.json_dir, args.image_root, args.out_root, pipelines, limit=50)
