import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw, ImageFont
from src.data_treatment.augmentation import (
    load_json, load_image, flip_horizontal, flip_vertical,
    rotate, scale, translate, color_jitter, add_noise, _extract_points
)


def draw_points_on_image(img, points, color=(255, 0, 0), radius=5):
    """Draw points on image with circles."""
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    
    for x, y in points: # circle
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=color,
            outline=(255, 255, 0),
            width=2
        )
    
    return img_copy


def test_augmentation(image_path, json_path=None):
    """
    Test various augmentations on a single image and save results to disk.
    
    Args:
        image_path: Path to image file
        json_path: Optional path to JSON annotation file
    """
    img = load_image(image_path)
    img_name = Path(image_path).stem
    
    points = []
    if json_path and os.path.exists(json_path):
        data = load_json(json_path)
        points = [p for p in _extract_points(data.get("shapes", [])) if p is not None]
        print(f"Loaded {len(points)} points from {json_path}")
    else:
        print("No JSON file provided, proceeding without point annotations")
    
    # define augmentations to test
    augmentations = [
        ("original", lambda i, p: (i, p)),
        ("flip_horizontal", flip_horizontal),
        ("flip_vertical", flip_vertical),
        ("rotate_90", lambda i, p: rotate(i, p, 90)),
        ("rotate_180", lambda i, p: rotate(i, p, 180)),
        ("scale_0.8", lambda i, p: scale(i, p, 0.8)),
        ("scale_1.2", lambda i, p: scale(i, p, 1.2)),
        ("color_jitter", lambda i, p: (color_jitter(i, 0.3, 0.3, 0.3), p)),
        ("noise", lambda i, p: (add_noise(i, 25.0), p)),
    ]
    
    output_dir = Path(__file__).parent.parent / "data" / "test-augmentations" / img_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nTesting {len(augmentations)} augmentations on {img_name}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    for name, aug_func in augmentations:
        try:
            aug_img, aug_points = aug_func(img, points)
            
            if aug_points:
                aug_img_with_points = draw_points_on_image(aug_img, aug_points)
                point_count = len(aug_points)
            else:
                aug_img_with_points = aug_img
                point_count = 0
            
            output_path = output_dir / f"{name}.jpg"
            aug_img_with_points.save(output_path, quality=95)
            
            print(f"✓ {name:20s} - Size: {aug_img.size}, Points: {point_count} -> {output_path.name}")
            
        except Exception as e:
            print(f"✗ {name:20s} - Error: {e}")
    
    print("=" * 60)
    print(f"✓ Saved {len(augmentations)} augmented images to: {output_dir}")
    print(f"\nYou can view them with: xdg-open {output_dir}")


def test_combinations(image_path, json_path=None):
    """Test combinations of augmentations and save to disk."""
    img = load_image(image_path)
    img_name = Path(image_path).stem
    points = []
    
    if json_path and os.path.exists(json_path):
        data = load_json(json_path)
        points = [p for p in _extract_points(data.get("shapes", [])) if p is not None]
    
    combinations = [
        ("flip_h_rotate_90", [
            lambda i, p: flip_horizontal(i, p),
            lambda i, p: rotate(i, p, 90)
        ]),
        ("scale_color_jitter", [
            lambda i, p: scale(i, p, 0.9),
            lambda i, p: (color_jitter(i, 0.2, 0.2, 0.2), p)
        ]),
        ("rotate_noise", [
            lambda i, p: rotate(i, p, 45),
            lambda i, p: (add_noise(i, 8.0), p)
        ]),
        ("flip_h_flip_v", [
            lambda i, p: flip_horizontal(i, p),
            lambda i, p: flip_vertical(i, p)
        ]),
    ]
    
    output_dir = Path(__file__).parent.parent / "data" / "test-augmentations" / f"{img_name}_combinations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    orig_img_with_points = draw_points_on_image(img, points) if points else img
    orig_path = output_dir / "original.jpg"
    orig_img_with_points.save(orig_path, quality=95)
    
    print(f"\nTesting augmentation combinations")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    print(f"✓ {'original':30s} - Size: {img.size}, Points: {len(points)} -> {orig_path.name}")
    
    for name, aug_funcs in combinations:
        try:
            current_img = img
            current_points = points
            
            for aug_func in aug_funcs:
                current_img, current_points = aug_func(current_img, current_points)
            
            aug_img_with_points = draw_points_on_image(current_img, current_points) if current_points else current_img
            
            output_path = output_dir / f"{name}.jpg"
            aug_img_with_points.save(output_path, quality=95)
            
            print(f"✓ {name:30s} - Size: {current_img.size}, Points: {len(current_points)} -> {output_path.name}")
            
        except Exception as e:
            print(f"✗ {name:30s} - Error: {e}")
    
    print("=" * 60)
    print(f"✓ Saved {len(combinations) + 1} images to: {output_dir}")
    print(f"\nYou can view them with: xdg-open {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Quick test for visualizing data augmentations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with image only
  python tests/test_augmentations.py --image data/dataset/image1.jpg
  
  # Test with image and JSON annotations
  python tests/test_augmentations.py --image data/dataset/image1.jpg --json data/dataset/image1.json
  
  # Test combinations
  python tests/test_augmentations.py --image data/dataset/image1.jpg --combinations
  
  # Auto-find first available image in dataset
  python tests/test_augmentations.py --auto
        """
    )
    
    parser.add_argument("--image", "-i", help="Path to image file")
    parser.add_argument("--json", "-j", help="Path to JSON annotation file (optional)")
    parser.add_argument("--combinations", "-c", action="store_true", 
                       help="Test augmentation combinations")
    parser.add_argument("--auto", "-a", action="store_true",
                       help="Auto-find first image in data/dataset/")
    
    args = parser.parse_args()
    
    if args.auto:
        dataset_dir = Path(__file__).parent.parent / "data" / "dataset"
        if dataset_dir.exists():
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                images = list(dataset_dir.rglob(f"*{ext}"))
                if images:
                    args.image = str(images[0])

                    json_path = images[0].with_suffix('.json')
                    if json_path.exists():
                        args.json = str(json_path)
                    print(f"Auto-selected: {args.image}")
                    if args.json:
                        print(f"Found JSON: {args.json}")
                    break
            
            if not args.image:
                print("Error: No images found in data/dataset/")
                return
        else:
            print(f"Error: Dataset directory not found: {dataset_dir}")
            return
    
    if not args.image:
        parser.print_help()
        print("\nError: Please provide --image or use --auto to find an image automatically")
        return
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    if args.combinations:
        test_combinations(args.image, args.json)
    else:
        test_augmentation(args.image, args.json)


if __name__ == "__main__":
    main()
