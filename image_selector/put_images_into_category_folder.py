#!/usr/bin/env python3
"""
Image Categorizer Script
Organizes images into category folders based on od_type_primary from JSONL metadata
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from collections import defaultdict

def load_jsonl_data(jsonl_file):
    """Load data from JSONL file"""
    data = []
    print(f"Loading JSONL file: {jsonl_file}")

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Error parsing line {line_num}: {e}")
                    continue

    print(f"Loaded {len(data)} records from JSONL file")
    return data

def categorize_images(jsonl_data, images_folder, output_folder):
    """Categorize images into folders based on od_type_primary"""

    # Group data by od_type_primary
    categories = defaultdict(list)

    for item in jsonl_data:
        od_type = item.get('od_type_primary', 'unknown')
        categories[od_type].append(item)

    print(f"\nFound {len(categories)} categories:")
    for category, items in categories.items():
        print(f"  {category}: {len(items)} images")

    # Create output folder if it doesn't exist
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput folder: {output_folder}")

    # Process each category
    total_processed = 0
    images_folder = Path(images_folder)

    for category, items in categories.items():
        print(f"\nProcessing category: {category}")

        # Create category folder
        category_folder = output_folder / category
        category_folder.mkdir(exist_ok=True)

        # Create images subfolder
        images_subfolder = category_folder / "images"
        images_subfolder.mkdir(exist_ok=True)

        category_data = []

        # Process each item in category
        for item in items:
            test_case_id = item.get('test_case_id')
            if not test_case_id:
                print(f"Warning: No test_case_id found for item, skipping")
                continue

            # Find corresponding image file
            image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
            image_file = None

            for ext in image_extensions:
                potential_file = images_folder / f"{test_case_id}{ext}"
                if potential_file.exists():
                    image_file = potential_file
                    break

            if not image_file:
                print(f"Warning: Image file not found for {test_case_id}")
                continue

            # Copy image to category folder
            try:
                dest_file = images_subfolder / image_file.name
                shutil.copy2(image_file, dest_file)
                category_data.append(item)
                total_processed += 1
                print(f"  ✓ Copied: {test_case_id}{image_file.suffix}")
            except Exception as e:
                print(f"  ✗ Error copying {test_case_id}: {e}")

        # Create category JSONL file
        if category_data:
            jsonl_output = category_folder / f"{category}_dataset.jsonl"
            with open(jsonl_output, 'w', encoding='utf-8') as f:
                for item in category_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

            print(f"  ✓ Created: {jsonl_output.name} ({len(category_data)} records)")

    print("\n=== Summary ===")
    print(f"Total images processed: {total_processed}")
    print(f"Categories created: {len(categories)}")
    print(f"Output location: {output_folder}")

    # Show folder structure
    print("\nFolder structure created:")
    for category in sorted(categories.keys()):
        category_folder = output_folder / category
        if category_folder.exists():
            images_count = len(list((category_folder / "images").glob("*")))
            jsonl_file = category_folder / f"{category}_dataset.jsonl"
            jsonl_count = sum(1 for _ in open(jsonl_file)) if jsonl_file.exists() else 0
            print(f"  {category}/")
            print(f"    images/ ({images_count} files)")
            print(f"    {category}_dataset.jsonl ({jsonl_count} records)")

def main():
    parser = argparse.ArgumentParser(
        description='Categorize images into folders based on od_type_primary from JSONL metadata',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python put_images_into_category_folder.py -i images_folder -j metadata.jsonl -o output_folder
  python put_images_into_category_folder.py --images ./selected_images --jsonl ./selected_dataset.jsonl --output ./categorized_images
        """
    )

    parser.add_argument(
        '-i', '--images',
        required=True,
        help='Path to folder containing images'
    )

    parser.add_argument(
        '-j', '--jsonl',
        required=True,
        help='Path to JSONL file containing image metadata'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output folder path (will be created if it doesn\'t exist)'
    )

    args = parser.parse_args()

    # Validate input paths
    if not os.path.exists(args.images):
        print(f"Error: Images folder '{args.images}' does not exist")
        return 1

    if not os.path.exists(args.jsonl):
        print(f"Error: JSONL file '{args.jsonl}' does not exist")
        return 1

    # Check if images folder contains any image files
    images_folder = Path(args.images)
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(images_folder.glob(ext))

    if not image_files:
        print(f"Warning: No image files found in '{args.images}'")

    # Load and process data
    try:
        jsonl_data = load_jsonl_data(args.jsonl)
        if not jsonl_data:
            print("Error: No valid data found in JSONL file")
            return 1

        categorize_images(jsonl_data, args.images, args.output)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
