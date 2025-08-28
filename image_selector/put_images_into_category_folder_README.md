# Image Categorization Script

This script automatically organizes images into category folders based on the `od_type_primary` field from JSONL metadata.

## Features

- **Automatic Classification**: Creates category folders based on the `od_type_primary` field in JSONL
- **Image Copying**: Copies corresponding images to their respective category folders
- **JSONL Generation**: Creates separate JSONL files for each category
- **Path Validation**: Automatically validates input paths
- **Detailed Output**: Displays classification statistics and folder structure

## Usage

### Basic Syntax

```bash
python put_images_into_category_folder.py -i <images_folder> -j <jsonl_file> -o <output_folder>
```

### Parameters

- `-i, --images`: Path to images folder (required)
- `-j, --jsonl`: Path to JSONL metadata file (required)
- `-o, --output`: Output folder path (required, will be created if it doesn't exist)

### Usage Examples

#### Example 1: Using data from current folder
```bash
python put_images_into_category_folder.py \
    --images selected_images_v2/selected_images \
    --jsonl selected_images_v2/selected_dataset.jsonl \
    --output categorized_images
```

#### Example 2: Using absolute paths
```bash
python put_images_into_category_folder.py \
    -i /path/to/images/folder \
    -j /path/to/metadata.jsonl \
    -o /path/to/output/folder
```

#### Example 3: Using relative paths
```bash
python put_images_into_category_folder.py \
    -i ../image_preview/downloaded_images \
    -j ../dataset/metadata.jsonl \
    -o ./organized_images
```

## Output Structure

The script creates the following folder structure:

```
output_folder/
├── animal/
│   ├── images/           # Animal category images
│   │   ├── OD-FN-07749.png
│   │   ├── OD-REP-09464.png
│   │   └── ...
│   └── animal_dataset.jsonl  # Animal category metadata
├── person/
│   ├── images/           # Person category images
│   │   ├── OD-POS-01498.png
│   │   ├── OD-EDGE-08253.png
│   │   └── ...
│   └── person_dataset.jsonl  # Person category metadata
├── vehicle/
│   ├── images/           # Vehicle category images
│   │   ├── OD-FN-06823.png
│   │   ├── OD-POS-00146.png
│   │   └── ...
│   └── vehicle_dataset.jsonl # Vehicle category metadata
└── package/
    ├── images/           # Package category images
    │   ├── OD-NEG-04629.png
    │   ├── OD-POS-00380.png
    │   └── ...
    └── package_dataset.jsonl # Package category metadata
```

## Classification Criteria

The script classifies based on the `od_type_primary` field in each JSONL record:

- **animal**: Animal category images
- **person**: Person category images
- **vehicle**: Vehicle category images
- **package**: Package category images

## File Matching Rules

- Image filenames must match the `test_case_id` field in JSONL (without extension)
- Supported image formats: `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`
- Script automatically finds matching image files

## Output Information

The script displays:

1. **Loading Status**: JSONL file loading progress
2. **Category Statistics**: Number of images per category
3. **Processing Progress**: Copy status for each image
4. **Final Summary**: Total processed count and folder structure

### Example Output

```
Loading JSONL file: selected_images_v2/selected_dataset.jsonl
Loaded 149 records from JSONL file

Found 4 categories:
  animal: 27 images
  person: 41 images
  vehicle: 61 images
  package: 20 images

Processing category: animal
  ✓ Copied: OD-FN-07749.png
  ✓ Copied: OD-REP-09464.png
  ...

=== Summary ===
Total images processed: 149
Categories created: 4
Output location: categorized_images

Folder structure created:
  animal/
    images/ (27 files)
    animal_dataset.jsonl (27 records)
  person/
    images/ (41 files)
    person_dataset.jsonl (41 records)
  ...
```

## Important Notes

1. **Path Validation**: Script checks if input paths exist
2. **Output Folder**: Automatically created if it doesn't exist
3. **File Overwrite**: Existing files with same names will be overwritten
4. **Error Handling**: Shows warnings for missing images but continues processing
5. **Encoding Support**: Supports UTF-8 encoded JSONL files

## Troubleshooting

### Common Issues

1. **"Images folder does not exist"**
   - Check if the images folder path is correct
   - Use absolute paths or ensure relative paths are valid

2. **"JSONL file does not exist"**
   - Check if the JSONL file path is correct
   - Ensure the file has `.jsonl` extension

3. **"Warning: Image file not found"**
   - Check if image filenames match `test_case_id` values
   - Confirm images are in supported formats

4. **"No valid data found in JSONL file"**
   - Check JSONL file format
   - Ensure each line is a valid JSON object

## Technical Details

- **Programming Language**: Python 3.x
- **Dependencies**: os, json, argparse, shutil, pathlib, collections
- **File Handling**: Uses `shutil.copy2()` to preserve file metadata
- **JSON Processing**: Uses `json.loads()` with Chinese character support
- **Path Handling**: Uses `pathlib.Path` for cross-platform compatibility

## Advanced Usage

You can modify the script to customize classification logic:

- Change the `od_type_primary` field name
- Add new classification criteria
- Modify folder naming rules
- Add more metadata fields to output JSONL

## License

This script is provided for educational and development purposes only.
