# Image Selector Server

A Flask-based web server that displays images from an object detection dataset with their corresponding metadata, prompts, and risk tags. Allows users to select or drop images and save selections to create a new curated dataset.

## Features

- **Split-panel UI**: Left side shows images, right side shows metadata
- **Navigation**: Previous/Next buttons to browse through images
- **Selection System**: Select or drop images with visual feedback
- **Keyboard shortcuts**: Use arrow keys to navigate, S to select, D to drop
- **Rich metadata display**: Shows all JSONL data in human-readable format
- **Dataset Export**: Save selected images and generate new JSONL file
- **Responsive design**: Clean, modern interface that works on different screen sizes
- **Image serving**: Direct access to images with proper caching

## Requirements

- Python 3.7+
- Flask
- Images in PNG, JPG, JPEG, GIF, or BMP format
- JSONL file with corresponding metadata

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python host_image_selector_server.py --images ../image_preview/downloaded_images --jsonl ../dataset/od_synth_cases_10000_cctv_v1_.jsonl
```

### Command Line Options

- `-i, --images`: Path to folder containing images (required)
- `-j, --jsonl`: Path to JSONL file containing image metadata (required)
- `-o, --output`: Output folder for selected images and new JSONL file (default: selected_images)
- `-p, --port`: Port to run the server on (default: 5001)
- `--host`: Host to bind the server to (default: 127.0.0.1)
- `--debug`: Run in debug mode

### Examples

```bash
# Run on default port 5001
python host_image_selector_server.py -i downloaded_images -j dataset/od_synth_cases_10000_cctv_v1_.jsonl

# Run on custom port
python host_image_selector_server.py -i images_folder -j metadata.jsonl --port 8080

# Run on all interfaces
python host_image_selector_server.py -i images_folder -j metadata.jsonl --host 0.0.0.0

# Run in debug mode
python host_image_selector_server.py -i images_folder -j metadata.jsonl --debug

# Specify custom output folder
python host_image_selector_server.py -i images_folder -j metadata.jsonl --output my_selected_dataset
```

## How It Works

1. **Data Loading**: The server loads metadata from the JSONL file and scans the image folder
2. **Matching**: Images are matched with metadata based on filename (without extension)
3. **Web Interface**: A Flask server provides a web UI with API endpoints
4. **Selection**: Users can select or drop images using buttons or keyboard shortcuts
5. **Export**: Selected images are copied to output folder and new JSONL is generated

## File Structure

The script expects:
- **Images**: Files with extensions `.png`, `.jpg`, `.jpeg`, `.gif`, or `.bmp`
- **Metadata**: JSONL file where each line is a JSON object with a `test_case_id` field
- **Matching**: Image filenames (without extension) must match `test_case_id` values

## API Endpoints

- `GET /`: Main web interface
- `GET /api/current`: Get current image data
- `GET /api/next`: Go to next image
- `GET /api/prev`: Go to previous image
- `GET /api/goto/<index>`: Go to specific image index
- `GET /api/select/<test_case_id>`: Select an image
- `GET /api/drop/<test_case_id>`: Drop an image
- `GET /api/save`: Save selections and generate new dataset
- `GET /images/<filename>`: Serve image files

## Keyboard Shortcuts

- **Left Arrow**: Previous image
- **Right Arrow**: Next image
- **S**: Select current image
- **D**: Drop current image

## Selection Workflow

1. **Navigate** through images using arrow keys or navigation buttons
2. **Select** images you want to keep by clicking "Select" button or pressing 'S'
3. **Drop** images you want to exclude by clicking "Drop" button or pressing 'D'
4. **Review** your selections - selected images show green "SELECTED" badge, dropped show red "DROPPED" badge
5. **Save** your selections by clicking "Save Selections" button
6. **Export** creates:
   - `selected_images/` folder containing copies of selected images
   - `selected_dataset.jsonl` file with metadata for selected images

## Output Structure

After saving selections, the following structure is created:
```
output_folder/
├── selected_images/          # Copies of selected images
│   ├── image1.png
│   ├── image2.jpg
│   └── ...
└── selected_dataset.jsonl    # New JSONL with selected metadata
```

## Troubleshooting

### No Images Found
- Ensure image folder path is correct
- Check that image filenames match test case IDs in the JSONL file
- Verify image files have supported extensions

### No Metadata Found
- Ensure JSONL file path is correct
- Check JSONL file format (one JSON object per line)
- Verify `test_case_id` field exists in each JSON object

### Port Already in Use
- Use `--port` option to specify a different port
- Check if another service is using the default port 5001

### Selection Not Working
- Ensure you're viewing the correct image before selecting/dropping
- Check browser console for any JavaScript errors
- Try refreshing the page

## Browser Compatibility

- Chrome/Chromium (recommended)
- Firefox
- Safari
- Edge

## Difference from Image Preview

This tool extends the image preview functionality with:
- **Selection capability**: Choose which images to keep or drop
- **Visual feedback**: Clear indicators for selected/dropped images
- **Dataset curation**: Export functionality to create new datasets
- **Keyboard shortcuts**: Efficient workflow with S/D keys
- **Statistics**: Track selection progress

## License

This script is provided as-is for educational and development purposes.
