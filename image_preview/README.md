# Image Preview Server

A Flask-based web server that displays images from an object detection dataset with their corresponding metadata, prompts, and risk tags in a user-friendly interface.

## Features

- **Split-panel UI**: Left side shows images, right side shows metadata
- **Navigation**: Previous/Next buttons to browse through images
- **Keyboard shortcuts**: Use arrow keys to navigate
- **Rich metadata display**: Shows all JSONL data in human-readable format
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
python host_image_preview_server.py --images downloaded_images --jsonl dataset/od_synth_cases_10000_cctv_v1_.jsonl
```

### Command Line Options

- `-i, --images`: Path to folder containing images (required)
- `-j, --jsonl`: Path to JSONL file containing image metadata (required)
- `-p, --port`: Port to run the server on (default: 5000)
- `--host`: Host to bind the server to (default: 127.0.0.1)
- `--debug`: Run in debug mode

### Examples

```bash
# Run on default port 5000
python host_image_preview_server.py -i downloaded_images -j dataset/od_synth_cases_10000_cctv_v1_.jsonl

# Run on custom port
python host_image_preview_server.py -i images_folder -j metadata.jsonl --port 8080

# Run on all interfaces
python host_image_preview_server.py -i images_folder -j metadata.jsonl --host 0.0.0.0

# Run in debug mode
python host_image_preview_server.py -i images_folder -j metadata.jsonl --debug
```

## How It Works

1. **Data Loading**: The server loads metadata from the JSONL file and scans the image folder
2. **Matching**: Images are matched with metadata based on filename (without extension)
3. **Web Interface**: A Flask server provides a web UI with API endpoints
4. **Navigation**: Users can browse through images using buttons or keyboard shortcuts

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
- `GET /images/<filename>`: Serve image files

## Keyboard Shortcuts

- **Left Arrow**: Previous image
- **Right Arrow**: Next image

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
- Check if another service is using the default port 5000

## Browser Compatibility

- Chrome/Chromium (recommended)
- Firefox
- Safari
- Edge

## License

This script is provided as-is for educational and development purposes.
