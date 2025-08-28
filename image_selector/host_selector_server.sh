#!/bin/bash

# Image Selector Server Launcher
# This script helps launch the image selector server with common configurations

echo "========================================"
echo "    Image Selector Server Launcher"
echo "========================================"

# Default values
DEFAULT_IMAGES="../image_preview/downloaded_images"
DEFAULT_JSONL="../dataset/od_synth_cases_10000_cctv_v1_.jsonl"
DEFAULT_OUTPUT="selected_images"
DEFAULT_PORT=5001

# Function to check if path exists
check_path() {
    local path="$1"
    local description="$2"
    if [ ! -e "$path" ]; then
        echo "Warning: $description '$path' does not exist"
        return 1
    fi
    return 0
}

# Get user input for paths
read -p "Enter path to images folder [$DEFAULT_IMAGES]: " images_path
images_path=${images_path:-$DEFAULT_IMAGES}

read -p "Enter path to JSONL file [$DEFAULT_JSONL]: " jsonl_path
jsonl_path=${jsonl_path:-$DEFAULT_JSONL}

read -p "Enter output folder name [$DEFAULT_OUTPUT]: " output_path
output_path=${output_path:-$DEFAULT_OUTPUT}

read -p "Enter port number [$DEFAULT_PORT]: " port
port=${port:-$DEFAULT_PORT}

# Check if paths exist
echo ""
echo "Checking paths..."
check_path "$images_path" "Images folder"
check_path "$jsonl_path" "JSONL file"

# Create output directory if it doesn't exist
if [ ! -d "$output_path" ]; then
    echo "Creating output directory: $output_path"
    mkdir -p "$output_path"
fi

echo ""
echo "Starting Image Selector Server with:"
echo "  Images: $images_path"
echo "  JSONL:  $jsonl_path"
echo "  Output: $output_path"
echo "  Port:   $port"
echo ""
echo "Server will be available at: http://localhost:$port"
echo ""
echo "Controls:"
echo "  - Arrow keys to navigate"
echo "  - S key to select images"
echo "  - D key to drop images"
echo "  - Use the Save Selections button to export"
echo ""

# Launch the server
python host_image_selector_server.py \
    --images "$images_path" \
    --jsonl "$jsonl_path" \
    --output "$output_path" \
    --port "$port" \
    --host 0.0.0.0
