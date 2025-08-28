#!/bin/bash

# Image Player Server Launcher
# This script helps launch the image player server for OD testing

echo "========================================"
echo "    Image Player Server Launcher"
echo "========================================"

# Default values
DEFAULT_IMAGES="../image_selector/categorized_images"
DEFAULT_PORT=5002
DEFAULT_DURATION=60
DEFAULT_MOTION_INTENSITY=100

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
read -p "Enter path to categorized images folder [$DEFAULT_IMAGES]: " images_path
images_path=${images_path:-$DEFAULT_IMAGES}

read -p "Enter port number [$DEFAULT_PORT]: " port
port=${port:-$DEFAULT_PORT}

read -p "Enter playback duration per image in seconds [$DEFAULT_DURATION]: " duration
duration=${duration:-$DEFAULT_DURATION}

read -p "Enter motion intensity (0-100) [$DEFAULT_MOTION_INTENSITY]: " motion_intensity
motion_intensity=${motion_intensity:-$DEFAULT_MOTION_INTENSITY}

# Check if paths exist
echo ""
echo "Checking paths..."
check_path "$images_path" "Categorized images folder"

# Check if images exist in the categorized folder
if [ -d "$images_path" ]; then
    echo "Scanning for images in categories..."

    # Check each category
    for category in animal person vehicle package; do
        category_path="$images_path/$category"
        if [ -d "$category_path" ]; then
            images_dir="$category_path/images"
            jsonl_file="$category_path/${category}_dataset.jsonl"

            if [ -d "$images_dir" ] && [ -f "$jsonl_file" ]; then
                image_count=$(find "$images_dir" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.gif" -o -name "*.bmp" -o -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.webm" -o -name "*.mkv" \) | wc -l)
                echo "  ✅ $category: $image_count images"
            else
                echo "  ⚠️  $category: Missing images directory or JSONL file"
            fi
        fi
    done
fi

echo ""
echo "Starting Image Player Server with:"
echo "  Images: $images_path"
echo "  Port:   $port"
echo "  Duration: $duration seconds per image"
echo "  Motion Intensity: $motion_intensity%"
echo ""
echo "Server will be available at: http://localhost:$port"
echo ""
echo "Available interfaces:"
echo "  - Regular interface: http://localhost:$port"
echo "  - Big screen interface: http://localhost:$port/bigscreen"
echo ""
echo "Features:"
echo "  - Each image plays for $duration seconds with configurable motion effects"
echo "  - Motion intensity: $motion_intensity% (0=no motion, 100=maximum intensity)"
echo "  - Enhanced media support: PNG, JPG, GIF, MP4, AVI, MOV, WebM, MKV"
echo "  - Recursive file search in subfolders"
echo "  - Automatic playback with detailed timing records"
echo "  - Server-side auto-save with timestamped filename"
echo "  - Export playback history to JSONL format"
echo "  - Mouse cursor visible in big screen mode"
echo ""
echo "Controls (regular interface):"
echo "  - 'Start Playback' button to begin automatic playback"
echo "  - 'Stop Playback' button to stop"
echo "  - 'Save History' button to export timing data"
echo ""
echo "Big screen features:"
echo "  - Auto-starts playback when page loads"
echo "  - No visible controls (perfect for big screens)"
echo "  - No text overlays - clean image-only display"
echo "  - Dynamic motion effects based on intensity setting ($motion_intensity%) - amplitude control"
echo "  - Motion timing: Static 20% → Motion 40% → Static 40% of each image duration"
echo "  - Motion speed: 8 smooth steps with 400ms intervals for natural movement"
echo "  - Full-screen media display with mouse cursor visible"
echo "  - Auto-completion with download prompt"
echo "  - Supports all media formats (images, GIFs, videos)"
echo ""

# Launch the server
python image_player_server.py \
    --images "$images_path" \
    --port "$port" \
    --host 0.0.0.0 \
    --duration "$duration" \
    --motion-intensity "$motion_intensity"
