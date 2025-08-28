# Image Player Server

A Flask-based web application that automatically plays images with motion effects for object detection testing. Each image plays for 60 seconds with smooth animations, and the system records detailed playback timing for later analysis.

## Features

- **Automatic Playback**: Images play automatically for 60 seconds each (configurable)
- **Motion Effects Control**: Configurable motion intensity (0-100%) with `--motion-intensity` parameter
  - Controls movement amplitude (not percentage of images with motion)
  - Pure movement effects (translate, scale, rotate) - no brightness/contrast changes
  - Timing: Static 20% → Motion 40% → Static 40%
  - Motion Speed: 8 steps with 400ms intervals for smooth, natural movement
- **Enhanced Media Support**: GIF, Video (.mp4, .avi, .mov, .webm, .mkv) files supported
- **Recursive File Search**: Finds media files in subfolders within category folders
- **Real-time Updates**: Web interface with live status updates via WebSocket
- **Playback History**: Detailed timing records for each image played with start/end times
- **Auto-Stop**: Playback stops automatically after completing all images
- **Completion Notification**: Shows completion message with download option
- **Server-side Auto-save**: Automatic timestamped JSONL save on server completion
- **Expected Object Type**: Uses folder name as expected object type in records
- **JSONL Export**: Export playback history for Elasticsearch analysis
- **Download Endpoint**: Direct download of playback history
- **Mouse Visibility**: Mouse cursor visible in big screen mode
- **Responsive Design**: Modern web interface with countdown timer

## Requirements

- Python 3.7+
- Flask
- Flask-SocketIO
- Images in PNG, JPG, JPEG, GIF, BMP formats
- Videos in MP4, AVI, MOV formats (optional)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python image_player_server.py --images ../image_selector/categorized_images
```

### Command Line Options

- `-i, --images`: Path to categorized images folder (required)
- `-p, --port`: Port to run the server on (default: 5002)
- `--host`: Host to bind the server to (default: 127.0.0.1)
- `-d, --duration`: Playback duration per image in seconds (default: 60)
- `--motion-intensity`: Motion effect intensity - controls movement amplitude (0-100, default: 100)

### Examples

```bash
# Run with categorized images (default settings)
python image_player_server.py -i ../image_selector/categorized_images

# High motion intensity for strong camera detection
python image_player_server.py -i ../image_selector/categorized_images --motion-intensity 100

# Low motion intensity for subtle effects
python image_player_server.py -i ../image_selector/categorized_images --motion-intensity 25 --duration 30

# No motion effects (static images only)
python image_player_server.py -i ../image_selector/categorized_images --motion-intensity 0

# Run on different port with custom settings
python image_player_server.py -i ./categorized_images --port 8080 --duration 45 --motion-intensity 60

# Run on all interfaces
python image_player_server.py -i ./categorized_images --host 0.0.0.0
```

## Interfaces

The server provides two different interfaces:

### Regular Interface (`/`)
- Full control panel with start/stop buttons
- Real-time status display
- Manual playback control
- Access: `http://localhost:5002`

### Big Screen Interface (`/bigscreen`)
- **Auto-starts playback** when page loads
- **No visible controls** - perfect for big screens
- **No text overlays** - clean image-only display
- **Enhanced motion effects** with dramatic movement, scaling, and lighting
- **Full-screen display** optimized for maximum screen real estate
- **Cursor hidden** for distraction-free viewing
- Access: `http://localhost:5002/bigscreen`

## Data Structure

The system expects images organized in the following structure:

```
categorized_images/
├── animal/
│   ├── images/
│   │   ├── OD-FN-07749.png
│   │   └── ...
│   └── animal_dataset.jsonl
├── person/
│   ├── images/
│   │   ├── OD-POS-01498.png
│   │   └── ...
│   └── person_dataset.jsonl
├── vehicle/
│   ├── images/
│   │   ├── OD-FN-06823.png
│   │   └── ...
│   └── vehicle_dataset.jsonl
└── package/
    ├── images/
    │   ├── OD-NEG-04629.png
    │   └── ...
    └── package_dataset.jsonl
```

## How It Works

1. **Data Loading**: Scans all category folders and loads image metadata
2. **Playlist Creation**: Creates a sequential playlist of all images
3. **Automatic Playback**: Each image plays for 60 seconds with motion effects
4. **Timing Recording**: Records start/end times for each image played
5. **History Export**: Saves playback history to JSONL format

## Motion Effects

### Regular Interface
- Subtle translation and scaling animation
- Smooth 60-second motion cycle
- Maintains image clarity

### Big Screen Interface
- **Enhanced Motion**: Dramatic movement with larger translation distances
- **Dynamic Scaling**: Images scale from 0.92x to 1.1x during animation
- **Rotation Effects**: Subtle rotation (±1.5°) for added dynamism
- **Lighting Effects**: Brightness and contrast adjustments during motion
- **Glow Effects**: Subtle glow animation for enhanced visual impact
- **10 Animation Stages**: 10 distinct motion phases over 60 seconds

## Web Interface

The web interface provides:

- **Live Status**: Current image, progress, and playback state
- **Controls**: Start/Stop playback and save history buttons
- **Countdown Timer**: Shows remaining time for current image
- **Image Info**: Displays test case ID, OD type, and timing
- **Motion Effects**: Smooth animations for static images

## Playback Behavior

### Static Images (PNG, JPG, JPEG, BMP)
- Play for 60 seconds with smooth motion effects
- Animation includes translation and scaling
- Simulates camera movement and object motion

### GIF Images
- Play for 60 seconds without additional motion effects
- Native GIF animation is preserved

### Video Files (MP4, AVI, MOV)
- Play for 60 seconds with native video controls
- No additional motion effects applied
- Supports autoplay and looping

## Output Format

The system generates a JSONL file with playback history:

```json
{"test_case_id": "OD-FN-07749", "expected_object_type": "animal", "od_type_primary": "animal", "category": "animal", "start_time": "2024-01-15T10:30:00.123456", "end_time": "2024-01-15T10:31:00.123456", "duration_seconds": 60.0}
{"test_case_id": "OD-POS-01498", "expected_object_type": "person", "od_type_primary": "person", "category": "person", "start_time": "2024-01-15T10:31:00.123456", "end_time": "2024-01-15T10:32:00.123456", "duration_seconds": 60.0}
```

### Fields Description

- `test_case_id`: Unique identifier for the image/test case
- `expected_object_type`: Expected object type (from folder name: animal, person, vehicle, package)
- `od_type_primary`: Original object detection type from metadata
- `category`: Image category folder name
- `start_time`: ISO format timestamp when playback started
- `end_time`: ISO format timestamp when playback ended
- `duration_seconds`: Actual playback duration in seconds

## API Endpoints

- `GET /`: Main web interface
- `GET /bigscreen`: Big screen interface (auto-start, no controls)
- `GET /api/start`: Start automatic playback
- `GET /api/stop`: Stop automatic playback
- `GET /api/status`: Get current playback status
- `GET /api/debug`: Debug endpoint for server status
- `GET /api/download`: Download playback history as JSONL file
- `GET /api/save_history?output=filename.jsonl`: Save playback history

### Server-side Auto-save

When playback completes, the system automatically saves a timestamped JSONL file on the server with the format:
`playback_history_YYYYMMDD_HHMMSS.jsonl`

This file is saved in the server's working directory and can be retrieved later for analysis.

### Testing New Features

To test the enhanced functionality:

```bash
# Test different motion intensities
python image_player_server.py -i ../image_selector/categorized_images --motion-intensity 25
python image_player_server.py -i ../image_selector/categorized_images --motion-intensity 75

# Test with shorter duration for faster testing
python image_player_server.py -i ../image_selector/categorized_images --duration 10 --motion-intensity 50

# Test media format support
# Place some .gif, .mp4, .avi files in category subfolders and run:
python image_player_server.py -i ../image_selector/categorized_images

# Quick demo with all new features
python image_player_server.py \
  -i ../image_selector/categorized_images \
  --duration 15 \
  --motion-intensity 80 \
  --host 0.0.0.0 \
  --port 5002

# Debug motion effects (open browser console to see motion logs)
python image_player_server.py \
  -i ../image_selector/categorized_images \
  --duration 10 \
  --motion-intensity 100 \
  --host 0.0.0.0
```

## Troubleshooting Motion Effects

### Why some images don't move:

1. **Check Browser Console**: Open browser dev tools (F12) and look for `[MOTION]` logs
2. **Motion Intensity = 0**: If set to 0, no motion effects will be applied
3. **Video Files**: Videos don't get motion effects (only static images and GIFs)
4. **Loading Errors**: Check for image loading errors in console

### Debug Command:
```bash
python image_player_server.py -i ../image_selector/categorized_images --duration 10 --motion-intensity 100
```
Then open `http://localhost:5002/bigscreen` and check browser console.

## WebSocket Events

- `image_update`: Fired when a new image starts playing
  - Contains image data, start time, progress info, duration, and motion intensity
- `playback_complete`: Fired when all images have been played
  - Contains total images count, total duration, and server filename

## Integration with Elasticsearch

The JSONL output format is designed for easy import into Elasticsearch:

```bash
# Example: Import playback history to Elasticsearch
curl -X POST "localhost:9200/playback_history/_bulk" \
  -H 'Content-Type: application/json' \
  --data-binary @playback_history.jsonl
```

## Keyboard Shortcuts

- **Space**: Start/Stop playback (when implemented)
- **S**: Save history (when implemented)

## Troubleshooting

### Common Issues

1. **"Images folder does not exist"**
   - Verify the path to your categorized images folder
   - Use absolute paths if relative paths don't work

2. **"No images found"**
   - Check that your images are in the correct folder structure
   - Ensure JSONL files exist and contain valid data

3. **WebSocket connection issues**
   - Check browser console for connection errors
   - Ensure the server is running and accessible

4. **Motion effects not working**
   - Check browser compatibility (Chrome/Firefox recommended)
   - Ensure CSS animations are not disabled

## Performance Considerations

- **Image Loading**: Large images may take time to load
- **Memory Usage**: Video files consume more memory
- **Browser Compatibility**: Modern browsers work best
- **Network**: Fast network recommended for smooth playback

## Customization

### Changing Playback Duration

Modify the `time.sleep(60)` value in the playback loop to change duration.

### Custom Motion Effects

Edit the `@keyframes moveImage` CSS animation to customize motion effects.

### Category Order

Modify the `category_order` list to change the playback sequence.

## Technical Details

- **Framework**: Flask with Flask-SocketIO
- **Frontend**: HTML5, CSS3, JavaScript with Socket.IO
- **Animation**: CSS3 keyframes for smooth motion effects
- **Timing**: Python datetime for precise timing records
- **File Serving**: Direct file serving for images and videos

## License

This script is provided for educational and development purposes.
