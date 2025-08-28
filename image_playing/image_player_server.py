#!/usr/bin/env python3
"""
Image Player Server
Displays images with motion effects and records playback timing for OD testing
"""

import os
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

class ImagePlayer:
    def __init__(self, categorized_images_path, playback_duration=60, motion_intensity=100):
        # Convert relative path to absolute path
        self.categorized_images_path = Path(categorized_images_path).resolve()
        self.playback_duration = playback_duration  # Duration in seconds
        self.motion_intensity = motion_intensity  # Motion intensity 0-100
        self.playback_history = []
        self.current_playlist = []
        self.is_playing = False
        self.current_image_index = 0
        self.playback_start_time = None

        # Load all images from categorized folders
        self.load_playlist()

    def load_playlist(self):
        """Load all images from categorized folders"""
        self.current_playlist = []

        if not self.categorized_images_path.exists():
            print(f"Warning: Categorized images path {self.categorized_images_path} does not exist")
            return

        # Define category order (can be customized)
        category_order = ['animal', 'person', 'vehicle', 'package']

        for category in category_order:
            category_path = self.categorized_images_path / category
            if not category_path.exists():
                continue

            images_path = category_path / "images"
            jsonl_file = category_path / f"{category}_dataset.jsonl"

            if not images_path.exists() or not jsonl_file.exists():
                continue

            # Load JSONL data
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                test_case_id = data.get('test_case_id')
                                od_type_primary = data.get('od_type_primary', category)

                                if test_case_id:
                                    # Find corresponding image file (search recursively in subfolders)
                                    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.mp4', '.avi', '.mov', '.webm', '.mkv']
                                    image_file = None

                                    # First search in the images subfolder
                                    images_subfolder = images_path / "images"
                                    if images_subfolder.exists():
                                        for ext in image_extensions:
                                            potential_file = images_subfolder / f"{test_case_id}{ext}"
                                            if potential_file.exists():
                                                image_file = potential_file
                                                break

                                    # If not found in images subfolder, search recursively in the category folder
                                    if not image_file:
                                        for root, dirs, files in os.walk(images_path):
                                            for file in files:
                                                if file.startswith(test_case_id) and any(file.endswith(ext) for ext in image_extensions):
                                                    image_file = Path(root) / file
                                                    break
                                            if image_file:
                                                break

                                    if image_file:
                                        # Create relative path for Flask serving
                                        # Calculate relative path from the category folder
                                        try:
                                            relative_path = image_file.relative_to(self.categorized_images_path / category)
                                            relative_path_str = f"{category}/{relative_path}"
                                        except ValueError:
                                            # Fallback for files outside the expected structure
                                            relative_path_str = f"{category}/images/{image_file.name}"

                                        image_info = {
                                            'test_case_id': test_case_id,
                                            'od_type_primary': od_type_primary,
                                            'image_path': relative_path_str,
                                            'category': category,
                                            'is_video': image_file.suffix.lower() in ['.mp4', '.avi', '.mov', '.webm', '.mkv'],
                                            'is_gif': image_file.suffix.lower() == '.gif'
                                        }
                                        self.current_playlist.append(image_info)
                                        print(f"[DEBUG] Added to playlist: {test_case_id} -> {relative_path}")
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                print(f"Error loading {jsonl_file}: {e}")
                continue

        print(f"Loaded {len(self.current_playlist)} images for playback")

    def get_current_image(self):
        """Get current image data"""
        if not self.current_playlist:
            return None

        if self.current_image_index >= len(self.current_playlist):
            self.current_image_index = 0

        return self.current_playlist[self.current_image_index]

    def next_image(self):
        """Move to next image"""
        if self.current_playlist:
            self.current_image_index = (self.current_image_index + 1) % len(self.current_playlist)
            return self.get_current_image()
        return None

    def start_playback(self):
        """Start automatic playback"""
        if self.is_playing:
            return

        self.is_playing = True
        self.playback_start_time = datetime.now()

        print(f"[DEBUG] Starting playback with {len(self.current_playlist)} images")
        print(f"[DEBUG] First image: {self.current_playlist[0]['test_case_id'] if self.current_playlist else 'None'}")
        print(f"[DEBUG] First image path: {self.current_playlist[0]['image_path'] if self.current_playlist else 'None'}")

        def playback_loop():
            while self.is_playing:
                current_image = self.get_current_image()
                if current_image:
                    print(f"[DEBUG] Playing image {self.current_image_index + 1}/{len(self.current_playlist)}: {current_image['test_case_id']}")
                    print(f"[DEBUG] Image path: {current_image['image_path']}")

                    # Record start time
                    start_time = datetime.now()

                    # Emit to frontend
                    print(f"[DEBUG] Emitting image_update to frontend")
                    socketio.emit('image_update', {
                        'image_data': current_image,
                        'start_time': start_time.isoformat(),
                        'index': self.current_image_index,
                        'total': len(self.current_playlist),
                        'duration_seconds': self.playback_duration,
                        'motion_intensity': self.motion_intensity
                    })

                    # Wait for configured duration
                    time.sleep(self.playback_duration)

                    # Record end time
                    end_time = datetime.now()

                    # Add to playback history with folder name as expected object type
                    self.playback_history.append({
                        'test_case_id': current_image['test_case_id'],
                        'expected_object_type': current_image['category'],  # Folder name as expected object type
                        'od_type_primary': current_image['od_type_primary'],
                        'category': current_image['category'],
                        'start_time': start_time.isoformat(),
                        'end_time': end_time.isoformat(),
                        'duration_seconds': (end_time - start_time).total_seconds()
                    })

                    # Move to next image
                    self.next_image()

                    # Check if we've completed all images (back to start)
                    if self.current_image_index == 0:
                        print("[DEBUG] Completed all images, stopping playback")
                        self.is_playing = False
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"playback_history_{timestamp}.jsonl"

                        # Save to server with timestamp
                        self.save_playback_history(filename)
                        print(f"[DEBUG] Playback history saved to server as: {filename}")

                        # Notify frontend that playback is complete
                        socketio.emit('playback_complete', {
                            'total_images': len(self.current_playlist),
                            'total_duration': sum(h['duration_seconds'] for h in self.playback_history),
                            'server_filename': filename
                        })
                        break
                else:
                    time.sleep(1)

        thread = threading.Thread(target=playback_loop, daemon=True)
        thread.start()

    def stop_playback(self):
        """Stop automatic playback"""
        self.is_playing = False

    def save_playback_history(self, output_file):
        """Save playback history to JSONL file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for record in self.playback_history:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        print(f"Saved {len(self.playback_history)} playback records to {output_path}")

# Global player instance
player = None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/bigscreen')
def big_screen():
    """Big screen interface - auto-start, no controls"""
    return render_template('big_screen.html')

@app.route('/api/debug')
def debug_status():
    """Debug endpoint to check server status"""
    if not player:
        return jsonify({
            'status': 'error',
            'message': 'Player not initialized'
        })

    return jsonify({
        'status': 'ok',
        'is_playing': player.is_playing,
        'playlist_length': len(player.current_playlist),
        'current_index': player.current_image_index,
        'playback_duration': player.playback_duration,
        'images_directory': str(player.categorized_images_path),
        'directory_exists': player.categorized_images_path.exists()
    })

@app.route('/api/download')
def download_history():
    """Download endpoint for playback history"""
    if not player or not player.playback_history:
        return jsonify({'error': 'No playback history available'}), 404

    # Create response with JSONL file
    from io import StringIO
    import csv

    output = StringIO()
    for record in player.playback_history:
        output.write(json.dumps(record, ensure_ascii=False) + '\n')

    # Create response
    response = app.response_class(
        response=output.getvalue(),
        status=200,
        mimetype='application/json',
        headers={'Content-Disposition': 'attachment; filename=playback_history.jsonl'}
    )

    return response

@app.route('/api/start')
def start_playback():
    """Start playback"""
    if player:
        player.start_playback()
        return jsonify({'success': True, 'message': 'Playback started'})
    return jsonify({'success': False, 'message': 'Player not initialized'}), 500

@app.route('/api/stop')
def stop_playback():
    """Stop playback"""
    if player:
        player.stop_playback()
        return jsonify({'success': True, 'message': 'Playback stopped'})
    return jsonify({'success': False, 'message': 'Player not initialized'}), 500

@app.route('/api/status')
def get_status():
    """Get current playback status"""
    if not player:
        return jsonify({'success': False, 'message': 'Player not initialized'}), 500

    current_image = player.get_current_image()
    return jsonify({
        'is_playing': player.is_playing,
        'current_image': current_image,
        'playlist_length': len(player.current_playlist),
        'current_index': player.current_image_index,
        'playback_history_count': len(player.playback_history)
    })

@app.route('/api/save_history')
def save_history():
    """Save playback history"""
    if not player:
        return jsonify({'success': False, 'message': 'Player not initialized'}), 500

    output_file = request.args.get('output', 'playback_history.jsonl')
    try:
        player.save_playback_history(output_file)
        return jsonify({'success': True, 'message': f'History saved to {output_file}'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/images/<path:filepath>')
def serve_image(filepath):
    """Serve images"""
    print(f"[DEBUG] Image request: {filepath}")

    # Extract the directory path from the filepath
    # filepath format: "path/to/categorized_images/category/images/filename.png"
    parts = filepath.split('/')
    print(f"[DEBUG] Path parts: {parts}")

    if len(parts) >= 3:
        # Find the category folder and serve from there
        category = parts[-3]  # e.g., "animal", "person", etc.
        filename = parts[-1]  # e.g., "OD-FN-07749.png"

        print(f"[DEBUG] Extracted category: {category}, filename: {filename}")

        if player and player.categorized_images_path:
            category_path = player.categorized_images_path / category / "images"
            full_image_path = category_path / filename

            print(f"[DEBUG] Looking for image at: {full_image_path}")
            print(f"[DEBUG] Category path exists: {category_path.exists()}")
            print(f"[DEBUG] Image file exists: {full_image_path.exists()}")

            if category_path.exists() and full_image_path.exists():
                print(f"[DEBUG] Serving image: {full_image_path}")
                return send_from_directory(category_path, filename)
            else:
                print(f"[DEBUG] Image not found: {full_image_path}")
        else:
            print("[DEBUG] Player or categorized_images_path not available")
    else:
        print("[DEBUG] Invalid path structure")

    return "Image not found", 404

def create_templates():
    """Create HTML templates"""
    template_dir = Path(__file__).parent / 'templates'
    template_dir.mkdir(exist_ok=True)

    template_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Player - OD Testing</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a1a;
            color: white;
            overflow: hidden;
        }

        .header {
            background: #2c3e50;
            padding: 1rem;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }

        .header h1 {
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
        }

        .controls {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 1rem;
            background: rgba(0,0,0,0.8);
            padding: 1rem;
            border-radius: 10px;
            z-index: 1000;
        }

        .btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1rem;
            transition: background 0.2s;
        }

        .btn:hover {
            background: #2980b9;
        }

        .btn.stop {
            background: #e74c3c;
        }

        .btn.stop:hover {
            background: #c0392b;
        }

        .status {
            position: fixed;
            top: 20px;
            left: 20px;
            background: rgba(0,0,0,0.8);
            padding: 1rem;
            border-radius: 10px;
            font-size: 0.9rem;
        }

        .image-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #000;
            overflow: hidden;
        }

        .image-container img, .image-container video {
            max-width: 90%;
            max-height: 90%;
            object-fit: contain;
            border-radius: 8px;
            box-shadow: 0 0 50px rgba(0,0,0,0.5);
            animation: none;
        }

        .image-container.moving img {
            animation: moveImage 60s linear infinite;
        }

        .image-info {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0,0,0,0.8);
            padding: 1rem;
            border-radius: 10px;
            font-size: 0.9rem;
            max-width: 300px;
        }

        .timer {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 4rem;
            color: rgba(255,255,255,0.3);
            font-weight: bold;
            pointer-events: none;
            z-index: 10;
        }

        @keyframes moveImage {
            0% {
                transform: translate(0, 0) scale(1);
            }
            25% {
                transform: translate(20px, -20px) scale(1.05);
            }
            50% {
                transform: translate(-20px, 20px) scale(0.95);
            }
            75% {
                transform: translate(15px, 15px) scale(1.02);
            }
            100% {
                transform: translate(0, 0) scale(1);
            }
        }

        .hidden {
            display: none !important;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Object Detection Image Player</h1>
        <div class="subtitle">Automatic playback with motion effects for OD testing</div>
    </div>

    <div class="status">
        <div>Status: <span id="statusText">Stopped</span></div>
        <div>Current: <span id="currentImage">-</span></div>
        <div>Progress: <span id="progress">0 / 0</span></div>
        <div>History: <span id="historyCount">0</span> records</div>
    </div>

    <div class="image-info">
        <div><strong>Test Case ID:</strong> <span id="testCaseId">-</span></div>
        <div><strong>OD Type:</strong> <span id="odType">-</span></div>
        <div><strong>Category:</strong> <span id="category">-</span></div>
        <div><strong>Start Time:</strong> <span id="startTime">-</span></div>
    </div>

    <div class="timer" id="timer">60</div>

    <div class="image-container" id="imageContainer">
        <div style="color: rgba(255,255,255,0.5); font-size: 1.5rem;">Waiting for playback to start...</div>
    </div>

    <div class="controls">
        <button class="btn" id="startBtn" onclick="startPlayback()">Start Playback</button>
        <button class="btn stop" id="stopBtn" onclick="stopPlayback()">Stop Playback</button>
        <button class="btn" id="saveBtn" onclick="saveHistory()">Save History</button>
    </div>

    <script>
        const socket = io();
        let countdownInterval = null;
        let timeLeft = 60;

        socket.on('image_update', function(data) {
            displayImage(data.image_data, data.start_time);
            updateInfo(data);
            startCountdown();
        });

        function startPlayback() {
            fetch('/api/start')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('statusText').textContent = 'Playing';
                        document.getElementById('startBtn').disabled = true;
                        document.getElementById('stopBtn').disabled = false;
                    } else {
                        alert(data.message);
                    }
                });
        }

        function stopPlayback() {
            fetch('/api/stop')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('statusText').textContent = 'Stopped';
                        document.getElementById('startBtn').disabled = false;
                        document.getElementById('stopBtn').disabled = true;
                        stopCountdown();
                    } else {
                        alert(data.message);
                    }
                });
        }

        function saveHistory() {
            const outputFile = prompt('Enter output filename:', 'playback_history.jsonl');
            if (outputFile) {
                fetch(`/api/save_history?output=${outputFile}`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert(data.message);
                        } else {
                            alert('Error: ' + data.message);
                        }
                    });
            }
        }

        function displayImage(imageData, startTime) {
            const container = document.getElementById('imageContainer');

            // Clear previous content
            container.innerHTML = '';

            const imagePath = imageData.image_path;
            const isVideo = imageData.is_video;
            const isGif = imageData.is_gif;

            if (isVideo) {
                const video = document.createElement('video');
                video.src = `/images/${imagePath}`;
                video.controls = false;
                video.autoplay = true;
                video.loop = true;
                video.style.maxWidth = '90%';
                video.style.maxHeight = '90%';
                video.style.objectFit = 'contain';
                video.style.borderRadius = '8px';
                video.style.boxShadow = '0 0 50px rgba(0,0,0,0.5)';
                container.appendChild(video);
            } else {
                const img = document.createElement('img');
                img.src = `/images/${imagePath}`;
                img.alt = imageData.test_case_id;

                // Add motion effect for static images (not GIF)
                if (!isGif) {
                    container.classList.add('moving');
                    setTimeout(() => {
                        container.classList.remove('moving');
                    }, 60000); // Remove after 60 seconds
                }

                container.appendChild(img);
            }
        }

        function updateInfo(data) {
            document.getElementById('testCaseId').textContent = data.image_data.test_case_id;
            document.getElementById('odType').textContent = data.image_data.od_type_primary;
            document.getElementById('category').textContent = data.image_data.category;
            document.getElementById('startTime').textContent = new Date(data.start_time).toLocaleTimeString();
            document.getElementById('progress').textContent = `${data.index + 1} / ${data.total}`;
        }

        function startCountdown(duration = 60) {
            stopCountdown();
            timeLeft = duration;

            countdownInterval = setInterval(() => {
                timeLeft--;
                const timerElement = document.getElementById('timer');
                if (timerElement) {
                    timerElement.textContent = timeLeft;
                }

                if (timeLeft <= 0) {
                    stopCountdown();
                }
            }, 1000);
        }

        function stopCountdown() {
            if (countdownInterval) {
                clearInterval(countdownInterval);
                countdownInterval = null;
            }
            document.getElementById('timer').textContent = '60';
        }

        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('historyCount').textContent = data.playback_history_count;
                        document.getElementById('currentImage').textContent =
                            data.current_image ? data.current_image.test_case_id : '-';
                    }
                });
        }

        // Update status every 5 seconds
        setInterval(updateStatus, 5000);

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            updateStatus();
        });
    </script>
</body>
</html>'''

    # Create regular interface
    with open(template_dir / 'index.html', 'w', encoding='utf-8') as f:
        f.write(template_content)

    # Create big screen interface (auto-start, no controls, no text)
    big_screen_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Player - Big Screen</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            background: #000;
            overflow: hidden;
            height: 100vh;
            width: 100vw;
            cursor: default;
        }

        .image-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #000;
            overflow: hidden;
        }

        .image-container img, .image-container video {
            width: 100%;
            height: 100%;
            object-fit: cover;
            animation: none;
        }

        .image-container.moving img {
            animation: enhancedMotion 60s ease-in-out infinite;
        }

        .loading {
            color: rgba(255,255,255,0.3);
            font-size: 2rem;
            font-family: Arial, sans-serif;
            cursor: pointer;
            padding: 20px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 10px;
            transition: all 0.3s ease;
        }

        .loading:hover {
            color: rgba(255,255,255,0.7);
            border-color: rgba(255,255,255,0.7);
            background: rgba(255,255,255,0.1);
        }

        /* Enhanced motion effects with dramatic movement */
        @keyframes enhancedMotion {
            0% {
                transform: translate(0, 0) scale(1) rotate(0deg);
                filter: brightness(1) contrast(1) saturate(1);
            }
            8% {
                transform: translate(60px, -45px) scale(1.15) rotate(1deg);
                filter: brightness(1.2) contrast(1.1) saturate(1.1);
            }
            16% {
                transform: translate(-70px, 35px) scale(0.88) rotate(-1.5deg);
                filter: brightness(0.85) contrast(0.95) saturate(0.9);
            }
            25% {
                transform: translate(50px, 60px) scale(1.12) rotate(2deg);
                filter: brightness(1.15) contrast(1.08) saturate(1.05);
            }
            33% {
                transform: translate(-45px, -70px) scale(0.92) rotate(-2.5deg);
                filter: brightness(0.9) contrast(0.92) saturate(0.95);
            }
            41% {
                transform: translate(80px, 20px) scale(1.18) rotate(1.8deg);
                filter: brightness(1.25) contrast(1.15) saturate(1.2);
            }
            50% {
                transform: translate(-60px, 80px) scale(0.85) rotate(-3deg);
                filter: brightness(0.8) contrast(0.88) saturate(0.85);
            }
            58% {
                transform: translate(35px, -80px) scale(1.08) rotate(0.5deg);
                filter: brightness(1.1) contrast(1.05) saturate(1.08);
            }
            66% {
                transform: translate(-80px, -30px) scale(0.95) rotate(-1.8deg);
                filter: brightness(0.95) contrast(0.98) saturate(0.92);
            }
            75% {
                transform: translate(70px, 50px) scale(1.14) rotate(2.5deg);
                filter: brightness(1.2) contrast(1.12) saturate(1.15);
            }
            83% {
                transform: translate(-55px, 70px) scale(0.89) rotate(-2.2deg);
                filter: brightness(0.88) contrast(0.9) saturate(0.88);
            }
            91% {
                transform: translate(65px, -55px) scale(1.11) rotate(1.5deg);
                filter: brightness(1.18) contrast(1.1) saturate(1.12);
            }
            100% {
                transform: translate(0, 0) scale(1) rotate(0deg);
                filter: brightness(1) contrast(1) saturate(1);
            }
        }

        /* Additional dramatic effect */
        .image-container.intense img {
            animation: enhancedMotion 60s ease-in-out infinite,
                      subtleGlow 4s ease-in-out infinite alternate;
        }

        @keyframes subtleGlow {
            from {
                box-shadow: 0 0 20px rgba(255,255,255,0.1);
            }
            to {
                box-shadow: 0 0 40px rgba(255,255,255,0.3);
            }
        }

        /* Hide any potential text or UI elements */
        .hidden {
            display: none !important;
        }

        /* Ensure fullscreen display */
        html, body {
            margin: 0;
            padding: 0;
            height: 100%;
            width: 100%;
            overflow: hidden;
        }
    </style>
</head>
<body>
    <div class="image-container" id="imageContainer">
        <div class="loading" onclick="manualStart()" title="Click to start playback manually">Loading... Click to start</div>
    </div>

    <script>
        const socket = io();

        socket.on('connect', function() {
            console.log('WebSocket connected');
        });

        socket.on('disconnect', function() {
            console.log('WebSocket disconnected');
        });

        socket.on('connect_error', function(error) {
            console.error('WebSocket connection error:', error);
        });

        socket.on('image_update', function(data) {
            console.log('Received image_update:', data);
            displayImage(data.image_data, data.duration_seconds || 60, data.motion_intensity || 100);
            startCountdown(data.duration_seconds || 60);
        });

        socket.on('playback_complete', function(data) {
            console.log('Playback completed:', data);
            showCompletionMessage(data);
        });

        function displayImage(imageData, durationSeconds, motionIntensity) {
            const container = document.getElementById('imageContainer');

            // Clear previous content
            container.innerHTML = '';

            const imagePath = imageData.image_path;
            const isVideo = imageData.is_video;
            const isGif = imageData.is_gif;

            console.log('Displaying image:', imageData.test_case_id, imagePath);

            if (isVideo) {
                // For videos, just play them without motion effects
                const video = document.createElement('video');
                // imagePath is already in the correct format for Flask routing
                video.src = `/images/${encodeURIComponent(imagePath)}`;
                video.controls = false;
                video.autoplay = true;
                video.loop = true;
                video.muted = true; // Mute to allow autoplay

                // Add error handling
                video.onerror = function() {
                    console.error('Video failed to load:', imagePath);
                    // Show error message
                    container.innerHTML = '<div style="color: red; font-size: 2rem;">Video Error: ' + imageData.test_case_id + '</div>';
                };

                container.appendChild(video);
            } else {
                // For static images
                const img = document.createElement('img');
                // imagePath is already in the correct format for Flask routing
                img.src = `/images/${encodeURIComponent(imagePath)}`;
                img.alt = imageData.test_case_id;

                // Add error handling
                img.onload = function() {
                    console.log('Image loaded successfully:', imagePath);

                    // Add enhanced motion effect for all static content (not video)
                    if (!isVideo && motionIntensity > 0) {
                        console.log(`[MOTION] Applying motion effect to ${imageData.test_case_id}: intensity=${motionIntensity}, duration=${durationSeconds}s, isVideo=${isVideo}, isGif=${isGif}`);

                        // Add visual indicator that motion is active
                        const motionIndicator = document.createElement('div');
                        motionIndicator.id = 'motion-indicator';
                        motionIndicator.style.cssText = `
                            position: fixed;
                            top: 10px;
                            right: 10px;
                            background: #ff0000;
                            color: white;
                            padding: 5px 10px;
                            border-radius: 5px;
                            font-size: 12px;
                            z-index: 10000;
                        `;
                        motionIndicator.textContent = `MOTION: ${motionIntensity}%`;

                        // Remove existing indicator
                        const existing = document.getElementById('motion-indicator');
                        if (existing) existing.remove();

                        document.body.appendChild(motionIndicator);

                        // Remove any existing motion styles
                        container.classList.remove('moving', 'intense');
                        container.style.animation = '';
                        container.style.transform = '';

                        // Apply simple and direct motion effect
                        applySimpleMotionEffect(container, motionIntensity, durationSeconds);

                        // Reset animation after playback duration
                        setTimeout(() => {
                            console.log(`[MOTION] Resetting motion for ${imageData.test_case_id}`);
                            container.classList.remove('moving', 'intense');
                            container.style.animation = '';
                            container.style.transform = '';

                            // Remove motion indicator
                            const indicator = document.getElementById('motion-indicator');
                            if (indicator) indicator.remove();
                        }, durationSeconds * 1000);
                    } else {
                        console.log(`[MOTION] Skipping motion effect for ${imageData.test_case_id}: intensity=${motionIntensity}, isVideo=${isVideo}, isGif=${isGif}`);
                    }
                };

                img.onerror = function() {
                    console.error('Image failed to load:', imagePath);
                    // Show error message and try to continue to next image
                    container.innerHTML = '<div style="color: red; font-size: 3rem; text-align: center;">Image Error: ' + imageData.test_case_id + '<br><small>Continuing to next image...</small></div>';

                    // Auto-advance to next image after 3 seconds
                    setTimeout(() => {
                        console.log('Auto-advancing due to image error');
                        // Request next image from server
                        fetch('/api/next')
                            .then(response => response.json())
                            .then(data => {
                                if (data.success) {
                                    console.log('Successfully advanced to next image');
                                }
                            })
                            .catch(error => {
                                console.error('Failed to advance:', error);
                            });
                    }, 3000);
                };

                container.appendChild(img);
            }
        }

        // Manual start function
        function manualStart() {
            console.log('Manual start triggered');
            fetch('/api/start')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        console.log('Manually started playback for big screen');
                        document.querySelector('.loading').textContent = 'Starting playback...';
                    } else {
                        console.error('Failed to start:', data.message);
                        alert('Failed to start: ' + data.message);
                    }
                })
                .catch(error => {
                    console.error('Error starting:', error);
                    alert('Error starting playback: ' + error.message);
                });
        }

        function showCompletionMessage(data) {
            const container = document.getElementById('imageContainer');
            container.innerHTML = `
                <div style="
                    color: #00ff00;
                    font-size: 3rem;
                    text-align: center;
                    padding: 2rem;
                    border: 4px solid #00ff00;
                    border-radius: 20px;
                    background: rgba(0,0,0,0.8);
                ">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">🎉</div>
                    <div style="margin-bottom: 1rem;">Playback Complete!</div>
                    <div style="font-size: 1.5rem; margin-bottom: 2rem;">
                        ${data.total_images} images played<br>
                        Total duration: ${Math.round(data.total_duration)}s
                    </div>
                    <button onclick="downloadHistory()" style="
                        background: #00ff00;
                        color: black;
                        border: none;
                        padding: 1rem 2rem;
                        font-size: 1.5rem;
                        border-radius: 10px;
                        cursor: pointer;
                        font-weight: bold;
                    ">📥 Download Results</button>
                </div>
            `;
        }

        function downloadHistory() {
            // Create download link
            const link = document.createElement('a');
            link.href = '/api/download';
            link.download = 'playback_history.jsonl';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }

        function applySimpleMotionEffect(container, intensity, durationSeconds) {
            console.log(`[MOTION] Applying motion effect: intensity=${intensity}%, duration=${durationSeconds}s`);

            const factor = intensity / 100;
            const maxTranslate = 80 * factor; // 80px max movement - more natural
            const stepInterval = (durationSeconds * 0.4 * 1000) / 10; // 10 steps in motion phase

            // Phase 1: Ensure static (0-20%)
            container.style.transform = 'translate(0, 0) scale(1) rotate(0deg)';
            container.style.transition = 'transform 0.5s ease-out';

            // Phase 2: Motion (20-60%)
            setTimeout(() => {
                console.log(`[MOTION] Starting motion phase with ${maxTranslate}px amplitude`);

                // Natural motion sequence
                const motions = [
                    `translate(${maxTranslate * 0.6}px, ${-maxTranslate * 0.4}px) scale(1.15) rotate(3deg)`,
                    `translate(${-maxTranslate * 0.8}px, ${maxTranslate * 0.5}px) scale(0.92) rotate(-6deg)`,
                    `translate(${maxTranslate * 0.5}px, ${maxTranslate * 0.8}px) scale(1.22) rotate(8deg)`,
                    `translate(${-maxTranslate * 0.7}px, ${-maxTranslate * 0.6}px) scale(0.88) rotate(-9deg)`,
                    `translate(${maxTranslate * 0.8}px, ${maxTranslate * 0.3}px) scale(1.12) rotate(5deg)`,
                    `translate(${-maxTranslate * 0.5}px, ${-maxTranslate * 0.8}px) scale(0.95) rotate(-7deg)`,
                    `translate(${maxTranslate * 0.4}px, ${maxTranslate * 0.7}px) scale(1.18) rotate(9deg)`,
                    `translate(0, 0) scale(1) rotate(0deg)` // Back to center
                ];

                motions.forEach((motion, index) => {
                    setTimeout(() => {
                        console.log(`[MOTION] Step ${index + 1}: ${motion}`);
                        container.style.transform = motion;
                    }, index * 400); // 400ms between each step - slower pace
                });

            }, durationSeconds * 0.2 * 1000);

            // Phase 3: Back to static (60%+)
            setTimeout(() => {
                console.log(`[MOTION] Back to static`);
                container.style.transition = 'transform 1.5s ease-out';
                container.style.transform = 'translate(0, 0) scale(1) rotate(0deg)';
            }, durationSeconds * 0.6 * 1000);
        }



        // Auto-start playback when page loads
        document.addEventListener('DOMContentLoaded', function() {
            console.log('Page loaded, attempting to start playback...');

            // Add a small delay to ensure WebSocket is ready
            setTimeout(() => {
                fetch('/api/start')
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            console.log('Auto-started playback for big screen');
                        } else {
                            console.error('Failed to auto-start:', data.message);
                        }
                    })
                    .catch(error => {
                        console.error('Error auto-starting:', error);
                    });
            }, 1000); // Wait 1 second for WebSocket to connect
        });
    </script>
</body>
</html>'''

    with open(template_dir / 'big_screen.html', 'w', encoding='utf-8') as f:
        f.write(big_screen_content)

    print(f"Created templates at {template_dir / 'index.html'} and {template_dir / 'big_screen.html'}")

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Image player server for OD testing with motion effects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python image_player_server.py --images ../image_selector/categorized_images
  python image_player_server.py -i ./categorized_images --port 5002
        """
    )

    parser.add_argument(
        '-i', '--images',
        required=True,
        help='Path to categorized images folder'
    )

    parser.add_argument(
        '-p', '--port',
        type=int,
        default=5002,
        help='Port to run the server on (default: 5002)'
    )

    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host to bind the server to (default: 127.0.0.1)'
    )

    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Playback duration per image in seconds (default: 60)'
    )
    parser.add_argument(
        '--motion-intensity',
        type=int,
        default=100,
        choices=range(0, 101),
        metavar='[0-100]',
        help='Motion effect intensity (0=no motion, 100=maximum intensity, default: 100)'
    )

    args = parser.parse_args()

    # Validate path
    if not os.path.exists(args.images):
        print(f"Error: Images folder '{args.images}' does not exist")
        return 1

    # Create templates
    create_templates()

    # Initialize player
    global player
    player = ImagePlayer(args.images, args.duration, args.motion_intensity)

    if not player.current_playlist:
        print("Error: No images found in the specified directory")
        return 1

    print("Starting Image Player Server...")
    print(f"Loaded {len(player.current_playlist)} images for playback")
    print(f"Images directory (absolute): {player.categorized_images_path}")
    print(f"Images directory exists: {player.categorized_images_path.exists()}")
    print(f"Playback duration: {player.playback_duration} seconds per image")
    print(f"Server will be available at: http://{args.host}:{args.port}")
    print("\nControls:")
    print(f"- Images will play for {player.playback_duration} seconds each with motion effects")
    print("- GIF and video files play without motion effects")
    print("- Playback history is automatically recorded")
    print("- Use the web interface to control playback")

    try:
        socketio.run(app, host=args.host, port=args.port, debug=False)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        # Save history before exiting
        if player and player.playback_history:
            player.save_playback_history('playback_history.jsonl')
    except Exception as e:
        print(f"Error starting server: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
