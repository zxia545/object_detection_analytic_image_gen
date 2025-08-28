#!/usr/bin/env python3
"""
Image Selector Server for Object Detection Dataset
Displays images with their corresponding prompts and metadata in a web UI
Allows users to select or drop images and save selections to new dataset
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
import re

app = Flask(__name__)

class ImageSelectorServer:
    def __init__(self, image_folder, jsonl_file, output_folder=None):
        self.image_folder = Path(image_folder)
        self.jsonl_file = Path(jsonl_file)
        self.output_folder = Path(output_folder) if output_folder else Path.cwd() / "selected_images"
        self.image_data = {}
        self.image_list = []
        self.current_index = 0
        self.selected_images = set()  # Store selected image IDs
        self.dropped_images = set()   # Store dropped image IDs

        # Load and process data
        self.load_jsonl_data()
        self.load_images()
        self.setup_output_folder()

    def setup_output_folder(self):
        """Setup output folder"""
        self.output_folder.mkdir(exist_ok=True)
        print(f"Output folder: {self.output_folder}")

    def load_jsonl_data(self):
        """Load metadata from JSONL file"""
        print(f"Loading metadata from {self.jsonl_file}")
        with open(self.jsonl_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    test_case_id = data.get('test_case_id', f'line_{line_num}')
                    self.image_data[test_case_id] = data
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue
        print(f"Loaded metadata for {len(self.image_data)} test cases")

    def load_images(self):
        """Load list of available images"""
        print(f"Scanning for images in {self.image_folder}")
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}

        for file_path in self.image_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                # Extract test case ID from filename (remove extension)
                test_case_id = file_path.stem
                if test_case_id in self.image_data:
                    self.image_list.append({
                        'filename': file_path.name,
                        'test_case_id': test_case_id,
                        'filepath': str(file_path)
                    })

        # Sort by test case ID for consistent ordering
        self.image_list.sort(key=lambda x: x['test_case_id'])
        print(f"Found {len(self.image_list)} images with matching metadata")

    def get_current_image_data(self):
        """Get data for current image"""
        if not self.image_list:
            return None

        current_item = self.image_list[self.current_index]
        metadata = self.image_data.get(current_item['test_case_id'], {})

        return {
            'image': current_item,
            'metadata': metadata,
            'current_index': self.current_index,
            'total_count': len(self.image_list),
            'is_selected': current_item['test_case_id'] in self.selected_images,
            'is_dropped': current_item['test_case_id'] in self.dropped_images
        }

    def select_image(self, test_case_id):
        """Select an image"""
        if test_case_id in self.dropped_images:
            self.dropped_images.remove(test_case_id)
        self.selected_images.add(test_case_id)
        print(f"Selected image: {test_case_id}")

    def drop_image(self, test_case_id):
        """Drop an image"""
        if test_case_id in self.selected_images:
            self.selected_images.remove(test_case_id)
        self.dropped_images.add(test_case_id)
        print(f"Dropped image: {test_case_id}")

    def save_selections(self):
        """Save selected images and generate new JSONL file"""
        selected_data = []
        selected_folder = self.output_folder / "selected_images"
        selected_folder.mkdir(exist_ok=True)

        print(f"Saving selections to {self.output_folder}")

        # Copy selected images and collect their data
        for test_case_id in self.selected_images:
            if test_case_id in self.image_data:
                # Copy image file
                image_item = next((item for item in self.image_list if item['test_case_id'] == test_case_id), None)
                if image_item:
                    src_path = Path(image_item['filepath'])
                    dst_path = selected_folder / src_path.name
                    shutil.copy2(src_path, dst_path)
                    print(f"Copied: {src_path.name}")

                # Add to selected data
                selected_data.append(self.image_data[test_case_id])

        # Save new JSONL file
        jsonl_output_path = self.output_folder / "selected_dataset.jsonl"
        with open(jsonl_output_path, 'w') as f:
            for data in selected_data:
                f.write(json.dumps(data) + '\n')

        print(f"Saved {len(selected_data)} selected images and metadata to {jsonl_output_path}")
        return len(selected_data)

    def format_risk_tags(self, risk_tags):
        """Format risk tags for human readability"""
        if not risk_tags:
            return "None"

        formatted_tags = []
        for tag in risk_tags:
            # Convert snake_case to Title Case
            formatted = tag.replace('_', ' ').title()
            formatted_tags.append(formatted)

        return ", ".join(formatted_tags)

    def format_attributes(self, attributes):
        """Format attributes for human readability"""
        if not attributes:
            return {}

        formatted = {}
        for key, value in attributes.items():
            # Convert snake_case to Title Case
            formatted_key = key.replace('_', ' ').title()
            if isinstance(value, (int, float)):
                if key == 'occlusion_pct':
                    formatted_value = f"{value}%"
                elif key == 'object_count':
                    formatted_value = str(value)
                else:
                    formatted_value = str(value)
            else:
                formatted_value = str(value).replace('_', ' ').title()

            formatted[formatted_key] = formatted_value

        return formatted

# Global server instance
server = None

@app.route('/')
def index():
    """Main page"""
    if not server or not server.image_list:
        return "No images found or server not initialized", 404

    data = server.get_current_image_data()
    if not data:
        return "No image data available", 404

    return render_template('index.html', data=data, server=server)

@app.route('/api/current')
def get_current():
    """API endpoint to get current image data"""
    if not server:
        return jsonify({'error': 'Server not initialized'}), 500

    data = server.get_current_image_data()
    if not data:
        return jsonify({'error': 'No image data available'}), 404

    # Format the data for the frontend
    metadata = data['metadata']
    formatted_data = {
        'image': data['image'],
        'test_case_id': metadata.get('test_case_id', ''),
        'scenario_category': metadata.get('scenario_category', '').replace('_', ' ').title(),
        'test_subcategory': metadata.get('test_subcategory', '').replace('_', ' ').title(),
        'property_type': metadata.get('property_type', '').replace('_', ' ').title(),
        'od_type_primary': metadata.get('od_type_primary', '').replace('_', ' ').title(),
        'od_state': metadata.get('od_state', '').replace('_', ' ').title(),
        'motion_type': metadata.get('motion_type', '').replace('_', ' ').title(),
        'background_prompt': metadata.get('background_prompt', ''),
        'positive_prompt': metadata.get('positive_prompt', ''),
        'negative_prompt': metadata.get('negative_prompt', ''),
        'composition_guidance': metadata.get('composition_guidance', ''),
        'prompt': metadata.get('prompt', ''),
        'expected_detection': metadata.get('expected_detection', {}),
        'risk_tags': server.format_risk_tags(metadata.get('risk_tags', [])),
        'attributes': server.format_attributes(metadata.get('attributes', {})),
        'current_index': data['current_index'],
        'total_count': data['total_count'],
        'is_selected': data['is_selected'],
        'is_dropped': data['is_dropped'],
        'selected_count': len(server.selected_images),
        'dropped_count': len(server.dropped_images)
    }

    return jsonify(formatted_data)

@app.route('/api/next')
def next_image():
    """Go to next image"""
    if not server:
        return jsonify({'error': 'Server not initialized'}), 500

    server.current_index = (server.current_index + 1) % len(server.image_list)
    return jsonify({'success': True, 'new_index': server.current_index})

@app.route('/api/prev')
def prev_image():
    """Go to previous image"""
    if not server:
        return jsonify({'error': 'Server not initialized'}), 500

    server.current_index = (server.current_index - 1) % len(server.image_list)
    return jsonify({'success': True, 'new_index': server.current_index})

@app.route('/api/goto/<int:index>')
def goto_image(index):
    """Go to specific image index"""
    if not server:
        return jsonify({'error': 'Server not initialized'}), 500

    if 0 <= index < len(server.image_list):
        server.current_index = index
        return jsonify({'success': True, 'new_index': server.current_index})
    else:
        return jsonify({'error': 'Invalid index'}), 400

@app.route('/api/select/<test_case_id>')
def select_image(test_case_id):
    """Select an image"""
    if not server:
        return jsonify({'error': 'Server not initialized'}), 500

    server.select_image(test_case_id)
    return jsonify({'success': True, 'selected_count': len(server.selected_images)})

@app.route('/api/drop/<test_case_id>')
def drop_image(test_case_id):
    """Drop an image"""
    if not server:
        return jsonify({'error': 'Server not initialized'}), 500

    server.drop_image(test_case_id)
    return jsonify({'success': True, 'dropped_count': len(server.dropped_images)})

@app.route('/api/save')
def save_selections():
    """Save selected images and generate new JSONL"""
    if not server:
        return jsonify({'error': 'Server not initialized'}), 500

    try:
        saved_count = server.save_selections()
        return jsonify({
            'success': True,
            'saved_count': saved_count,
            'output_folder': str(server.output_folder)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve images from the image folder"""
    if not server:
        return "Server not initialized", 500

    return send_from_directory(server.image_folder, filename)

def create_templates():
    """Create HTML template for the UI"""
    template_dir = Path(__file__).parent / 'templates'
    template_dir.mkdir(exist_ok=True)

    template_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Selector - Object Detection Dataset</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
            color: #333;
        }

        .header {
            background: #2c3e50;
            color: white;
            padding: 1rem;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .header h1 {
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
        }

        .header .subtitle {
            font-size: 1rem;
            opacity: 0.9;
        }

        .stats-bar {
            background: #34495e;
            color: white;
            padding: 0.5rem 1rem;
            text-align: center;
            font-size: 0.9rem;
        }

        .container {
            display: flex;
            height: calc(100vh - 140px);
            padding: 1rem;
            gap: 1rem;
        }

        .left-panel {
            flex: 1;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        .right-panel {
            flex: 1;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 1.5rem;
            overflow-y: auto;
        }

        .image-container {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #f8f9fa;
            position: relative;
        }

        .image-container img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            border-radius: 4px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }

        .image-status {
            position: absolute;
            top: 10px;
            right: 10px;
            padding: 0.5rem;
            border-radius: 4px;
            font-weight: bold;
            font-size: 0.9rem;
        }

        .status-selected {
            background: #27ae60;
            color: white;
        }

        .status-dropped {
            background: #e74c3c;
            color: white;
        }

        .image-info {
            padding: 1rem;
            background: #f8f9fa;
            border-top: 1px solid #e9ecef;
        }

        .image-info h3 {
            color: #2c3e50;
            margin-bottom: 0.5rem;
            font-size: 1.2rem;
        }

        .navigation {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem;
            background: #f8f9fa;
            border-top: 1px solid #e9ecef;
            gap: 1rem;
        }

        .nav-btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1rem;
            transition: background 0.2s;
        }

        .nav-btn:hover {
            background: #2980b9;
        }

        .nav-btn:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
        }

        .action-btn {
            background: #27ae60;
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1rem;
            transition: background 0.2s;
            margin: 0 0.25rem;
        }

        .action-btn:hover {
            background: #219a52;
        }

        .drop-btn {
            background: #e74c3c;
        }

        .drop-btn:hover {
            background: #c0392b;
        }

        .save-btn {
            background: #f39c12;
            margin-top: 1rem;
        }

        .save-btn:hover {
            background: #e67e22;
        }

        .counter {
            font-size: 1.1rem;
            color: #7f8c8d;
            font-weight: 500;
            white-space: nowrap;
        }

        .stats {
            display: flex;
            gap: 1rem;
            font-size: 0.9rem;
        }

        .stat-item {
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
        }

        .stat-selected {
            background: #d5f4e6;
            color: #27ae60;
        }

        .stat-dropped {
            background: #fadbd8;
            color: #e74c3c;
        }

        .metadata-section {
            margin-bottom: 2rem;
        }

        .metadata-section h3 {
            color: #2c3e50;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #3498db;
            font-size: 1.3rem;
        }

        .metadata-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .metadata-item {
            background: #f8f9fa;
            padding: 0.75rem;
            border-radius: 6px;
            border-left: 4px solid #3498db;
        }

        .metadata-label {
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 0.25rem;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .metadata-value {
            color: #555;
            font-size: 1rem;
        }

        .prompt-section {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 6px;
            margin-bottom: 1rem;
        }

        .prompt-label {
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 0.5rem;
            display: block;
        }

        .prompt-text {
            background: white;
            padding: 0.75rem;
            border-radius: 4px;
            border: 1px solid #e9ecef;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            line-height: 1.4;
            white-space: pre-wrap;
            max-height: 200px;
            overflow-y: auto;
        }

        .risk-tags {
            background: #e74c3c;
            color: white;
            padding: 0.75rem;
            border-radius: 6px;
            margin-bottom: 1rem;
        }

        .risk-tags .metadata-label {
            color: white;
        }

        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 200px;
            color: #7f8c8d;
            font-size: 1.2rem;
        }

        .error {
            background: #e74c3c;
            color: white;
            padding: 1rem;
            border-radius: 6px;
            margin: 1rem 0;
        }

        .save-result {
            background: #27ae60;
            color: white;
            padding: 1rem;
            border-radius: 6px;
            margin: 1rem 0;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Object Detection Dataset Selector</h1>
        <div class="subtitle">Select or Drop Images to Create New Dataset</div>
    </div>

    <div class="stats-bar">
        <div class="stats">
            <div class="counter" id="counter">0 / 0</div>
            <div class="stat-item stat-selected" id="selectedCount">Selected: 0</div>
            <div class="stat-item stat-dropped" id="droppedCount">Dropped: 0</div>
        </div>
    </div>

    <div class="container">
        <div class="left-panel">
            <div class="image-container" id="imageContainer">
                <div class="loading">Loading image...</div>
            </div>
            <div class="image-info" id="imageInfo">
                <h3>Image Information</h3>
                <div id="imageDetails">Loading...</div>
            </div>
            <div class="navigation">
                <button class="nav-btn" id="prevBtn" onclick="previousImage()">← Previous</button>
                <div style="display: flex; gap: 0.5rem;">
                    <button class="action-btn" id="selectBtn" onclick="selectImage()">Select</button>
                    <button class="action-btn drop-btn" id="dropBtn" onclick="dropImage()">Drop</button>
                </div>
                <button class="nav-btn" id="nextBtn" onclick="nextImage()">Next →</button>
            </div>
            <div style="padding: 1rem; background: #f8f9fa; border-top: 1px solid #e9ecef;">
                <button class="action-btn save-btn" id="saveBtn" onclick="saveSelections()" style="width: 100%;">Save Selections</button>
                <div id="saveResult" style="margin-top: 1rem;"></div>
            </div>
        </div>

        <div class="right-panel" id="rightPanel">
            <div class="loading">Loading metadata...</div>
        </div>
    </div>

    <script>
        let currentData = null;

        async function loadCurrentImage() {
            try {
                const response = await fetch('/api/current');
                if (!response.ok) throw new Error('Failed to load image data');

                const data = await response.json();
                currentData = data;
                displayImage(data);
                displayMetadata(data);
                updateNavigation();
                updateStats();
            } catch (error) {
                console.error('Error loading image:', error);
                document.getElementById('imageContainer').innerHTML =
                    '<div class="error">Error loading image: ' + error.message + '</div>';
                document.getElementById('rightPanel').innerHTML =
                    '<div class="error">Error loading metadata: ' + error.message + '</div>';
            }
        }

        function displayImage(data) {
            const container = document.getElementById('imageContainer');
            const info = document.getElementById('imageInfo');

            // Clear previous status
            container.innerHTML = `<img src="/images/${data.image.filename}" alt="${data.test_case_id}">`;

            // Add status indicator
            if (data.is_selected) {
                container.innerHTML += '<div class="image-status status-selected">SELECTED</div>';
            } else if (data.is_dropped) {
                container.innerHTML += '<div class="image-status status-dropped">DROPPED</div>';
            }

            info.innerHTML = `
                <h3>${data.test_case_id}</h3>
                <div><strong>Filename:</strong> ${data.image.filename}</div>
                <div><strong>Status:</strong>
                    <span style="color: ${data.is_selected ? '#27ae60' : (data.is_dropped ? '#e74c3c' : '#7f8c8d')}">
                        ${data.is_selected ? 'Selected' : (data.is_dropped ? 'Dropped' : 'Not Reviewed')}
                    </span>
                </div>
            `;
        }

        function displayMetadata(data) {
            const panel = document.getElementById('rightPanel');

            panel.innerHTML = `
                <div class="metadata-section">
                    <h3>Test Case Information</h3>
                    <div class="metadata-grid">
                        <div class="metadata-item">
                            <div class="metadata-label">Test Case ID</div>
                            <div class="metadata-value">${data.test_case_id}</div>
                        </div>
                        <div class="metadata-item">
                            <div class="metadata-label">Scenario Category</div>
                            <div class="metadata-value">${data.scenario_category}</div>
                        </div>
                        <div class="metadata-item">
                            <div class="metadata-label">Test Subcategory</div>
                            <div class="metadata-value">${data.test_subcategory}</div>
                        </div>
                        <div class="metadata-item">
                            <div class="metadata-label">Property Type</div>
                            <div class="metadata-value">${data.property_type}</div>
                        </div>
                        <div class="metadata-item">
                            <div class="metadata-label">OD Type Primary</div>
                            <div class="metadata-value">${data.od_type_primary}</div>
                        </div>
                        <div class="metadata-item">
                            <div class="metadata-label">OD State</div>
                            <div class="metadata-value">${data.od_state}</div>
                        </div>
                        <div class="metadata-item">
                            <div class="metadata-label">Motion Type</div>
                            <div class="metadata-value">${data.motion_type}</div>
                        </div>
                    </div>
                </div>

                <div class="metadata-section">
                    <h3>Risk Assessment</h3>
                    <div class="risk-tags">
                        <div class="metadata-label">Risk Tags</div>
                        <div class="metadata-value">${data.risk_tags}</div>
                    </div>
                </div>

                <div class="metadata-section">
                    <h3>Scene Attributes</h3>
                    <div class="metadata-grid">
                        ${Object.entries(data.attributes).map(([key, value]) => `
                            <div class="metadata-item">
                                <div class="metadata-label">${key}</div>
                                <div class="metadata-value">${value}</div>
                            </div>
                        `).join('')}
                    </div>
                </div>

                <div class="metadata-section">
                    <h3>Expected Detection</h3>
                    <div class="metadata-grid">
                        ${Object.entries(data.expected_detection).map(([key, value]) => `
                            <div class="metadata-item">
                                <div class="metadata-label">${key.replace('_', ' ').toUpperCase()}</div>
                                <div class="metadata-value">${value ? 'Yes' : 'No'}</div>
                            </div>
                        `).join('')}
                    </div>
                </div>

                <div class="metadata-section">
                    <h3>Background Prompt</h3>
                    <div class="prompt-section">
                        <span class="prompt-label">Scene Description</span>
                        <div class="prompt-text">${data.background_prompt}</div>
                    </div>
                </div>

                <div class="metadata-section">
                    <h3>Generation Prompts</h3>
                    <div class="prompt-section">
                        <span class="prompt-label">Positive Prompt</span>
                        <div class="prompt-text">${data.positive_prompt}</div>
                    </div>
                    <div class="prompt-section">
                        <span class="prompt-label">Negative Prompt</span>
                        <div class="prompt-text">${data.negative_prompt}</div>
                    </div>
                    <div class="prompt-section">
                        <span class="prompt-label">Composition Guidance</span>
                        <div class="prompt-text">${data.composition_guidance}</div>
                    </div>
                </div>

                <div class="metadata-section">
                    <h3>Full Prompt</h3>
                    <div class="prompt-section">
                        <span class="prompt-label">Complete Generation Prompt</span>
                        <div class="prompt-text">${data.prompt}</div>
                    </div>
                </div>
            `;
        }

        function updateNavigation() {
            document.getElementById('counter').textContent =
                `${currentData.current_index + 1} / ${currentData.total_count}`;

            document.getElementById('prevBtn').disabled = currentData.total_count <= 1;
            document.getElementById('nextBtn').disabled = currentData.total_count <= 1;
        }

        function updateStats() {
            document.getElementById('selectedCount').textContent = `Selected: ${currentData.selected_count}`;
            document.getElementById('droppedCount').textContent = `Dropped: ${currentData.dropped_count}`;
        }

        async function nextImage() {
            try {
                const response = await fetch('/api/next');
                if (response.ok) {
                    await loadCurrentImage();
                }
            } catch (error) {
                console.error('Error navigating to next image:', error);
            }
        }

        async function previousImage() {
            try {
                const response = await fetch('/api/prev');
                if (response.ok) {
                    await loadCurrentImage();
                }
            } catch (error) {
                console.error('Error navigating to previous image:', error);
            }
        }

        async function selectImage() {
            if (!currentData) return;

            try {
                const response = await fetch(`/api/select/${currentData.test_case_id}`);
                if (response.ok) {
                    await loadCurrentImage();
                }
            } catch (error) {
                console.error('Error selecting image:', error);
            }
        }

        async function dropImage() {
            if (!currentData) return;

            try {
                const response = await fetch(`/api/drop/${currentData.test_case_id}`);
                if (response.ok) {
                    await loadCurrentImage();
                }
            } catch (error) {
                console.error('Error dropping image:', error);
            }
        }

        async function saveSelections() {
            try {
                const saveBtn = document.getElementById('saveBtn');
                const saveResult = document.getElementById('saveResult');

                saveBtn.disabled = true;
                saveBtn.textContent = 'Saving...';
                saveResult.innerHTML = '';

                const response = await fetch('/api/save');
                const result = await response.json();

                if (response.ok) {
                    saveResult.innerHTML = `
                        <div class="save-result">
                            <strong>Success!</strong><br>
                            Saved ${result.saved_count} images to: ${result.output_folder}/selected_images<br>
                            Created new dataset: ${result.output_folder}/selected_dataset.jsonl
                        </div>
                    `;
                } else {
                    saveResult.innerHTML = `<div class="error">Error: ${result.error}</div>`;
                }
            } catch (error) {
                document.getElementById('saveResult').innerHTML =
                    `<div class="error">Error saving selections: ${error.message}</div>`;
            } finally {
                const saveBtn = document.getElementById('saveBtn');
                saveBtn.disabled = false;
                saveBtn.textContent = 'Save Selections';
            }
        }

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') {
                previousImage();
            } else if (e.key === 'ArrowRight') {
                nextImage();
            } else if (e.key === 's' || e.key === 'S') {
                selectImage();
            } else if (e.key === 'd' || e.key === 'D') {
                dropImage();
            }
        });

        // Load initial data
        document.addEventListener('DOMContentLoaded', loadCurrentImage);
    </script>
</body>
</html>'''

    with open(template_dir / 'index.html', 'w') as f:
        f.write(template_content)

    print(f"Created template at {template_dir / 'index.html'}")

def main():
    parser = argparse.ArgumentParser(
        description='Host an image selector server for object detection dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python host_image_selector_server.py --images downloaded_images --jsonl dataset/od_synth_cases_10000_cctv_v1_.jsonl
  python host_image_selector_server.py -i images_folder -j metadata.jsonl --output selected_dataset
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
        help='Output folder for selected images and new JSONL file (default: selected_images)'
    )

    parser.add_argument(
        '-p', '--port',
        type=int,
        default=5001,
        help='Port to run the server on (default: 5001)'
    )

    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host to bind the server to (default: 127.0.0.1)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode'
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.images):
        print(f"Error: Image folder '{args.images}' does not exist")
        return 1

    if not os.path.exists(args.jsonl):
        print(f"Error: JSONL file '{args.jsonl}' does not exist")
        return 1

    # Create templates
    create_templates()

    # Initialize server
    global server
    server = ImageSelectorServer(args.images, args.jsonl, args.output)

    if not server.image_list:
        print("Error: No images found or no matching metadata")
        return 1

    print(f"Starting selector server on http://{args.host}:{args.port}")
    print(f"Found {len(server.image_list)} images with metadata")
    print(f"Output folder: {server.output_folder}")
    print("Controls:")
    print("  Arrow keys: Navigate between images")
    print("  S key: Select current image")
    print("  D key: Drop current image")
    print("  Press Ctrl+C to stop the server")

    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
