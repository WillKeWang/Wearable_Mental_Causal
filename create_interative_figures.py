"""
Create an interactive HTML visualization with a slider to explore different thresholds.
"""

import json
import os
import base64
from io import BytesIO


def create_interactive_html(figures_dir='figures', 
                        #    before_dataset='6w_to_2w_before',
                        #    after_dataset='1w_to_5w_after',
                            dataset='6w_to_2w_before',
                           thresholds=[50, 60, 70, 80],
                           output_file='interactive_causal_viz.html'):
    """
    Create an interactive HTML file with a slider to switch between threshold visualizations.
    
    Args:
        figures_dir: Directory containing the PNG files
        # before_dataset: Name of before dataset
        # after_dataset: Name of after dataset
        dataset: Name of the dataset
        thresholds: List of thresholds that were generated
        output_file: Output HTML filename
    """
    
    # Collect image paths
    image_data = {}
    for threshold in thresholds:
        img_path = os.path.join(figures_dir, 
            # f'causal_graph_{before_dataset}_vs_{after_dataset}_{threshold}pct.png')
            f'causal_graph_{dataset}_{threshold}pct.png')
        
        if os.path.exists(img_path):
            # Read and encode image as base64
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                image_data[threshold] = img_base64
        else:
            print(f"Warning: Image not found: {img_path}")
    
    if not image_data:
        print("Error: No images found!")
        return
    
    # Create HTML with embedded images and slider
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Causal Discovery Visualization</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 10px;
        }}
        
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }}
        
        .controls {{
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        
        .slider-container {{
            display: flex;
            align-items: center;
            gap: 20px;
            margin-bottom: 15px;
        }}
        
        .slider-label {{
            font-weight: bold;
            min-width: 120px;
            color: #333;
        }}
        
        .slider {{
            flex: 1;
            height: 8px;
            -webkit-appearance: none;
            appearance: none;
            background: #ddd;
            outline: none;
            border-radius: 5px;
        }}
        
        .slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 24px;
            height: 24px;
            background: #4CAF50;
            cursor: pointer;
            border-radius: 50%;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }}
        
        .slider::-moz-range-thumb {{
            width: 24px;
            height: 24px;
            background: #4CAF50;
            cursor: pointer;
            border-radius: 50%;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            border: none;
        }}
        
        .threshold-value {{
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
            min-width: 80px;
            text-align: right;
        }}
        
        .info-box {{
            display: flex;
            justify-content: space-around;
            padding: 15px;
            background-color: #e8f5e9;
            border-radius: 5px;
            margin-bottom: 10px;
        }}
        
        .info-item {{
            text-align: center;
        }}
        
        .info-label {{
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }}
        
        .info-value {{
            font-size: 18px;
            font-weight: bold;
            color: #2e7d32;
        }}
        
        .image-container {{
            text-align: center;
            margin-top: 20px;
        }}
        
        .causal-image {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .description {{
            margin-top: 20px;
            padding: 15px;
            background-color: #fff3e0;
            border-left: 4px solid #ff9800;
            border-radius: 4px;
        }}
        
        .description h3 {{
            margin-top: 0;
            color: #e65100;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Interactive Causal Discovery Visualization</h1>
        <div class="subtitle">
            {dataset}
        </div>
        
        <div class="controls">
            <div class="slider-container">
                <div class="slider-label">Edge Frequency Threshold:</div>
                <input type="range" 
                       id="thresholdSlider" 
                       class="slider"
                       min="0" 
                       max="{len(thresholds)-1}" 
                       value="0" 
                       step="1">
                <div class="threshold-value" id="thresholdValue">{thresholds[0]}%</div>
            </div>
            
            <div class="info-box" id="infoBox">
                <div class="info-item">
                    <div class="info-label">Current Threshold</div>
                    <div class="info-value" id="currentThreshold">{thresholds[0]}%</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Available Thresholds</div>
                    <div class="info-value">{', '.join(map(str, thresholds))}%</div>
                </div>
            </div>
        </div>
        
        <div class="image-container">
            <img id="causalImage" class="causal-image" src="" alt="Causal Discovery Graph">
        </div>
    </div>
    
    <script>
        // Image data embedded in JavaScript
        const imageData = {json.dumps({str(k): f"data:image/png;base64,{v}" for k, v in image_data.items()})};
        const thresholds = {json.dumps(thresholds)};
        
        const slider = document.getElementById('thresholdSlider');
        const thresholdValue = document.getElementById('thresholdValue');
        const currentThreshold = document.getElementById('currentThreshold');
        const causalImage = document.getElementById('causalImage');
        
        // Initialize with first image
        updateImage(0);
        
        // Update image when slider changes
        slider.addEventListener('input', function() {{
            updateImage(this.value);
        }});
        
        function updateImage(index) {{
            const threshold = thresholds[index];
            thresholdValue.textContent = threshold + '%';
            currentThreshold.textContent = threshold + '%';
            causalImage.src = imageData[threshold];
        }}
    </script>
</body>
</html>
"""
    
    # Write HTML file
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"✓ Interactive visualization created: {output_file}")
    print(f"  Open this file in your web browser to explore the visualizations")
    print(f"  Thresholds available: {thresholds}")


if __name__ == "__main__":
    # Configuration - match your master script settings
    create_interactive_html(
        figures_dir='test_figures',
        # before_dataset='6w_to_2w_before',
        # after_dataset='1w_to_5w_after',
        dataset='4w_to_0w_before',
        thresholds=[50, 60, 70, 80],
        output_file="interactive_causal_viz.html"
    )