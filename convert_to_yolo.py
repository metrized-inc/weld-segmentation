import json
import os
from pathlib import Path
import shutil
from PIL import Image

class JSONtoYOLOConverter:
    """Convert custom JSON annotations to YOLO segmentation format"""
    
    def __init__(self, data_top_dir, data_bottom_dir, output_dir):
        self.data_top_dir = Path(data_top_dir)
        self.data_bottom_dir = Path(data_bottom_dir)
        self.output_dir = Path(output_dir)
        self.class_names = ['weld']  # Modify as needed
        
    def setup_directories(self):
        """Create YOLO dataset directory structure"""
        dirs = [
            self.output_dir / 'images' / 'train',
            self.output_dir / 'images' / 'val',
            self.output_dir / 'labels' / 'train',
            self.output_dir / 'labels' / 'val'
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def normalize_polygon(self, polyline, img_width, img_height):
        """Normalize polygon coordinates to [0, 1] range for YOLO"""
        normalized = []
        for x, y in polyline:
            norm_x = x / img_width
            norm_y = y / img_height
            normalized.extend([norm_x, norm_y])
        
        # CRITICAL: Close the polygon if not already closed
        if len(polyline) > 0 and polyline[0] != polyline[-1]:
            # Add first point at the end to close the polygon
            first_x, first_y = polyline[0]
            normalized.extend([first_x / img_width, first_y / img_height])
        
        return normalized
    
    def convert_json_to_yolo(self, json_path, output_label_path, img_path):
        """Convert single JSON file to YOLO format"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Get image dimensions
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        yolo_lines = []
        for label in data.get('labels', []):
            polyline = label.get('polyline', [])
            if not polyline:
                continue
            
            # Normalize coordinates
            norm_coords = self.normalize_polygon(polyline, img_width, img_height)
            
            # YOLO format: class_id x1 y1 x2 y2 ... xn yn
            class_id = 0  # Default to class 0, modify if you have multiple classes
            yolo_line = f"{class_id} " + " ".join(map(str, norm_coords))
            yolo_lines.append(yolo_line)
        
        # Write YOLO format file
        with open(output_label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
    
    def process_directory(self, source_dir, split='train', split_ratio=0.8):
        """Process all JSON files in a directory"""
        json_files = list(source_dir.rglob('*.json'))
        
        # Calculate split
        split_idx = int(len(json_files) * split_ratio)
        
        for idx, json_file in enumerate(json_files):
            # Determine train/val split
            current_split = 'train' if idx < split_idx else 'val'
            
            # Load JSON to get image path
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            img_path = Path(data['origin'])
            if not img_path.exists():
                print(f"Warning: Image not found: {img_path}")
                continue
            
            # Create output paths
            img_name = img_path.stem
            output_img = self.output_dir / 'images' / current_split / f"{img_name}.png"
            output_label = self.output_dir / 'labels' / current_split / f"{img_name}.txt"
            
            # Copy image
            shutil.copy2(img_path, output_img)
            
            # Convert and save label
            self.convert_json_to_yolo(json_file, output_label, img_path)
            
            print(f"Processed: {img_name} -> {current_split}")
    
    def create_yaml_config(self):
        """Create YOLO dataset configuration file"""
        yaml_content = f"""# YOLO Dataset Configuration
path: {self.output_dir.absolute()}
train: images/train
val: images/val

# Classes
names:
  0: {self.class_names[0]}

# Number of classes
nc: {len(self.class_names)}
"""
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        print(f"Created dataset config: {yaml_path}")
        return yaml_path
    
    def convert_all(self, split_ratio=0.8):
        """Run full conversion pipeline"""
        print("Setting up directories...")
        self.setup_directories()
        
        print("\nProcessing data-top directory...")
        if self.data_top_dir.exists():
            self.process_directory(self.data_top_dir, split_ratio=split_ratio)
        
        print("\nProcessing data-bottom directory...")
        if self.data_bottom_dir.exists():
            self.process_directory(self.data_bottom_dir, split_ratio=split_ratio)
        
        print("\nCreating YOLO config file...")
        yaml_path = self.create_yaml_config()
        
        print(f"\n✓ Conversion complete! Dataset ready at: {self.output_dir}")
        return yaml_path


# ============= USAGE EXAMPLE =============

if __name__ == "__main__":
    # Configure paths
    DATA_TOP = "data-top"  # Update to your actual path
    DATA_BOTTOM = "data-bottom"  # Update to your actual path
    OUTPUT_DIR = "yolo_dataset"
    
    # Convert dataset
    converter = JSONtoYOLOConverter(DATA_TOP, DATA_BOTTOM, OUTPUT_DIR)
    yaml_path = converter.convert_all(split_ratio=0.8)
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("\n1. Install YOLOv8:")
    print("   pip install ultralytics")
    print("\n2. Train the model:")
    print(f"   python train_yolo.py")
    print("\n3. Run inference:")
    print(f"   python inference_yolo.py")