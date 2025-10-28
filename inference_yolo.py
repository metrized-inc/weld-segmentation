from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import json

class WeldSegmentationInference:
    """Run inference on new images using trained YOLO model"""
    
    def __init__(self, model_path='runs/segment/weld_segmentation/weights/best.pt'):
        self.model = YOLO(model_path)
        print(f"Loaded model from: {model_path}")
    
    def predict_single_image(self, img_path, conf_threshold=0.25, save=True):
        """Run inference on a single image"""
        results = self.model.predict(
            source=img_path,
            conf=conf_threshold,
            save=save,
            save_txt=True,
            save_conf=True,
            project='runs/predict',
            name='inference',
            exist_ok=True
        )
        return results
    
    def predict_directory(self, img_dir, conf_threshold=0.25):
        """Run inference on all images in a directory"""
        img_dir = Path(img_dir)
        results = self.model.predict(
            source=str(img_dir),
            conf=conf_threshold,
            save=True,
            save_txt=True,
            save_conf=True,
            project='runs/predict',
            name='batch_inference',
            exist_ok=True
        )
        return results
    
    def visualize_results(self, img_path, results, output_path=None):
        """Visualize segmentation results with custom styling"""
        img = cv2.imread(str(img_path))
        
        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes.data.cpu().numpy()
                
                # Create colored overlay
                overlay = img.copy()
                
                for mask, box in zip(masks, boxes):
                    # Resize mask to image size
                    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
                    
                    # Create colored mask
                    color = (0, 255, 0)  # Green for welds
                    overlay[mask_resized > 0.5] = color
                    
                    # Draw bounding box
                    x1, y1, x2, y2, conf, cls = box
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Add confidence score
                    label = f"Weld {conf:.2f}"
                    cv2.putText(img, label, (int(x1), int(y1) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Blend overlay with original image
                img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        
        if output_path:
            cv2.imwrite(str(output_path), img)
            print(f"Saved visualization to: {output_path}")
        
        return img
    
    def export_to_json(self, results, output_path):
        """Export results to JSON format similar to input format"""
        all_results = []
        
        for result in results:
            if result.masks is None:
                continue
            
            img_path = result.path
            masks = result.masks.xy  # Polygons in original image coordinates
            boxes = result.boxes.data.cpu().numpy()
            
            labels = []
            for mask, box in zip(masks, boxes):
                polyline = mask.tolist()  # Convert to list of [x, y] points
                conf = float(box[4])
                
                labels.append({
                    "tags": [],
                    "polyline": polyline,
                    "confidence": conf
                })
            
            result_data = {
                "origin": str(img_path),
                "labels": labels
            }
            all_results.append(result_data)
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"Exported results to: {output_path}")
    
    def run_inference_pipeline(self, input_path, conf_threshold=0.25):
        """Complete inference pipeline with visualization and export"""
        input_path = Path(input_path)
        
        print(f"Running inference on: {input_path}")
        results = self.predict_single_image(input_path, conf_threshold)
        
        # Visualize
        vis_path = f"runs/predict/inference/{input_path.stem}_visualized.png"
        self.visualize_results(input_path, results, vis_path)
        
        # Export to JSON
        json_path = f"runs/predict/inference/{input_path.stem}_results.json"
        self.export_to_json(results, json_path)
        
        print("\n" + "="*60)
        print("Inference Complete!")
        print("="*60)
        print(f"Predictions saved to: runs/predict/inference/")
        print(f"Visualization: {vis_path}")
        print(f"JSON export: {json_path}")
        
        return results


# ============= USAGE EXAMPLES =============

def example_single_image():
    """Example: Inference on a single image"""
    predictor = WeldSegmentationInference()
    
    # Update with your image path
    img_path = "data-top/AA105019_U4-TOP_20251023132506_BLT2.2025.10.23.13.26.43.8928_._PRESENCE_.__image.PNG"
    results = predictor.run_inference_pipeline(img_path, conf_threshold=0.25)


def example_batch_inference():
    """Example: Batch inference on directory"""
    predictor = WeldSegmentationInference()
    
    # Update with your directory path
    img_dir = "data-bottom/"
    results = predictor.predict_directory(img_dir, conf_threshold=0.25)
    
    print(f"\nProcessed {len(results)} images")


if __name__ == "__main__":
    # Run single image inference
    # example_single_image()
    
    # Or run batch inference
    example_batch_inference()