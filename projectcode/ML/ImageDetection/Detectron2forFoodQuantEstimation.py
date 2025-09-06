import torch
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
import cv2
import os
import numpy as np

# Ensure CUDA is available
assert torch.cuda.is_available(), "CUDA is not available. Please check your setup."

print("PyTorch version:", torch.__version__)
print("Detectron2 version:", detectron2.__version__)
print("CUDA is available:", torch.cuda.is_available())
print("GPU device name:", torch.cuda.get_device_name(0))

# Get a default Detectron2 configuration
cfg = get_cfg()
cfg.merge_from_file(detectron2.model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = detectron2.model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

predictor = DefaultPredictor(cfg)

input_image_path = "/app/mediaFiles/images/inputimages/frame_126.jpg"
im = cv2.imread(input_image_path)
if im is None:
    raise FileNotFoundError(f"Image not found at {input_image_path}")

print(f"Running inference on {input_image_path}...")
outputs = predictor(im)

instances = outputs["instances"].to("cpu")
pred_class_ids = instances.pred_classes.tolist()
pred_masks = instances.pred_masks
scores = instances.scores.tolist()
pred_boxes = instances.pred_boxes.tensor.tolist() # Get bounding boxes

class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN).thing_classes

# --- Reference Object Configuration (Person) ---
REFERENCE_OBJECT_NAME = "person"
# Average adult shoulder width in cm (this is a heuristic, real-world variance is high!)
# You might choose other reliable average dimensions if available (e.g., head size, known object).
AVERAGE_REFERENCE_WIDTH_CM = 45.0
PIXELS_PER_CM = None # Initialize as None

# First, find a person to use as a reference
person_boxes = []
for i, class_name in enumerate(class_names):
    if class_name == REFERENCE_OBJECT_NAME:
        for j, pred_id in enumerate(pred_class_ids):
            if pred_id == i: # Check if this detection is the reference object
                person_boxes.append(pred_boxes[j])

if person_boxes:
    # Use the first detected person as the reference
    # For more robustness, you might select the largest person, or the one closest to the center
    ref_box = person_boxes[0]
    ref_width_pixels = ref_box[2] - ref_box[0] # x2 - x1
    
    if ref_width_pixels > 0:
        PIXELS_PER_CM = ref_width_pixels / AVERAGE_REFERENCE_WIDTH_CM
        print(f"\n--- Reference Object Calibration ({REFERENCE_OBJECT_NAME}) ---")
        print(f"  Detected reference width in pixels: {ref_width_pixels:.2f}")
        print(f"  Using average real-world width: {AVERAGE_REFERENCE_WIDTH_CM:.1f} cm")
        print(f"  Calculated Pixel-to-CM ratio: {PIXELS_PER_CM:.2f} pixels/cm")
    else:
        print(f"\nWarning: Detected {REFERENCE_OBJECT_NAME} has zero width in pixels. Cannot calibrate.")
else:
    print(f"\nWarning: No {REFERENCE_OBJECT_NAME} detected to use as reference. Cannot calibrate.")

# --- Object Pixel Counts and Size/Weight Estimates ---
print("\n--- Object Estimates (Banana) ---")

BANANA_DENSITY_G_PER_CM3 = 1.15 # Approx density of banana (g/cm^3)

for i, (pred_class_id, score, mask, bbox_xyxy) in enumerate(zip(
    pred_class_ids, scores, pred_masks, pred_boxes
)):
    class_name = class_names[pred_class_id]

    # Only process bananas for size/weight estimation
    if class_name == "banana":
        num_pixels = mask.sum().item()
        
        print(f"  Object {i+1} ({class_name}):")
        print(f"    Confidence: {score:.2f}")
        print(f"    Number of pixels: {num_pixels}")

        if PIXELS_PER_CM is not None and PIXELS_PER_CM > 0:
            # Convert bounding box dimensions to cm
            bbox_width_pixels = bbox_xyxy[2] - bbox_xyxy[0]
            bbox_height_pixels = bbox_xyxy[3] - bbox_xyxy[1]
            width_cm = bbox_width_pixels / PIXELS_PER_CM
            height_cm = bbox_height_pixels / PIXELS_PER_CM

            print(f"    Estimated BBox Dimensions: Width={width_cm:.2f} cm, Height={height_cm:.2f} cm")

            # Crude volume estimation for banana (approximate as a cylinder)
            # This is highly dependent on orientation and only seeing 2D projection.
            # Assume length is related to height, diameter is related to width,
            # but this needs proper calibration or shape fitting for accuracy.
            estimated_length_cm = height_cm
            # A banana's diameter is often roughly 1/4 to 1/3 of its length.
            # Using width_cm from bbox as a proxy for diameter (can be inaccurate if rotated)
            estimated_diameter_cm = width_cm
            
            # Use average banana dimensions if needed, or refine based on image features.
            # Average dimensions for a medium banana: Length ~18-20cm, Diameter ~3.5cm
            # This volume calculation is a rough example; better models needed for real application.

            if estimated_length_cm > 0 and estimated_diameter_cm > 0:
                # Volume of a cylinder: V = pi * (radius)^2 * height
                estimated_volume_cm3 = (np.pi * (estimated_diameter_cm / 2)**2 * estimated_length_cm)
                print(f"    Estimated Volume (Cylindrical Approx): {estimated_volume_cm3:.2f} cm^3")

                estimated_weight_g = estimated_volume_cm3 * BANANA_DENSITY_G_PER_CM3
                print(f"    Estimated Weight: {estimated_weight_g:.2f} g")
            else:
                print("    Cannot estimate volume/weight reliably from current dimensions.")

        else:
            print("    Cannot estimate size/weight: Reference object not found or calibration failed.")


# Visualize the results
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN).set(thing_classes=class_names), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# Save the output image
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
output_image_path = os.path.join(output_dir, "output_image.jpg")
cv2.imwrite(output_image_path, out.get_image()[:, :, ::-1])
print(f"Processed image saved to {output_image_path}")

print("\nInference complete!")
