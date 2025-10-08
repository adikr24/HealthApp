### Ideally ouput from this would be a collapsed estimation of food object eaten by the preson in the video ###

from projectcode.ML.StandardClasses.RunYoloOnVideoOrImages import YoloRelevantAnnotator
from projectcode.ML.StandardClasses.RunDetectronOnVideosOrImages import DetectronRelevantAnnotator


#### RUN YOLO FIRST ####################################################################
# Choose best weights now; easy to swap later
WEIGHTS = "/app/mediaFiles/yolo_results/YoloWeights/yolov8n.pt" 
CSV     = "/app/mediaFiles/yolo_results/YoloLabels/YoloRelevantLabels.csv"

#/app/mediaFiles/yolo_results/YoloWeights/yolov8l.pt

annot = YoloRelevantAnnotator(
    model_weights=WEIGHTS,
    conf_threshold=0.5,
    relevant_labels_csv=CSV,
)
# step 1 --> process frames in a folder and save results
saved_analysis = annot.process_frames_dir(
    frames_dir="/app/mediaFiles/videos/InputVideos/PeanutButterSandwich/FinalProduct_FoodConsumed/",
    output_dir="/app/mediaFiles/output/videoOutputs/PBJResults/FinalFoodConsumedYolo",
    glob_pat="*.jpg",   # or "frame_*.jpg" if that’s your naming convention
    save_empty=False,
    max_frames=250,  # process all frames
)

print("Frames saved:", saved_analysis)

######## RUN DETECTRON NEXT ##############################################################
det = DetectronRelevantAnnotator(
    model_cfg="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",  # swap to another zoo cfg if you like
    score_threshold=0.5,                                      # tweak like YOLO's conf
    relevant_labels_csv=CSV,                                  # same CSV filter (Relevant == 'Y')
    # weights_path="/path/to/custom_detectron_weights.pth",   # optional custom weights
)

saved_detectron = det.process_frames_dir(
    frames_dir="/app/mediaFiles/videos/InputVideos/PeanutButterSandwich/FinalProduct_FoodConsumed/",
    output_dir="/app/mediaFiles/output/videoOutputs/PBJResults/FinalFoodConsumedDetectron",
    glob_pat="*.jpg",
    save_empty=False,
    max_frames=250,  # process all frames
)

print("Detectron frames saved:", saved_detectron)

### Assuming sandwich is eaten in all frames ###
