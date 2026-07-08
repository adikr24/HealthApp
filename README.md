# 🍽️ Food Recognition & Quantity Estimation Project

This repository contains code and data pipelines to detect food, estimate ingredient quantities during cooking actions, and fetch nutritional information using computer vision, classification, and database connectors.

---

## 📂 Project Structure

### 1. DataEngineering
- **PreprocessPhotos.py** → Resize/prepare images for downstream workflows.
- **PreprocessVideo.py** → Preprocess videos for analysis; it uses the trimming/compression/extraction helper in **InputFileProcessor/PreprocessVidoes_CompressCutAndExtractFrames.py**.

### 2. InputFileProcessor
- **CompressandcutLargeIF.py** → Helper for compression and image preprocessing.
- **PreprocessVidoes_CompressCutAndExtractFrames.py** → Core helper for video trimming, compression, and frame extraction.

### 3. CalEST
This is the main end-to-end pipeline for ingredient/object detection and quantity estimation.

##### FirstPass_AnalyzeAndAnnotateFrames
Purpose: classify and label objects in the input frames.
- **RunCustomYoloToIdentifyObjects.py** → Run the custom YOLO model to detect and label all relevant objects.
- **RefinePredictedLabels.py** → Refine the YOLO detections and assign left/right hand labels used later for hand-object interaction analysis.

##### SecondPass_GenerateMovementMetadata
Purpose: detect vessel activity, hand activity, and generate frame-level references for downstream logic.
- **Step1_VOIActivityMerged.py** → Build vessel-of-interest activity segmentation.
- **Step2_SeparateHandROIFramesBasedOnVOIAct.py** → Create hand interaction frames based on vessel activity.
- **Step3_MergeFSM.py** → Merge FSM heuristics into a single metadata stream.

###### Hand activity pipeline
- **Step1_RefineHandLabels.py** → Merge overlapping hand detections into clean hand boxes.
- **Step2_DrawROIOnHands.py** → Draw small palm-region ROIs around detected hands.
- **Step3_GetNearestObjectFromHands.py** → Link each hand ROI to the nearest detected object and visualize interaction distances.

###### Frame-reference FSM heuristics
The following heuristics separate activities and infer the object of interest for each frame/window:
- **FSM_Heuristic_HandOnlyClass/FSM_HandOnlyClass.py** → Infer the nearest object of interest from hand-only activity segments.
- **FSM_Heuristic_EdibleClass/FSM_EdibleClass.py** → Identify edible classes from YOLO labels and mark them as the OOI.
- **FSM_Heuristic_HandBottleClass/LabelDetection.py** → Extract OCR text from nearby frames.
- **FSM_Heuristic_HandBottleClass/FSM_HandBottleClass.py** → Use OCR output to infer bottle-related ingredients and write them into FSM metadata.

##### ThirdPass_DetectVolume
Purpose: infer the cooking vessel and estimate the amount of the poured ingredient.
- **Step1_DetectVOIOnly.py** → Detect the vessel of interest (for example, a blender) and draw bounding boxes.
- **Step2_DrawBoundingBoxesOnVOI.py** → Use the detected box to define an ROI.
- **Step3_DrawDetectionBandsOnVOIForOOI.py** → Create upper/lower detection bands to focus the volume analysis region.

###### Volume estimation sub-pipelines
- **LiquidObjectsGreekYogurtPour/** → Estimate volume from liquid pour behavior using frame-to-frame luminance/white-mask changes.
  - **Step4_GetLiquidObjectPourVolume_DND.py**
  - **Step5_GreekYogurtVolFlowI.py**
  - **Step6_GreekYogurtPourVolCalibrated.py**
- **PowderyObjectPour/** → Estimate powder pour behavior using scoop crop extraction and a ResNet-based fill-level classifier.
  - **Step4_CropScoop.py**
  - **Step5_GetPowderPourVol.py**
- **SolidObjectsBlueberriesPour/** → Estimate solid-object pour quantity from blob/thickness changes.
  - **Step3_SetupVolumeBandsForObjects.py**
  - **Step4_GetSolidObjectPourQuantity.py**

###### ML training utilities
- **ObjectDetection (Training Custom YOLO)**
  - **ML/ObjectDetection/TrainYolo/TrainYoloOnCustomObjects.py** → Train a custom YOLO model for object detection.
  - The current custom detector is trained for 40 classes defined in the training config.
- **ImageClassification (Training Custom ResNet)**
  - **ML/ImageClassification/TransferLearningResnet** → Example/inference-style ResNet script for image classification.
  - **ML/ImageClassification/InferenceFromResnet.py** → Run inference using a saved ResNet checkpoint.

### 4. Prototyping and Reference

#### ActionDetection
This section contains earlier prototypes and experiment code for action and pour estimation.

##### QuantityDetection-CoffeePour
- **Step1_RunDetectron2.py** → Run Detectron2 to detect objects such as cups and spoons.
- **Step1_RunYolo.py** → Run YOLO for additional object detection.
- **Step2_DrawHeatMapForCoffee.py** → Detect bright/dark pixel changes in a bounding box to build a flow heatmap.
- **Step3_RemoveNoiseFromHeatMap.py** → Remove noise/rim glare and keep the pouring stream signal.
- **Step4_VolumeEstimationFromHeatMap.py** → Estimate volume from the cleaned heatmap proxy.

##### QuantityDetection-MilkPour
- A similar heatmap-based prototype for milk pouring.

##### ActionDetectionSampleCodeRef
- This folder contains reference implementations and experiments with different models such as TSM, MMAction, and Epic-Kitchens-style pipelines.
- It is useful as a technical reference for available action-recognition approaches.

#### FetchAPIData
- **getProductBarCode.py** → Extract barcode data from product images.
- **USDA_OPEN_FOOD_API.py** → Fetch food nutrients from the USDA API.

### 5. SampleCodeForReference
This folder contains trial-and-error or reference scripts that were used while exploring different models and approaches. It is useful for understanding the evolution of the project but is not the main production path.

---

## 🚀 Recommended Starting Points
1. **PreprocessVideo.py** → Prepare input videos and extract frames.
2. **RunCustomYoloToIdentifyObjects.py** → Detect and label objects in the frames.
3. **SecondPass_GenerateMovementMetadata** → Build object interaction/activity metadata.
4. **ThirdPass_DetectVolume** → Estimate quantity for liquids, powders, and solids.

---

## 🧠 Model Training Notes
- **YOLO training**: Use **ML/ObjectDetection/TrainYolo/TrainYoloOnCustomObjects.py** to train the custom object detector.
- **ResNet training for scoop fill-level estimation**: Use **ML/SampleCodeforReference/ScoopVolume/TrainResnetModelForScoopVol.py** to train the ResNet used for powder-pour inference.
- **Example food-image training**: **ML/SampleCodeforReference/TrainResnetOnCustomObjects.py** shows how to fine-tune a ResNet on custom food images such as coffee, black coffee, and banana.

---

## ⚙️ Requirements
- Python 3.9+
- PyTorch
- Ultralytics YOLO
- OpenCV, NumPy, Pandas
- MySQL (for local DB access)
- Optional: Detectron2, MMAction, or other action-recognition dependencies depending on the experiment path

---

## 📝 Notes
- The repository contains both production-style workflow code and older experimental scripts.
- The main CalEST pipeline is organized around three passes: object labeling, activity metadata generation, and volume estimation.
- The training and inference paths are intentionally separated so that experiment code and production code can evolve independently.
