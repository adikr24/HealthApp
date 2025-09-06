# 🍽️ Food Recognition & Quantity Estimation Project

This repository contains code and data pipelines to **detect food, estimate quantities (coffee/milk pour), and fetch nutritional values** using computer vision and database connections.  

---

## 📂 Project Structure

### **1. DataEngineering Section**
- **PreprocessPhotos.py** → Convert all photos to size **640x640**.  
- **PreprocessVideos.py** → Convert input videos to **mp4 format**, resize, and reduce fps for consistency.  

---

### **2. DBconnection Section**
- **DataModel.sql** → SQL schema containing **basic food data from USDA** (nutritional values).  
- **DBConn.py** → Python connector to interact with the MySQL DB (requires local MySQL setup).  

---

### **3. InputFileProcessor Section**
- **CompressandcutLargeIF.py** → Precursor to `PreprocessPhotos.py`, defines class for photo compression/conversion.  
- ~~**CompressandcutLargeVF.py**~~ → *Deprecated, can be removed.*  
- **piCodeToRecordVideos.py** → Raspberry Pi (tested on **Pi 5 + Pi Camera v3**) script to start/stop video recording.  
- **ProcessVideosToExtractFramesForAnnotations.py** → Extract frames from a video for annotation.  

---

### **4. ML Section**

#### 📌 ActionDetection
##### **QuantityDetection-CoffeePour**
1. **Step1_RunDetectron2.py** → Run Detectron2 to detect objects (cup, spoon, etc.).  
2. **Step1_RunYolo.py** → Run YOLO for additional object detection (post-Detectron2).  
   - Draws bounding box on the cup (default assumption: cup used for coffee pour).  
3. **Step2_DrawHeatMapForCoffee.py** → Detect **bright/dark pixel changes** within bounding boxes → generate heatmaps for flow.  
4. **Step3_RemoveNoiseFromHeatMap.py** → Filter out **rim glare / spurt noise** to prevent double counting. Keeps only **falling stream pixels**.  
5. **Step4_VolumeEstimationFromHeatMap.py** → Estimate volume poured by counting heatmap pixels.  

##### **QuantityDetection-MilkPour**
- Same structure as CoffeePour (Detect → Heatmap → Noise Removal → Volume Estimation).

---

#### 📌 FetchAPIData
- **getProductBarCode.py** → Extract barcode data from product images.  
- **USDA_OPEN_FOOD_API.py** → Fetch **food nutrients from USDA API** using barcode lookup.  

---

#### 📌 ImageDetection
- **TransferLearningResnet.py** → Train a **ResNet** model on custom food images.  
- **InferenceFromResnet.py** → Run inference using trained ResNet checkpoints to predict food labels.  

---

### **5. SampleCodeForReference**
- Contains **trial & error scripts** for experiments.  
- *Not needed for production.*  

---

## 🚀 Pipeline Overview
1. **Data Engineering** → Preprocess images/videos.  
2. **Input Processing** → Record & extract frames.  
3. **ML Detection** → Detect cups/spoons → heatmap pixel flow → clean noise → estimate volume.  
4. **Database/API** → Get nutritional data (basic foods from SQL + packaged foods from USDA API).  
5. **Inference Models** → Use ResNet for food classification.  

---

## ⚙️ Requirements
- Python 3.9+  
- PyTorch, Detectron2, YOLOv8  
- OpenCV, NumPy, Pandas  
- MySQL (for local DB)  
- Check Docker file to replicate environment and results
---
