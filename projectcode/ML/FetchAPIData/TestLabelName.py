import easyocr

# Path to a single image
image_path = '/app/mediaFiles/output/videoOutputs/DetectedFrames/coffee_volume_output/annotated_frames/frame_00074.jpg'

# Use GPU if available
reader = easyocr.Reader(['en'], gpu=True)

# Run OCR
results = reader.readtext(image_path, detail=False)   # returns list of strings
text = " ".join(results)

print(text)


