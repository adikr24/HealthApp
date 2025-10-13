from projectcode.InputFileProcessor.CompressandcutLargeIF import ImageResizer
import os


input_dir =  "/app/mediaFiles/images/inputimages/InputImagesForYoloPrep/YoloLabeling/ProteinShakeLabeling_II/OriginalImages"
output_dir = "/app/mediaFiles/images/inputimages/InputImagesForYoloPrep/YoloLabeling/ProteinShakeLabeling_II/CompressedImages"

resizer = ImageResizer(
    input_dir= input_dir,
    output_dir= output_dir,
    target_size=(640, 640),
    keep_aspect_ratio=True,
    image_quality=80
)

resizer.process_all()

#resizer.get_message()

