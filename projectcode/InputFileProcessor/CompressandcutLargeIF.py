import os
from PIL import Image
from pillow_heif import register_heif_opener # type: ignore

register_heif_opener() # Enables .heic support in Pillow

class ImageResizer:
    def __init__(self,
                 input_dir: str,
                 output_dir: str,
                 target_size: tuple = (640, 640),
                 keep_aspect_ratio: bool = True,
                 image_quality: int = 85):
        """
        Resize and convert HEIC/JPG/PNG images to JPEG.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_size = target_size
        self.keep_aspect_ratio = keep_aspect_ratio
        self.image_quality = image_quality

        os.makedirs(self.output_dir, exist_ok=True)

    def is_image_file(self, filename):
        return filename.lower().endswith(('.jpg', '.jpeg', '.png', '.heic'))

    def resize_image(self, input_path, output_path):
        img = Image.open(input_path).convert("RGB")

        if self.keep_aspect_ratio:
            img.thumbnail(self.target_size, Image.Resampling.LANCZOS)
        else:
            img = img.resize(self.target_size, Image.Resampling.LANCZOS)

        img.save(output_path, "JPEG", quality=self.image_quality, optimize=True)

    def process_all(self):
        print(f"🔄 Processing images in: {self.input_dir}")
        
        # Use os.walk to find all files in the directory tree
        for root, _, files in os.walk(self.input_dir):
            for filename in files:
                if not self.is_image_file(filename):
                    continue

                input_path = os.path.join(root, filename)
                
                # Create the corresponding output directory structure
                relative_path = os.path.relpath(root, self.input_dir)
                output_folder = os.path.join(self.output_dir, relative_path)
                os.makedirs(output_folder, exist_ok=True)
                
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_folder, base_name + ".jpg")

                try:
                    self.resize_image(input_path, output_path)
                    print(f"✅ {input_path} → {output_path}")
                except Exception as e:
                    print(f"❌ Error processing {input_path}: {e}")

        print("\n🎉 All images resized and saved to:", self.output_dir)