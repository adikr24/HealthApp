# Use your base image with PyTorch and SlowFast already installed
FROM projectone:latest

# Switch to the root user to install system-level packages
USER root

# Install git and other necessary system dependencies in a single layer
# to improve build efficiency and caching.
RUN apt-get update && apt-get install -y git

# Now, switch back to your non-root user for security best practices.
USER appuser

# Set the working directory where the EPIC-Kitchens repo will be cloned.
# This location is outside the /app mount to prevent it from being hidden.
# --- HIGHLIGHTED CHANGE ---
WORKDIR /opt/epic-kitchens

# Clone the EPIC-Kitchens repository. The --depth 1 flag reduces image size.
RUN git clone --depth 1 https://github.com/epic-kitchens/C1-Action-Recognition-TSN-TRN-TSM.git

# Set the PYTHONPATH to include the newly cloned repository, making its modules importable.
# The 'src' folder contains the core Python modules (e.g., models, ops, datasets).
# --- HIGHLIGHTED CHANGE ---
ENV PYTHONPATH=/opt/epic-kitchens/C1-Action-Recognition-TSN-TRN-TSM/src:$PYTHONPATH
# Also add the repository root to PYTHONPATH for broader module discovery.
# --- HIGHLIGHTED CHANGE ---
ENV PYTHONPATH=/opt/epic-kitchens/C1-Action-Recognition-TSN-TRN-TSM:$PYTHONPATH

# Change into the models directory within the cloned repo to download checkpoint files.
# --- HIGHLIGHTED CHANGE ---
WORKDIR /opt/epic-kitchens/C1-Action-Recognition-TSN-TRN-TSM/models
RUN bash download_models.sh

# Move back to the /app directory, which is where your local project is mounted.
# This ensures that your VS Code workspace (your local files) is the primary working directory.
# --- HIGHLIGHTED CHANGE ---
WORKDIR /app

# Set up other directories your code needs within your mounted workspace.
RUN mkdir -p /app/mediaFiles/dataset_root/pouringmilk && \
    mkdir -p /app/mediaFiles/images/labeledimages

# Switch to the root user to install system-level packages
USER root
RUN apt-get update && apt-get install -y \
	libzbar0 \
    libzbar-dev && \
    rm -rf /var/lib/apt/lists/*

# Now, switch back to your non-root user for security best practices.
USER appuser

# Install pyzbar (into appuser’s environment)
RUN pip install --user --no-cache-dir pyzbar

##########################################################################
# --- Add OCR tools (Tesseract, EasyOCR, pytesseract) --------------------
##########################################################################

# Switch back to root to install Tesseract + deps
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev && \
    rm -rf /var/lib/apt/lists/*

# Optional: tell pytesseract where trained data lives
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# Back to non-root user for Python packages
USER appuser
WORKDIR /home/appuser

RUN pip install --user --no-cache-dir \
    easyocr \
    pytesseract \
    pyclipper \
    shapely
	
RUN pip install --user --no-cache-dir supervision
	
# Launch into bash by default to provide an interactive terminal.
CMD ["bash"]