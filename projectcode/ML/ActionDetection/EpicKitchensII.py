import torch
import torchvision.transforms as transforms
# Make sure your TSM model is accessible here, e.g., from src.models.tsm import TSM
# For demonstration/testing purposes where `src.models.tsm` might not be present,
# a mock TSM class is provided below. In a real environment, remove the mock.
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import csv
import os
import csv
# --- MOCK TSM CLASS FOR DEMONSTRATION/LOCAL TESTING ---
# If you have the actual 'src.models.tsm.TSM' available, you can remove this mock.
class TSM(torch.nn.Module):
    """
    A simplified mock TSM model for demonstration purposes.
    In a real application, you would import the actual TSM model.
    """
    def __init__(self, num_class, num_segments, modality):
        super().__init__()
        # The actual TSM model would have a more complex backbone and temporal modules.
        # This mock uses a simple linear layer to simulate output.
        # The input features to the linear layer must account for all frames in a segment.
        # It should be num_segments * C * H * W
        self.fc = torch.nn.Linear(num_segments * 3 * 224 * 224, num_class) # Corrected input size
        self.num_segments = num_segments
        print(f"Mock TSM initialized with num_class={num_class}, num_segments={num_segments}")

    def forward(self, x):
        # x shape: [batch_size, num_segments, C, H, W]
        # For mock, we flatten all segment frames and pass through linear layer
        batch_size, num_segments, C, H, W = x.shape
        x_flattened = x.view(batch_size, num_segments * C * H * W)
        return self.fc(x_flattened)

# -------------------------------
# Load TSM model
# -------------------------------
def load_tsm_model(checkpoint_path, num_class=397, num_segments=8, modality='RGB', device='cpu'):
    """
    Loads the TSM model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint file.
        num_class (int): Number of action classes.
        num_segments (int): Number of frames per segment for TSM.
        modality (str): Input modality (e.g., 'RGB').
        device (str): Device to load the model on ('cpu' or 'cuda').

    Returns:
        torch.nn.Module: Loaded TSM model.
    """
    # Instantiate the TSM model. Ensure 'TSM' is imported correctly if not using mock.
    # num_segments is passed here to ensure the mock TSM initializes correctly.
    model = TSM(num_class=num_class, num_segments=num_segments, modality=modality)
    
    # Load the checkpoint and apply state_dict.
    # Strict=False is often used to handle slight mismatches (e.g., 'module.' prefix).
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt) # Get 'state_dict' if it's nested
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    print(f"Model loaded successfully from {checkpoint_path}")
    
    model.eval() # Set model to evaluation mode (disables dropout, batch norm updates)
    model.to(device) # Move model to the specified device
    return model

# -------------------------------
# Load action classes
# -------------------------------
def load_action_classes(csv_path):
    """
    Loads action class mappings from a CSV file.

    Args:
        csv_path (str): Path to the CSV file containing action IDs and names.

    Returns:
        dict: A dictionary mapping action IDs to action names.
    """
    df = pd.read_csv(csv_path)
    # Handle cases where 'id' might be stored as a string representation of a list
    df["id"] = df["id"].apply(lambda x: int(eval(x)) if isinstance(x, str) and x.startswith("[") else int(x))
    action_map = dict(zip(df["id"], df["action"]))
    print(f"Loaded {len(action_map)} action classes from {csv_path}")
    return action_map

# -------------------------------
# Prepare TSM segments from 64-frame compartments
# This replaces the original sample_frames and chunk_frames logic for the new strategy.
# -------------------------------
def prepare_tsm_segments_from_compartments(video_path, compartment_size=64, tsm_segment_length=8, size=224):
    """
    Prepares TSM segments by sampling tsm_segment_length frames from each
    compartment_size-frame window of the video.

    Args:
        video_path (str): Path to the video file.
        compartment_size (int): The size of the video compartment (e.g., 64 frames).
        tsm_segment_length (int): The number of frames required for one TSM segment (e.g., 8 frames).
        size (int): Desired width and height for resized frames.

    Returns:
        tuple: A tuple containing:
            - tsm_segments (list): List of PyTorch tensors, each representing one TSM segment.
                                   Shape: [1, tsm_segment_length, C, H, W].
            - compartment_start_indices (list): List of original video frame indices
                                                 marking the start of each 64-frame compartment.
            - total_frames_in_video (int): Total number of frames in the original video.
            - video_fps (float): Original FPS of the video.
    """
    # --- MOCK VIDEO CAPTURE FOR DEMONSTRATION ---
    # This block simulates video capture if OpenCV is not fully functional or video path is a dummy.
    if "P03_25.MP4" in video_path and not cv2.__version__: # Basic check if cv2 might not be fully functional
        print(f"Mocking video capture for {video_path} as OpenCV might not be fully set up or video path is a dummy.")
        # Simulating a 37-second video at 60 FPS
        total_frames_in_video = 37 * 60 # 2220 frames
        video_fps = 60.0 
        dummy_frame_np = np.zeros((size, size, 3), dtype=np.uint8) # Create a black dummy frame
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        tsm_segments = []
        compartment_start_indices = []

        for comp_start_idx in range(0, total_frames_in_video, compartment_size):
            current_comp_frames = []
            comp_end_idx = min(comp_start_idx + compartment_size, total_frames_in_video)
            
            # Sample tsm_segment_length frames evenly from this compartment
            if comp_end_idx > comp_start_idx: # Ensure compartment is not empty
                sampled_indices_in_comp = np.linspace(comp_start_idx, comp_end_idx - 1, tsm_segment_length, dtype=int)
            else:
                sampled_indices_in_comp = [] # Empty compartment, no indices

            for _ in range(tsm_segment_length): # Just create dummy frames for the segment length
                current_comp_frames.append(preprocess(dummy_frame_np))
            
            tsm_segments.append(torch.stack(current_comp_frames).unsqueeze(0))
            compartment_start_indices.append(comp_start_idx)
            
        print(f"Mocked {len(tsm_segments)} TSM segments from {compartment_size}-frame compartments.")
        return tsm_segments, compartment_start_indices, total_frames_in_video, video_fps
    # --- END MOCK VIDEO CAPTURE ---

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {video_path}. Please check the path and ensure OpenCV can read it.")

    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    tsm_segments = [] # List to store the 8-frame PyTorch segments for TSM
    compartment_start_indices = [] # List to store the starting frame index of each 64-frame compartment

    # Iterate through the video in 64-frame compartments
    for comp_start_idx in range(0, total_frames_in_video, compartment_size):
        comp_end_idx = min(comp_start_idx + compartment_size, total_frames_in_video)
        
        if comp_end_idx <= comp_start_idx:
            continue # Skip if the compartment is empty or invalid (e.g., at very end of video)

        current_tsm_segment_frames = []
        # Calculate indices for even sampling of `tsm_segment_length` frames from this compartment
        # Use a small epsilon to ensure `np.linspace` end is inclusive for remaining frames if `comp_end_idx` is `total_frames_in_video`
        sampled_indices_in_comp = np.linspace(comp_start_idx, comp_end_idx - 1, tsm_segment_length, dtype=int)
        
        for frame_idx_to_read in sampled_indices_in_comp:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_to_read)
            ret, frame = cap.read()
            if not ret:
                # If a frame can't be read, pad with a dummy frame or duplicate the last valid frame
                # For simplicity, using a zero tensor as a dummy. In real scenarios, more robust padding might be needed.
                print(f"Warning: Could not read frame {frame_idx_to_read}. Padding with a black frame.")
                current_tsm_segment_frames.append(torch.zeros(3, size, size))
                continue
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (size, size))
            frame = preprocess(frame)
            current_tsm_segment_frames.append(frame)
        
        # Ensure the segment always has `tsm_segment_length` frames, padding if necessary
        while len(current_tsm_segment_frames) < tsm_segment_length:
            # Pad with the last collected frame or a zero tensor if no frames were collected
            current_tsm_segment_frames.append(current_tsm_segment_frames[-1] if current_tsm_segment_frames else torch.zeros(3, size, size))

        # Stack the collected frames and add a batch dimension for TSM input
        tsm_segments.append(torch.stack(current_tsm_segment_frames).unsqueeze(0)) # Result: [1, tsm_segment_length, 3, H, W]
        compartment_start_indices.append(comp_start_idx) # Store the starting frame index of this 64-frame compartment

    cap.release()
    print(f"Prepared {len(tsm_segments)} TSM segments from {compartment_size}-frame compartments.")
    return tsm_segments, compartment_start_indices, total_frames_in_video, video_fps

# -------------------------------
# Run full inference - MODIFIED FOR PER-COMPARTMENT PREDICTION
# -------------------------------
def run_inference(video_path, checkpoint_path, action_csv, device='cpu', 
                  compartment_size=64, tsm_segment_length=8):
    """
    Runs TSM inference on a video and provides predictions for each 64-frame compartment.

    Args:
        video_path (str): Path to the input video.
        checkpoint_path (str): Path to the TSM model checkpoint.
        action_csv (str): Path to the CSV mapping action IDs to names.
        device (str): Device to run inference on ('cpu' or 'cuda').
        compartment_size (int): The size of the video compartment for reporting predictions (e.g., 64 frames).
        tsm_segment_length (int): The number of frames TSM model expects per segment (e.g., 8 frames).

    Returns:
        tuple: A tuple containing:
            - per_compartment_predictions (dict): A dictionary where keys are original
                                                   video frame indices (start of compartment)
                                                   and values are a list of (action_name, probability)
                                                   tuples for the top 5 predictions of that compartment.
            - overall_top5 (list): List of (action_name, probability) tuples
                                   for the top 5 overall actions across the video.
    """
    action_map = load_action_classes(action_csv)
    num_classes = len(action_map)

    # Load the TSM model with its actual num_segments (which is tsm_segment_length, 8)
    model = load_tsm_model(checkpoint_path, num_class=num_classes, 
                           num_segments=tsm_segment_length, modality='RGB', device=device)
    
    # Prepare TSM segments directly from compartments
    tsm_segments, compartment_start_indices, total_frames_in_video, video_fps = \
        prepare_tsm_segments_from_compartments(video_path, compartment_size=compartment_size, 
                                               tsm_segment_length=tsm_segment_length)
    
    # Perform inference for each TSM segment
    per_segment_probs = [] # Stores the probability distribution for each 8-frame TSM segment
    for segment in tsm_segments:
        segment = segment.to(device) # Move segment to the specified device
        with torch.no_grad(): # Disable gradient calculation for inference
            outputs = model(segment)
            # Apply softmax to get probabilities and squeeze to remove batch dimension
            probs = torch.nn.functional.softmax(outputs, dim=1).squeeze().cpu() 
            per_segment_probs.append(probs)

    # -------------------------------
    # Per-compartment top-5 predictions
    # Each TSM segment's prediction is considered the prediction for its corresponding 64-frame compartment.
    # -------------------------------
    per_compartment_predictions = {} # Dictionary to store compartment_start_frame -> list of (action, probability)
    
    # Iterate through the probabilities of each TSM segment and its corresponding 64-frame compartment's start index
    for i, prob_tensor in enumerate(per_segment_probs):
        comp_start_idx = compartment_start_indices[i] # This is the original video frame index for the start of the 64-frame compartment
        
        # Get the **top 5** predictions for this particular segment/compartment
        k_top5_comp = min(5, prob_tensor.numel()) # Ensure k is not greater than number of classes
        top5_idx_comp = torch.topk(prob_tensor, k_top5_comp).indices.tolist() # Get indices of top 5 classes
        
        # Create a list of (action_name, probability) tuples for the top 5
        top5_actions_comp = []
        for action_id in top5_idx_comp:
            action_name = action_map.get(action_id, f"Unknown Action ({action_id})")
            probability = prob_tensor[action_id].item()
            top5_actions_comp.append((action_name, probability))

        # Assign this segment's top-5 prediction list to its corresponding 64-frame compartment
        per_compartment_predictions[comp_start_idx] = top5_actions_comp

    # -------------------------------
    # Overall top-5 prediction (averaged across all TSM segment probabilities)
    # -------------------------------
    if per_segment_probs: # Ensure there are probabilities to average
        # Stack all segment probabilities and average them to get an overall video probability distribution
        all_probs = torch.stack(per_segment_probs).mean(dim=0).squeeze()
    else:
        # If no segments were processed, return zero probabilities (or handle as an error)
        all_probs = torch.zeros(num_classes)
        print("Warning: No TSM segments processed, overall probabilities are zero.")

    # Get the top 5 actions from the overall probability distribution
    k_overall = min(5, all_probs.numel()) # Ensure k is not greater than number of classes
    overall_top5_idx = torch.topk(all_probs, k_overall).indices.tolist()
    overall_top5 = [(action_map.get(i, f"Unknown Action ({i})"), all_probs[i].item()) for i in overall_top5_idx]

    # -------------------------------
    # Print results
    # -------------------------------
    print(f"\n⏱ Per-{compartment_size}-frame compartment top-5 predictions (compartment start frame → prediction):")
    # Sort predictions by compartment start frame index for sequential output
    for comp_idx in sorted(per_compartment_predictions.keys()):
        actions_with_probs = per_compartment_predictions[comp_idx]
        print(f"Compartment starting Frame {comp_idx:05d} → {actions_with_probs}")

    print(f"\n🎯 Overall top-5 predictions for video: {overall_top5}")
    return per_compartment_predictions, overall_top5

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # Determine the device (GPU if available, otherwise CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Define paths to your video, model checkpoint, and action class CSV
    # NOTE: You must replace these with your actual file paths for the code to run.
    # The current paths are placeholders from the original request.
#    video_path = "/app/mediaFiles/videos/labeledvideos/P03_25.MP4"
    video_path = "/app/mediaFiles/videos/labeledvideos/pouringMilkandsugar_trimmed.mp4"
    checkpoint_path = "/opt/epic-kitchens/C1-Action-Recognition-TSN-TRN-TSM/models/tsm_rgb.ckpt"
    action_csv = "/app/mediaFiles/videos/kinetic_400_label/tsm_397_actions.csv"
    output_dir = '/app/mediaFiles/videos/PredictedActions'

    # Define the compartment size (e.g., 64 frames for reporting a single prediction)
    # and the TSM segment length (number of frames TSM expects, typically 8).
    COMPARTMENT_SIZE = 64
    TSM_SEGMENT_LENGTH = 8 

    # Run the inference with the new compartment-based logic
    per_compartment_preds, overall_top5 = run_inference(
        video_path, 
        checkpoint_path, 
        action_csv, 
        device=device,
        compartment_size=COMPARTMENT_SIZE,
        tsm_segment_length=TSM_SEGMENT_LENGTH
    )

    print("\n--- Inference Complete ---")
    print(f"Total per-compartment predictions generated: {len(per_compartment_preds)}")

    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define file paths
    per_compartment_csv_file = os.path.join(output_dir, 'per_compartment_preds.csv')
    overall_top5_csv_file = os.path.join(output_dir, 'overall_top5.csv')

    type(per_compartment_preds)

        # Define the file path for the CSV
    output_csv_file = os.path.join(output_dir, 'per_compartment_preds.csv')

    # Convert the dictionary data to a list of rows
    csv_rows = []
    for frame, predictions in per_compartment_preds.items():
        for action, confidence in predictions:
            csv_rows.append({'Frame': frame, 'Action': action, 'Confidence': confidence})

    # Write the data to the CSV file
    with open(output_csv_file, 'w', newline='') as f:
        fieldnames = ['Frame', 'Action', 'Confidence']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Prediction results saved to {output_csv_file}")
    
    # -------------------------------
    # NEW CODE ADDITION START HERE
    # -------------------------------
    
    # Define the file path for the overall top 5 CSV
    overall_output_csv_file = os.path.join(output_dir, 'overall_top5.csv')
    
    # Write the overall top 5 data to the CSV file
    with open(overall_output_csv_file, 'w', newline='') as f:
        fieldnames = ['Action', 'Confidence']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # overall_top5 is already a list of tuples: [(action_name, confidence), ...]
        for action, confidence in overall_top5:
            writer.writerow({'Action': action, 'Confidence': confidence})

    print(f"Overall top 5 predictions saved to {overall_output_csv_file}")
    
    # -------------------------------
    # NEW CODE ADDITION END HERE
    # -------------------------------