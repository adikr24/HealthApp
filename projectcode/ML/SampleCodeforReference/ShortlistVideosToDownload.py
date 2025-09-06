import pandas as pd
import numpy as np

# -------------------------------
# Paths to CSVs
# -------------------------------
video_info_csv = "/app/mediaFiles/videos/kinetic_400_label/EPIC_100_video_info.csv"
validation_csv = "/app/mediaFiles/videos/kinetic_400_label/EPIC_100_validation.csv"
tsm_action_csv = "/app/mediaFiles/videos/kinetic_400_label/tsm_397_actions.csv"

# -------------------------------
# Load CSVs
# -------------------------------
video_info = pd.read_csv(video_info_csv)  # video_id,duration,fps,resolution
validation = pd.read_csv(validation_csv)  # narration_id,participant_id,video_id,start_frame,stop_frame,verb_class,noun_class,...

tsm_actions = pd.read_csv(tsm_action_csv)  # id, action

# -------------------------------
# Count top actions in validation set
# -------------------------------
# Merge verb_class + noun_class to get full action
validation['action'] = validation['verb'] + " " + validation['noun']

# Count frequency of actions
action_counts = validation['action'].value_counts().reset_index()
action_counts.columns = ['action', 'count']

# Pick top 100 actions (or more kitchen related if desired)
top_actions = action_counts.head(100)['action'].tolist()

# -------------------------------
# Filter validation rows to only top actions
# -------------------------------
top_validation_clips = validation[validation['action'].isin(top_actions)].copy()

# Merge with video info to get fps & duration
top_validation_clips = top_validation_clips.merge(video_info, on='video_id', how='left')

# -------------------------------
# Suggest clips to download (shorter clips first)
# -------------------------------
# Calculate clip length in seconds
top_validation_clips['clip_duration'] = top_validation_clips['stop_timestamp'].apply(lambda x: pd.to_timedelta(x).total_seconds()) - \
                                       top_validation_clips['start_timestamp'].apply(lambda x: pd.to_timedelta(x).total_seconds())

# Sort by clip duration ascending (shorter clips easier to process)
top_validation_clips = top_validation_clips.sort_values('clip_duration')

# Optional: filter clips less than 30 seconds
top_validation_clips = top_validation_clips[top_validation_clips['clip_duration'] <= 30]

# -------------------------------
# Generate suggested download info + FFmpeg commands
# -------------------------------
download_suggestions = []
for idx, row in top_validation_clips.iterrows():
    video_file = f"{row['video_id']}.mp4"
    start_time = row['start_timestamp']
    stop_time = row['stop_timestamp']
    clip_file = f"{row['narration_id']}.mp4"
    
    ffmpeg_cmd = f"ffmpeg -i {video_file} -ss {start_time} -to {stop_time} -c copy {clip_file}"
    
    download_suggestions.append({
        'narration_id': row['narration_id'],
        'video_id': row['video_id'],
        'action': row['action'],
        'clip_duration': row['clip_duration'],
        'download_file': video_file,
        'clip_file': clip_file,
        'ffmpeg_cmd': ffmpeg_cmd
    })

suggestions_df = pd.DataFrame(download_suggestions)

# Save suggestions to CSV
suggestions_df.to_csv("/app/mediaFiles/videos/top_validation_clips_suggestions.csv", index=False)

# -------------------------------
# Print top 10 suggestions as example
# -------------------------------
print(suggestions_df.head(10))
