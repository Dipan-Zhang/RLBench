# read mp4 using cv2 and extract image from it
import cv2
import os
import numpy as np
import argparse  # Note: fixed typo from argparser

def extract_frames(mp4_path, frame_interval=1):
    """
    Extract frames from mp4 video
    
    Args:
        mp4_path: Path to mp4 file
        output_dir: Directory to save extracted images
        frame_interval: Extract every nth frame
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(mp4_path))
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(mp4_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {mp4_path}")
        return
    
    frame_count = 0
    saved_count = 0
    
    while True:
        # Read frame
        ret, frame = cap.read()
        
        # Break if end of video
        if not ret:
            break
        
        # Save frame at specified intervals
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_dir, f"frame_{saved_count:05d}.png")
            cv2.imwrite(output_path, frame)
            saved_count += 1
            
        frame_count += 1
    
    # Release video capture object
    cap.release()
    
    print(f"Extracted {saved_count} frames from {mp4_path} to {output_dir}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract frames from MP4 video')
    parser.add_argument('--path', type=str, default='outputs/PickUpMug/exp_results/ours/trial_2025-04-29_22-26/video_front/obs_cam_overhead/trial_0/video.mp4', 
                        help='Path to MP4 video file')
    # parser.add_argument('--output_dir', type=str, default='extracted_frames',
    #                     help='Directory to save extracted frames')
    parser.add_argument('--interval', type=int, default=1,
                        help='Extract every nth frame')
    
    args = parser.parse_args()
    
    extract_frames(args.path, args.interval)