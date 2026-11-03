# (Optional, kept for reproducibility)
import boto3
import os
import cv2
from dotenv import load_dotenv

load_dotenv()
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
AWS_REGION = os.getenv('AWS_REGION')
BUCKET_NAME = 'apollo-deepfake-data'
VIDEO_FOLDER = 'raw/'
LOCAL_VIDEO_FOLDER = 'downloaded_videos'
FRAME_SAVE_FOLDER = 'extracted_frames'

s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY, region_name=AWS_REGION)
os.makedirs(LOCAL_VIDEO_FOLDER, exist_ok=True)
os.makedirs(FRAME_SAVE_FOLDER, exist_ok=True)

paginator = s3.get_paginator('list_objects_v2')
page_iterator = paginator.paginate(Bucket=BUCKET_NAME, Prefix=VIDEO_FOLDER)

for page in page_iterator:
    if 'Contents' in page:
        for obj in page['Contents']:
            s3_key = obj['Key']
            if s3_key.endswith(('.mp4', '.avi')):
                local_path = os.path.join(LOCAL_VIDEO_FOLDER, os.path.basename(s3_key))
                s3.download_file(BUCKET_NAME, s3_key, local_path)

                cap = cv2.VideoCapture(local_path)
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_path = os.path.join(FRAME_SAVE_FOLDER, f"{os.path.basename(s3_key)}_frame_{frame_count}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frame_count += 1
                cap.release()
                print(f"Extracted {frame_count} frames from {s3_key}")