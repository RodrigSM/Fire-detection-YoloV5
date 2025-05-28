import cv2
import torch
import numpy as np
import argparse
import sys
from pathlib import Path
import exifread
from datetime import datetime
import os
import geocoder
from functools import lru_cache
import time
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import socket
import requests
import subprocess
import re
import webbrowser

# Add YOLOv5 root directory to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.datasets import LoadImages

def get_accurate_gps():
    """
    Get accurate GPS coordinates using multiple methods
    Returns: tuple of (latitude, longitude, address) or None if not available
    """
    try:
        # Try multiple methods to get accurate location
        methods = [
            # Method 1: Using geocoder with IP
            lambda: geocoder.ip('me'),
            # Method 2: Using Nominatim geocoder
            lambda: Nominatim(user_agent="fire_detection").geocode(socket.gethostname()),
            # Method 3: Using a public IP geolocation API
            lambda: requests.get('https://ipapi.co/json/').json()
        ]
        
        for method in methods:
            try:
                result = method()
                if result:
                    if isinstance(result, dict):  # IP API response
                        lat, lon = result.get('latitude'), result.get('longitude')
                        if lat and lon:
                            # Get address from coordinates
                            geolocator = Nominatim(user_agent="fire_detection")
                            location = geolocator.reverse(f"{lat}, {lon}")
                            return lat, lon, location.address if location else "Unknown location"
                    else:  # geocoder or Nominatim response
                        if hasattr(result, 'lat') and hasattr(result, 'lng'):
                            lat, lon = result.lat, result.lng
                            # Get address from coordinates
                            geolocator = Nominatim(user_agent="fire_detection")
                            location = geolocator.reverse(f"{lat}, {lon}")
                            return lat, lon, location.address if location else "Unknown location"
            except Exception:
                continue
                
    except Exception as e:
        print(f"Error getting accurate GPS coordinates: {e}")
    return None

@lru_cache(maxsize=1)
def get_current_gps():
    """
    Get current GPS coordinates with caching
    Returns: tuple of (latitude, longitude, address) or None if not available
    """
    return get_accurate_gps()

def get_gps_from_image(image_path):
    """
    Extract GPS coordinates from image metadata if available
    Returns: tuple of (latitude, longitude, address) or None if not available
    """
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f)
            
        if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
            lat = tags['GPS GPSLatitude'].values
            lon = tags['GPS GPSLongitude'].values
            
            # Convert to decimal degrees
            lat = float(lat[0]) + float(lat[1])/60 + float(lat[2])/3600
            lon = float(lon[0]) + float(lon[1])/60 + float(lon[2])/3600
            
            # Check for South/West
            if tags['GPS GPSLatitudeRef'].values == 'S':
                lat = -lat
            if tags['GPS GPSLongitudeRef'].values == 'W':
                lon = -lon
            
            # Get address from coordinates
            geolocator = Nominatim(user_agent="fire_detection")
            location = geolocator.reverse(f"{lat}, {lon}")
            address = location.address if location else "Unknown location"
                
            return lat, lon, address
    except Exception as e:
        print(f"Error reading GPS from image: {e}")
    return None

def analyze_fire_severity(detections, img_shape, source_path=None):
    """
    Analyze fire severity based on detection size and confidence
    Returns: severity level, message, and GPS coordinates
    """
    if len(detections) == 0:
        return "No fire detected", "Safe", None, None, None
    
    # Initialize variables
    max_area = 0
    max_conf = 0
    fire_location = None
    detected_fire = False
    
    # Process detections
    for *xyxy, conf, cls in detections:
        cls_id = int(cls)
        area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
        
        # Check for any type of fire (classes 1,2,3,4)
        if cls_id in [1, 2, 3, 4]:  # safe fire, warning, danger, or smoke
            detected_fire = True
            
        if area > max_area:
            max_area = area
            max_conf = conf
            center_x = (xyxy[0] + xyxy[2]) / 2
            center_y = (xyxy[1] + xyxy[3]) / 2
            fire_location = (center_x, center_y)
    
    # Get GPS coordinates if any type of fire is detected
    gps_coords = None
    address = None
    if detected_fire:
        if source_path and os.path.isfile(source_path):
            result = get_gps_from_image(source_path)
            if result:
                gps_coords = (result[0], result[1])
                address = result[2]
        else:
            result = get_current_gps()
            if result:
                gps_coords = (result[0], result[1])
                address = result[2]
    
    # Calculate relative area (percentage of image)
    img_area = img_shape[0] * img_shape[1]
    relative_area = (max_area / img_area) * 100
    
    # Determine severity based on area and confidence
    if relative_area < 5 or max_conf < 0.5:
        return "Safe", "Low risk - Small fire detected", gps_coords, fire_location, address
    elif relative_area < 15 or max_conf < 0.7:
        return "Warning", "Medium risk - Growing fire detected", gps_coords, fire_location, address
    else:
        return "Danger", "High risk - Large fire detected!", gps_coords, fire_location, address

def print_iriun_setup_guide():
    """Print setup guide for Iriun Webcam"""
    print("\n=== Iriun Webcam Setup Guide ===")
    print("1. Install Iriun Webcam:")
    print("   - Android: https://play.google.com/store/apps/details?id=com.iriun.webcam")
    print("   - iOS: https://apps.apple.com/app/iriun-webcam/id1114420056")
    print("\n2. On your phone:")
    print("   - Open Iriun Webcam app")
    print("   - Make sure your phone and computer are on the same WiFi")
    print("   - Enable the camera in the app")
    print("   - Note the IP address shown in the app")
    print("\n3. On your computer:")
    print("   - Run this script")
    print("   - The camera should connect automatically")
    print("\nIf you need help, visit: https://iriun.com/")
    print("===============================\n")

def find_iriun_camera():
    """
    Try to find Iriun camera on the local network
    Returns: camera URL or None if not found
    """
    try:
        # Common Iriun ports
        ports = [8080, 8081, 8082, 8083]
        
        # Get local IP address
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        base_ip = '.'.join(local_ip.split('.')[:-1])  # Get first three octets
        
        print("\nSearching for Iriun camera...")
        
        # Try to find Iriun camera
        for port in ports:
            url = f"http://{base_ip}.1:{port}"
            print(f"Trying {url}...")
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    cap.release()
                    print(f"✓ Found Iriun camera at: {url}")
                    return url
            cap.release()
        
        print("\n❌ Iriun camera not found automatically.")
        print("Please check:")
        print("1. Is Iriun Webcam app running on your phone?")
        print("2. Are your phone and computer on the same WiFi?")
        print("3. Is the camera enabled in the Iriun app?")
        print("\nWould you like to:")
        print("1. Try again")
        print("2. View setup guide")
        print("3. Use local webcam instead")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        if choice == "1":
            return find_iriun_camera()
        elif choice == "2":
            print_iriun_setup_guide()
            return find_iriun_camera()
        else:
            return None
            
    except Exception as e:
        print(f"Error finding Iriun camera: {e}")
        return None

def run_detection(weights='runs/train/exp25/weights/best.pt', source=0, img_size=640, conf_thres=0.25, save_dir=None):
    # Initialize
    device = select_device('')
    model = DetectMultiBackend(weights, device=device)
    stride = model.stride
    img_size = check_img_size(img_size, s=stride)
    
    # Load model
    model.warmup()
    
    # Handle camera source
    if source == '0' or source == 0:
        print("\n=== Camera Setup ===")
        print("1. Use Iriun camera (phone)")
        print("2. Use local webcam")
        choice = input("Enter your choice (1-2): ").strip()
        
        if choice == "1":
            # Try to find Iriun camera
            iriun_url = find_iriun_camera()
            if iriun_url:
                source = iriun_url
                print("\n✓ Successfully connected to Iriun camera!")
            else:
                source = 0
                print("\nUsing local webcam instead.")
        else:
            source = 0
            print("\nUsing local webcam.")
    
    # Initialize video capture
    cap = None
    if isinstance(source, (int, str)):
        if str(source).isdigit():
            # Local webcam
            cap = cv2.VideoCapture(int(source))
            # Set lower resolution for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        elif source.startswith(('http://', 'https://', 'rtsp://')):
            # IP camera (including Iriun)
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"\n❌ Error: Could not connect to camera at {source}")
                print("\nTroubleshooting steps:")
                print("1. Check if your phone and computer are on the same WiFi network")
                print("2. Make sure Iriun Webcam app is running on your phone")
                print("3. Verify the camera is enabled in the Iriun app")
                print("4. Try restarting the Iriun app")
                print("\nWould you like to:")
                print("1. View setup guide")
                print("2. Try again")
                print("3. Use local webcam instead")
                
                choice = input("\nEnter your choice (1-3): ").strip()
                if choice == "1":
                    print_iriun_setup_guide()
                elif choice == "2":
                    return run_detection(weights, source, img_size, conf_thres, save_dir)
                else:
                    source = 0
                    cap = cv2.VideoCapture(0)
                return
    
    # Dataloader
    dataset = LoadImages(source, img_size=img_size, stride=stride)
    
    # Initialize FPS calculation variables
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    print("\n=== Fire Detection Started ===")
    print("Press 'q' to quit")
    print("=============================\n")
    
    # Run inference
    for path, im, im0s, vid_cap, s in dataset:
        # Calculate FPS
        frame_count += 1
        if frame_count >= 30:  # Update FPS every 30 frames
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()
        
        im = torch.from_numpy(im).to(device)
        im = im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        
        # Inference
        pred = model(im)
        
        # NMS
        pred = non_max_suppression(pred, conf_thres, 0.45, classes=None, max_det=1000)
        
        # Process predictions
        for i, det in enumerate(pred):
            im0 = im0s.copy()
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                
                # Analyze fire severity
                severity, message, gps_coords, fire_location, address = analyze_fire_severity(det, im0.shape, path)
                
                # Add severity text
                cv2.putText(im0, f"Status: {severity}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(im0, message, (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Add FPS counter
                cv2.putText(im0, f"FPS: {fps:.1f}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add GPS coordinates if available
                if gps_coords:
                    lat, lon = gps_coords
                    gps_text = f"GPS: {lat:.6f}, {lon:.6f}"
                    cv2.putText(im0, gps_text, (10, 150),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Create Google Maps link
                    maps_link = f"https://www.google.com/maps?q={lat},{lon}"
                    print(f"\nFire detected! View location on Google Maps: {maps_link}")
                    if address:
                        print(f"Address: {address}")
                
                # Add fire location in image coordinates
                if fire_location:
                    x, y = fire_location
                    cv2.circle(im0, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv2.putText(im0, f"Fire at: ({int(x)}, {int(y)})", (10, 190),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{model.names[c]} {conf:.2f}'
                    annotator = Annotator(im0, line_width=3, example=str(model.names))
                    annotator.box_label(xyxy, label, color=colors(c, True))
            
            # Save results if save_dir is specified
            if save_dir:
                if isinstance(source, (int, str)) and str(source).isdigit() or source.endswith('.mp4'):  # Video
                    if not hasattr(run_detection, 'video_writer'):
                        run_detection.video_writer = cv2.VideoWriter(
                            f'{save_dir}/output.mp4',
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            vid_cap.get(cv2.CAP_PROP_FPS),
                            (im0.shape[1], im0.shape[0])
                        )
                    run_detection.video_writer.write(im0)
                else:  # Image
                    cv2.imwrite(f'{save_dir}/output.jpg', im0)
            
            # Display the image
            cv2.imshow('Fire Detection', im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                break
    
    cv2.destroyAllWindows()
    if save_dir and hasattr(run_detection, 'video_writer'):
        run_detection.video_writer.release()
    if cap is not None:
        cap.release()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/train/exp25/weights/best.pt', help='model path')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam, http:// for IP camera')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--save-dir', type=str, default=None, help='directory to save results')
    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_opt()
    run_detection(**vars(opt)) 