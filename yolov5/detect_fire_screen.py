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
from fractions import Fraction
from mss import mss

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

def get_accurate_gps(timeout: int = 5):
    """
    Attempt to obtain accurate GPS coordinates via multiple methods.
    Args:
        timeout (int): seconds before each request times out.
    Returns:
        tuple: (latitude, longitude, address) or None if all methods fail.
    """
    methods = [
        ('IP Geolocation (ipapi)', lambda: requests.get('https://ipapi.co/json/', timeout=timeout).json()),
        ('IP Geocoder library', lambda: __import__('geocoder').ip('me')),
        ('Local Hostname Geocode', lambda: Nominatim(user_agent="gps_extractor_app", timeout=timeout)
            .geocode(socket.gethostname()))
    ]

    for name, method in methods:
        try:
            print(f"Trying GPS method: {name}")
            result = method()

            if not result:
                continue

            # Parse dict result
            if isinstance(result, dict):
                lat = result.get('latitude') or result.get('lat')
                lon = result.get('longitude') or result.get('lon')
            else:
                lat = getattr(result, 'lat', None)
                lon = getattr(result, 'lng', None)

            if lat is None or lon is None:
                continue

            # Reverse geocode for address
            geolocator = Nominatim(user_agent="gps_extractor_app", timeout=timeout)
            location = geolocator.reverse((lat, lon), language='en')
            address = location.address if location else None

            print(f"Accurate GPS -> lat={lat}, lon={lon}, addr={address}")
            return float(lat), float(lon), address

        except Exception as e:
            print(f"Method '{name}' failed: {e}")
            time.sleep(0.5)
            continue

    print("All GPS methods failed.")
    return None
@lru_cache(maxsize=1)
def get_current_gps():
    """
    Get current GPS coordinates with caching
    Returns: tuple of (latitude, longitude, address) or None if not available
    """
    return get_accurate_gps()

def _convert_to_degrees(value):
    """
    Helper to convert the exifread GPS rational values to float degrees.
    """
    d = Fraction(value[0].num, value[0].den)
    m = Fraction(value[1].num, value[1].den)
    s = Fraction(value[2].num, value[2].den)
    return float(d + (m / 60) + (s / 3600))


def get_gps_from_image(image_path):
    """
    Extract GPS coordinates from image metadata if available.
    Returns:
        tuple: (latitude, longitude, address) or None if not available
    """
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)

        # Check presence of required GPS tags
        gps_lat_key = 'GPS GPSLatitude'
        gps_lat_ref_key = 'GPS GPSLatitudeRef'
        gps_lon_key = 'GPS GPSLongitude'
        gps_lon_ref_key = 'GPS GPSLongitudeRef'

        if all(key in tags for key in [gps_lat_key, gps_lat_ref_key, gps_lon_key, gps_lon_ref_key]):
            lat_values = tags[gps_lat_key].values
            lon_values = tags[gps_lon_key].values
            lat = _convert_to_degrees(lat_values)
            lon = _convert_to_degrees(lon_values)

            # Apply hemisphere ref
            if tags[gps_lat_ref_key].values.strip().upper() == 'S':
                lat = -lat
            if tags[gps_lon_ref_key].values.strip().upper() == 'W':
                lon = -lon

            # Reverse geocode
            geolocator = Nominatim(user_agent="gps_extractor_app", timeout=10)
            location = geolocator.reverse((lat, lon), language='en')
            address = location.address if location else None

            print(f"Extracted GPS: lat={lat}, lon={lon}, address={address}")
            return lat, lon, address
        else:
            print("No GPS EXIF tags found in image.")

    except Exception as e:
        print(f"Error extracting GPS from image: {e}")

    return None


def analyze_fire_severity(detections, img_shape, source_path=None):
    """
    Analyze fire severity based on multiple factors including detection size, confidence, and number of detections
    Returns: severity level, message, and GPS coordinates
    """
    if len(detections) == 0:
        return "No fire detected", "Safe", None, None, None
    
    # Initialize variables
    total_area = 0
    max_conf = 0
    fire_locations = []
    detected_fire = False
    fire_count = 0
    
    # Process detections
    for *xyxy, conf, cls in detections:
        cls_id = int(cls)
        area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
        
        # Check for any type of fire (classes 1,2,3,4)
        if cls_id in [1, 2, 3, 4]:  # safe fire, warning, danger, or smoke
            detected_fire = True
            fire_count += 1
            total_area += area
            
            if conf > max_conf:
                max_conf = conf
            
            center_x = (xyxy[0] + xyxy[2]) / 2
            center_y = (xyxy[1] + xyxy[3]) / 2
            fire_locations.append((center_x, center_y))
    
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
    relative_area = (total_area / img_area) * 100
    
    # Calculate fire spread factor (number of distinct fire locations)
    spread_factor = len(fire_locations)
    
    # Determine severity based on multiple factors
    severity_score = 0
    
    # Factor 1: Area coverage (0-3 points)
    if relative_area < 5:
        severity_score += 0
    elif relative_area < 10:
        severity_score += 1
    elif relative_area < 20:
        severity_score += 2
    else:
        severity_score += 3
    
    # Factor 2: Number of fire detections (0-2 points)
    if fire_count == 1:
        severity_score += 0
    elif fire_count <= 3:
        severity_score += 1
    else:
        severity_score += 2
    
    # Factor 3: Confidence level (0-2 points)
    if max_conf < 0.5:
        severity_score += 0
    elif max_conf < 0.7:
        severity_score += 1
    else:
        severity_score += 2
    
    # Factor 4: Fire spread (0-1 point)
    if spread_factor > 2:
        severity_score += 1
    
    # Determine final severity based on total score (0-8 points)
    if severity_score <= 2:
        return "Safe", "Low risk - Small, contained fire detected", gps_coords, fire_locations[0] if fire_locations else None, address
    elif severity_score <= 5:
        return "Warning", "Medium risk - Growing fire with multiple detection points", gps_coords, fire_locations[0] if fire_locations else None, address
    else:
        return "Danger", "High risk - Large, spreading fire detected!", gps_coords, fire_locations[0] if fire_locations else None, address

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

def capture_screen():
    """
    Capture the entire screen using mss
    Returns: numpy array of the screen capture
    """
    with mss() as sct:
        monitor = sct.monitors[1]  # Primary monitor
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        # Convert from BGRA to BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

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
        print("3. Capture from screen")
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            # Try to find Iriun camera
            iriun_url = find_iriun_camera()
            if iriun_url:
                source = iriun_url
                print("\n✓ Successfully connected to Iriun camera!")
            else:
                source = 0
                print("\nUsing local webcam instead.")
        elif choice == "3":
            source = "screen"
            print("\nUsing screen capture.")
        else:
            source = 0
            print("\nUsing local webcam.")
    
    # Initialize video capture
    cap = None
    if isinstance(source, (int, str)):
        if source == "screen":
            cap = None  # We'll handle screen capture differently
        elif str(source).isdigit():
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
    if source != "screen":
        dataset = LoadImages(source, img_size=img_size, stride=stride)
    else:
        dataset = None
    
    # Initialize FPS calculation variables
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    print("\n=== Fire Detection Started ===")
    print("Press 'q' to quit")
    print("=============================\n")
    
    # Run inference
    while True:
        # Get frame
        if source == "screen":
            im0 = capture_screen()
            # Resize for model input
            im = cv2.resize(im0, (img_size, img_size))
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)
        else:
            try:
                path, im, im0s, vid_cap, s = next(iter(dataset))
                im0 = im0s.copy()
            except StopIteration:
                break
        
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
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                
                # Analyze fire severity
                severity, message, gps_coords, fire_location, address = analyze_fire_severity(det, im0.shape, None)
                
                # Add severity text
                cv2.putText(im0, f"Status: {severity}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(im0, message, (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Add FPS counter
                cv2.putText(im0, f"FPS: {fps:.1f}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add fire location in image coordinates
                if fire_location:
                    x, y = fire_location
                    cv2.circle(im0, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv2.putText(im0, f"Fire at: ({int(x)}, {int(y)})", (10, 150),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{model.names[c]} {conf:.2f}'
                    annotator = Annotator(im0, line_width=3, example=str(model.names))
                    annotator.box_label(xyxy, label, color=colors(c, True))
            
            # Save results if save_dir is specified
            if save_dir:
                if source == "screen":
                    if not hasattr(run_detection, 'video_writer'):
                        run_detection.video_writer = cv2.VideoWriter(
                            f'{save_dir}/screen_output.mp4',
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            30.0,  # FPS for screen capture
                            (im0.shape[1], im0.shape[0])
                        )
                    run_detection.video_writer.write(im0)
                elif isinstance(source, (int, str)) and str(source).isdigit() or source.endswith('.mp4'):  # Video
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