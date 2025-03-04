import time
import json
import os
import numpy as np
import socket
import logging
from datetime import datetime
from dataclasses import dataclass

# Hardware imports
import board
from adafruit_lsm6ds.lsm6dsox import LSM6DSOX as LSM6DS
from adafruit_lis3mdl import LIS3MDL
from picamera2 import Picamera2, Preview

# Fire detection model import
from fire_detection_inference import SimplifiedFireDetectionCNN, Config, load_model, run_inference

# Constants
GROUND_STATION_IP = "192.168.0.20"  # Update with your ground station's IP address
GROUND_STATION_PORT = 5555
IMAGE_DIR = "images"
IMAGE_INTERVAL = 60            # Take a picture every 60 seconds
MISSION_DURATION = 10 * IMAGE_INTERVAL  # For demo: 10 images
MAX_RETRIES = 3
SOCKET_BUFFER_SIZE = 4096

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Heartbeat:
    time: float
    accelx: float
    accely: float
    accelz: float
    battery: float
    image_count: int
    health: int
    
    def to_dict(self):
        return {
            'time': self.time,
            'accelx': self.accelx,
            'accely': self.accely,
            'accelz': self.accelz,
            'battery': self.battery,
            'image_count': self.image_count,
            'health': self.health
        }

class CubeSatMission:
    def __init__(self, image_dir=IMAGE_DIR):
        logger.info("Initializing CubeSat mission...")
        self.image_dir = image_dir
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Initialize hardware components
        self.init_hardware()
        
        # Load fire detection model and configuration
        self.model = load_model(model_path="fire_detection_model.pth", device="cpu")
        self.config = Config()
        
        # Mission state
        self.mission_start_time = None
        self.image_count = 0
        self.health = 0  # Start with healthy status
        self.connected = False
        self.socket = None
        
        # Connect via WiFi (TCP/IP) to the ground station
        self.connect_to_ground_station()
    
    def init_hardware(self):
        try:
            self.i2c = board.I2C()
            self.accel_gyro = LSM6DS(self.i2c)
            self.mag = LIS3MDL(self.i2c)
            self.picam2 = Picamera2()
            self.picam2.start_preview(Preview.DRM)
            self.capture_config = self.picam2.create_still_configuration()
            self.picam2.configure(self.capture_config)
            self.picam2.start()
            logger.info("Hardware initialization successful")
        except Exception as e:
            logger.error(f"Error initializing hardware: {e}")
            self.health = 3  # Major issue
    
    def connect_to_ground_station(self):
        logger.info(f"Connecting to ground station at {GROUND_STATION_IP}:{GROUND_STATION_PORT}...")
        try:
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)  # Connection timeout
            self.socket.connect((GROUND_STATION_IP, GROUND_STATION_PORT))
            self.socket.settimeout(30)  # Operation timeout
            self.connected = True
            logger.info("Connected to ground station via WiFi")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to ground station: {e}")
            self.health = 1  # Minor issue
            self.connected = False
            return False
    
    def send_data_with_length_prefix(self, data):
        """Send data prefixed with its 4-byte length (big endian)."""
        if not self.connected:
            return False
        try:
            length = len(data)
            length_bytes = length.to_bytes(4, byteorder='big')
            self.socket.sendall(length_bytes)
            bytes_sent = 0
            while bytes_sent < length:
                chunk_size = min(SOCKET_BUFFER_SIZE, length - bytes_sent)
                sent = self.socket.send(data[bytes_sent:bytes_sent + chunk_size])
                if sent == 0:
                    raise RuntimeError("Socket connection broken")
                bytes_sent += sent
            return True
        except Exception as e:
            logger.error(f"Error sending data: {e}")
            self.connected = False
            return False
    
    def get_battery_level(self):
        """Simulate battery level with gradual drain."""
        if not self.mission_start_time:
            return 1.0
        elapsed = time.time() - self.mission_start_time
        total_duration = MISSION_DURATION
        battery_level = 1.0 - (0.3 * min(elapsed / total_duration, 1.0))
        battery_level += np.random.normal(0, 0.01)
        return max(min(battery_level, 1.0), 0.0)
    
    def get_accelerometer_data(self):
        try:
            return self.accel_gyro.acceleration
        except Exception as e:
            logger.error(f"Error reading accelerometer: {e}")
            self.health = max(1, self.health)
            return (0.0, 0.0, 0.0)
    
    def generate_heartbeat(self):
        current_time = time.time()
        accel = self.get_accelerometer_data()
        battery = self.get_battery_level()
        return Heartbeat(
            time=current_time,
            accelx=accel[0],
            accely=accel[1],
            accelz=accel[2],
            battery=battery,
            image_count=self.image_count,
            health=self.health
        )
    
    def send_heartbeat(self):
        """Send a heartbeat packet (type 1)."""
        if not self.connected:
            logger.warning("Not connected to ground station, cannot send heartbeat")
            return False
        heartbeat = self.generate_heartbeat()
        try:
            # Send heartbeat indicator (type 1)
            self.socket.send(b'\x01')
            # Send heartbeat data as JSON with length prefix
            heartbeat_json = json.dumps(heartbeat.to_dict()).encode('utf-8')
            if not self.send_data_with_length_prefix(heartbeat_json):
                return False
            # Wait for acknowledgment (expected: b'\x01')
            ack = self.socket.recv(1)
            if ack != b'\x01':
                logger.warning(f"Unexpected acknowledgment for heartbeat: {ack}")
                return False
            logger.info(f"Sent heartbeat: time={heartbeat.time}, battery={heartbeat.battery:.2f}, health={heartbeat.health}")
            return True
        except Exception as e:
            logger.error(f"Error sending heartbeat: {e}")
            self.connected = False
            self.health = max(1, self.health)
            return False
    
    def take_photo(self):
        """Capture a photo using the camera."""
        try:
            timestamp = time.time()
            img_path = os.path.join(self.image_dir, f"{timestamp}.png")
            logger.info(f"Taking photo: {img_path}")
            self.picam2.capture_file(img_path)
            logger.info("Photo captured successfully")
            self.image_count += 1
            return img_path, timestamp
        except Exception as e:
            logger.error(f"Error taking photo: {e}")
            self.health = max(3, self.health)
            return None, time.time()
    
    def classify_image(self, img_path):
        """Perform fire detection classification on the image."""
        try:
            if not img_path or not os.path.exists(img_path):
                logger.warning("Image file not found, skipping classification")
                return None
            logger.info(f"Classifying image: {img_path}")
            heatmap_scores, heatmap_binary = run_inference(self.model, img_path, self.config)
            logger.info("Classification complete")
            return heatmap_scores
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            self.health = max(2, self.health)
            return None
    
    def send_image(self, img_path, timestamp):
        """Send an image (type 2) to the ground station."""
        if not self.connected or not img_path or not os.path.exists(img_path):
            logger.warning("Cannot send image: not connected or image not found")
            return False
        try:
            # Send image indicator (type 2)
            self.socket.send(b'\x02')
            # Send the image timestamp (as a string) with length prefix
            timestamp_str = str(timestamp).encode('utf-8')
            if not self.send_data_with_length_prefix(timestamp_str):
                return False
            # Wait for acknowledgment for timestamp
            ack = self.socket.recv(1)
            if ack != b'\x01':
                logger.warning(f"Unexpected ack for image timestamp: {ack}")
                return False
            # Read image file
            with open(img_path, 'rb') as f:
                img_data = f.read()
            logger.info(f"Sending image of size {len(img_data)} bytes")
            # Send image data with length prefix
            if not self.send_data_with_length_prefix(img_data):
                return False
            # Wait for acknowledgment for image data
            ack = self.socket.recv(1)
            if ack != b'\x01':
                logger.warning(f"Unexpected ack for image data: {ack}")
                return False
            logger.info("Image sent successfully")
            return True
        except Exception as e:
            logger.error(f"Error sending image: {e}")
            self.connected = False
            self.health = max(2, self.health)
            return False
    
    def send_classification(self, classification_data, timestamp):
        """Send image classification data (type 3) to the ground station."""
        if not self.connected or classification_data is None:
            logger.warning("Cannot send classification: not connected or no data")
            return False
        try:
            # Send classification indicator (type 3)
            self.socket.send(b'\x03')
            # Send timestamp with length prefix
            timestamp_str = str(timestamp).encode('utf-8')
            if not self.send_data_with_length_prefix(timestamp_str):
                return False
            # Wait for acknowledgment for timestamp
            ack = self.socket.recv(1)
            if ack != b'\x01':
                logger.warning(f"Unexpected ack for classification timestamp: {ack}")
                return False
            # Prepare and send classification data as JSON with length prefix
            classification_json = json.dumps(
                classification_data.tolist() if isinstance(classification_data, np.ndarray) else classification_data
            ).encode('utf-8')
            if not self.send_data_with_length_prefix(classification_json):
                return False
            # Wait for acknowledgment for classification data
            ack = self.socket.recv(1)
            if ack != b'\x01':
                logger.warning(f"Unexpected ack for classification data: {ack}")
                return False
            logger.info("Classification data sent successfully")
            return True
        except Exception as e:
            logger.error(f"Error sending classification data: {e}")
            self.connected = False
            self.health = max(1, self.health)
            return False
    
    def check_connection(self):
        """If the connection is lost, try reconnecting."""
        if not self.connected:
            logger.info("Connection lost, attempting to reconnect...")
            retry_count = 0
            while retry_count < MAX_RETRIES:
                if self.connect_to_ground_station():
                    logger.info("Successfully reconnected to ground station")
                    break
                retry_count += 1
                time.sleep(2 * retry_count)
            if not self.connected:
                logger.error("Failed to reconnect after multiple attempts")
        return self.connected
    
    def run_mission(self):
        """Run the main mission sequence."""
        logger.info("Starting CubeSat mission...")
        self.mission_start_time = time.time()
        last_image_time = 0
        mission_end_time = self.mission_start_time + MISSION_DURATION
        
        # Send an initial heartbeat
        self.send_heartbeat()
        
        try:
            while time.time() < mission_end_time:
                current_time = time.time()
                elapsed = current_time - self.mission_start_time
                
                # Check connection every 30 seconds
                if int(elapsed) % 30 == 0:
                    self.check_connection()
                
                # Send heartbeat every 5 seconds
                if int(elapsed) % 5 == 0:
                    self.send_heartbeat()
                
                # Take an image every IMAGE_INTERVAL seconds
                if current_time - last_image_time >= IMAGE_INTERVAL:
                    img_path, timestamp = self.take_photo()
                    if img_path:
                        classification_data = self.classify_image(img_path)
                        if self.send_image(img_path, timestamp):
                            self.send_classification(classification_data, timestamp)
                        last_image_time = current_time
                    # End mission after 10 images
                    if self.image_count >= 10:
                        logger.info("Completed 10 images, mission complete!")
                        break
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Mission interrupted by user")
        except Exception as e:
            logger.error(f"Mission error: {e}")
            self.health = max(4, self.health)
        finally:
            # Send final heartbeat and clean up resources
            self.send_heartbeat()
            self.cleanup()
            logger.info("Mission complete!")
    
    def cleanup(self):
        """Clean up hardware and network resources."""
        try:
            self.picam2.stop()
            if self.socket:
                self.socket.close()
            logger.info("Resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    mission = CubeSatMission()
    mission.run_mission()
