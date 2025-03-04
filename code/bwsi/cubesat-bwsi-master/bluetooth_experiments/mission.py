import time
import json
import os
import numpy as np
import base64
import socket
import bluetooth
from io import BytesIO
from dataclasses import dataclass

# Hardware imports
import board
from adafruit_lsm6ds.lsm6dsox import LSM6DSOX as LSM6DS
from adafruit_lis3mdl import LIS3MDL
from picamera2 import Picamera2, Preview

# Fire detection model import
from fire_detection_inference import SimplifiedFireDetectionCNN, Config, load_model, run_inference

# Constants
IMAGE_DIR = "images"
BLUETOOTH_UUID = "94f39d29-7d6d-437d-973b-fba39e49d4ee"
IMAGE_INTERVAL = 60  # Take picture every 60 seconds
MISSION_DURATION = 10 * IMAGE_INTERVAL  # 10 images for the demo
MAX_RETRIES = 3
# SOCKET_BUFFER_SIZE = 8192
SOCKET_BUFFER_SIZE = 2048

@dataclass
class Heartbeat:
    time: float
    accelx: float
    accely: float
    accelz: float
    battery: float
    image_count: int
    health: int  # Health status codes as defined in your spec
    
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
        """Initialize the CubeSat mission"""
        print("Initializing CubeSat mission...")
        
        # Create image directory if it doesn't exist
        self.image_dir = image_dir
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Initialize hardware components
        self.init_hardware()
        
        # Load fire detection model
        self.model = load_model(model_path="fire_detection_model.pth", device="cpu")
        self.config = Config()
        
        # Mission state
        self.mission_start_time = None
        self.image_count = 0
        self.health = 0  # Start with healthy status
        self.connected = False
        self.socket = None
        
        # Connect to ground station
        self.connect_to_ground_station()
    
    def init_hardware(self):
        """Initialize hardware components"""
        try:
            # Initialize I2C
            self.i2c = board.I2C()
            
            # Initialize accelerometer/gyroscope
            self.accel_gyro = LSM6DS(self.i2c)
            
            # Initialize magnetometer
            self.mag = LIS3MDL(self.i2c)
            
            # Initialize camera
            self.picam2 = Picamera2()
            self.picam2.start_preview(Preview.DRM)
            self.capture_config = self.picam2.create_still_configuration()
            self.picam2.configure(self.capture_config)
            self.picam2.start()
            
            print("Hardware initialization successful")
        except Exception as e:
            print(f"Error initializing hardware: {e}")
            self.health = 3  # Major issue in a subsystem
    
    def connect_to_ground_station(self):
        """Connect to the ground station via Bluetooth"""
        print("Connecting to ground station...")
        
        try:
            # Search for ground station service
            ground_station_address = None
            
            nearby_devices = bluetooth.discover_devices(lookup_names=True)
            print(f"Found {len(nearby_devices)} devices")
            
            for addr, name in nearby_devices:
                print(f"Device: {name} at {addr}")
                # You can add a specific name to look for here
                if "CubeSat" in name or "Ground" in name:
                    ground_station_address = addr
                    break
            
            # If no specific device found, try the first device
            if not ground_station_address and nearby_devices:
                ground_station_address = nearby_devices[0][0]
            
            if not ground_station_address:
                print("No Bluetooth devices found, will try to connect using default settings")
                ground_station_address = 'D8:3A:DD:8E:CF:91'  # Can be replaced with a known address
            
            # Connect to ground station
            self.socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            
            # service_matches = bluetooth.find_service(uuid=BLUETOOTH_UUID, address=ground_station_address)
            service_matches = bluetooth.find_service(address=ground_station_address)
            
            if not service_matches:
                print("No matching services found. Trying default port...")
                self.socket.connect((ground_station_address, 1))  # Try default RFCOMM port
            else:
                first_match = service_matches[0]
                port = first_match["port"]
                name = first_match["name"]
                host = first_match["host"]
                
                print(f"Connecting to {name} on {host}:{port}")
                self.socket.connect((host, port))
            
            self.connected = True
            print("Connected to ground station")
        except Exception as e:
            print(f"Failed to connect to ground station: {e}")
            self.health = 1  # Minor issue in a subsystem
            self.connected = False
    
    def get_battery_level(self):
        """Simulate battery level (would be real hardware reading)"""
        # In a real implementation, this would read from a battery sensor
        # For demo, simulate battery drain over time
        if not self.mission_start_time:
            return 1.0
        
        elapsed = time.time() - self.mission_start_time
        total_duration = MISSION_DURATION
        
        # Start at 100%, end at around 70%
        battery_level = 1.0 - (0.3 * min(elapsed / total_duration, 1.0))
        
        # Add some noise to make it more realistic
        battery_level += np.random.normal(0, 0.01)
        return max(min(battery_level, 1.0), 0.0)
    
    def get_accelerometer_data(self):
        """Get data from the accelerometer"""
        try:
            accel_x, accel_y, accel_z = self.accel_gyro.acceleration
            return accel_x, accel_y, accel_z
        except Exception as e:
            print(f"Error reading accelerometer: {e}")
            self.health = max(1, self.health)  # Set at least to minor issue
            return 0.0, 0.0, 0.0
    
    def generate_heartbeat(self):
        """Generate a heartbeat packet with telemetry data"""
        current_time = time.time()
        accel_x, accel_y, accel_z = self.get_accelerometer_data()
        battery = self.get_battery_level()
        
        return Heartbeat(
            time=current_time,
            accelx=accel_x,
            accely=accel_y,
            accelz=accel_z,
            battery=battery,
            image_count=self.image_count,
            health=self.health
        )
    
    def send_heartbeat(self):
        """Send heartbeat packet to ground station"""
        if not self.connected:
            print("Not connected to ground station, cannot send heartbeat")
            return False
        
        heartbeat = self.generate_heartbeat()
        
        try:
            message = {
                'type': 'heartbeat',
                'data': heartbeat.to_dict()
            }
            
            json_data = json.dumps(message).encode('utf-8')
            json_data += b"END_MESSAGE"
            
            self.socket.send(json_data)
            print(f"Sent heartbeat: time={heartbeat.time}, battery={heartbeat.battery:.2f}, health={heartbeat.health}")
            return True
        except Exception as e:
            print(f"Error sending heartbeat: {e}")
            self.connected = False
            self.health = max(1, self.health)  # At least minor issue
            return False
    
    def take_photo(self):
        """Capture a photo using the camera"""
        try:
            timestamp = time.time()
            img_path = f"{self.image_dir}/{timestamp}.png"
            
            print(f"Taking photo: {img_path}")
            self.picam2.capture_file(img_path)
            print("Photo captured successfully")
            
            self.image_count += 1
            return img_path, timestamp
        except KeyboardInterrupt:
            print('Interrupted. Exiting...')
            raise
        except Exception as e:
            print(f'Error taking photo: {e}')
            self.health = max(3, self.health)  # Major issue in a subsystem
            return None, time.time()
    
    def classify_image(self, img_path):
        """Perform fire detection classification on the image"""
        try:
            if not img_path or not os.path.exists(img_path):
                print("Image file not found, skipping classification")
                return None
            
            print(f"Classifying image: {img_path}")
            # Run inference using the provided model and config
            heatmap_scores, heatmap_binary = run_inference(self.model, img_path, self.config)
            
            # The heatmap_scores is the classification data we'll send to the ground station
            print("Classification complete")
            return heatmap_scores
        except Exception as e:
            print(f"Error during classification: {e}")
            self.health = max(2, self.health)  # At least minor issues in multiple subsystems
            return None
    
    '''
    def send_image(self, img_path, timestamp):
        """Send image to ground station"""
        if not self.connected or not img_path or not os.path.exists(img_path):
            print("Cannot send image: not connected or image not found")
            return False
        
        try:
            # First, notify the ground station that we're about to send an image
            with open(img_path, 'rb') as f:
                img_data = f.read()
            
            img_size = len(img_data)
            
            # Send image metadata first
            message = {
                'type': 'image_start',
                'timestamp': timestamp,
                'size': img_size
            }
            
            json_data = json.dumps(message).encode('utf-8')
            json_data += b"END_MESSAGE"
            
            self.socket.send(json_data)
            print(f"Sending image of size {img_size} bytes")
            
            # Now send the actual image data in chunks
            sent_bytes = 0
            while sent_bytes < img_size:
                chunk_size = min(SOCKET_BUFFER_SIZE, img_size - sent_bytes)
                chunk = img_data[sent_bytes:sent_bytes + chunk_size]
                self.socket.send(chunk)
                sent_bytes += chunk_size
                
                # Report progress
                if img_size > 0 and sent_bytes % (SOCKET_BUFFER_SIZE * 10) == 0:
                    progress = (sent_bytes / img_size) * 100
                    print(f"Image upload: {progress:.1f}% complete")
            
            print("Image sent successfully")
            
            # Wait for ground station to acknowledge receipt and request classification data
            response = self.socket.recv(SOCKET_BUFFER_SIZE)
            if b"READY_FOR_CLASSIFICATION" in response:
                return True
            else:
                print(f"Unexpected response after image send: {response}")
                return False
        except Exception as e:
            print(f"Error sending image: {e}")
            self.connected = False
            self.health = max(2, self.health)  # At least minor issues in multiple subsystems
            return False
    '''
    # In mission.py: Modify the send_image function
    def send_image(self, img_path, timestamp):
        if not self.connected or not img_path or not os.path.exists(img_path):
            print("Cannot send image: not connected or image not found")
            return False
        
        try:
            # Read image file
            with open(img_path, 'rb') as f:
                img_data = f.read()
            
            img_size = len(img_data)
            
            # Send image metadata first
            message = {
                'type': 'image_start',
                'timestamp': timestamp,
                'size': img_size
            }
            
            json_data = json.dumps(message).encode('utf-8')
            json_data += b"END_MESSAGE"
            
            self.socket.send(json_data)
            print(f"Sending image of size {img_size} bytes")
            
            # Wait for acknowledgment to start
            response = self.socket.recv(1024)
            if b"START_TRANSFER" not in response:
                print(f"Ground station not ready to receive: {response}")
                return False
            
            # Use smaller chunks
            CHUNK_SIZE = 4096  # Smaller chunks
            
            # Now send the actual image data in chunks with acknowledgments
            sent_bytes = 0
            chunk_num = 0
            
            while sent_bytes < img_size:
                chunk_size = min(CHUNK_SIZE, img_size - sent_bytes)
                chunk = img_data[sent_bytes:sent_bytes + chunk_size]
                
                # Send chunk number and size header
                header = f"CHUNK:{chunk_num}:{chunk_size}:".encode('utf-8')
                self.socket.send(header)
                
                # Small delay to let receiver process
                time.sleep(0.01)
                
                # Send the chunk data
                bytes_sent = self.socket.send(chunk)
                
                # Wait for acknowledgment
                ack = self.socket.recv(1024)
                if f"ACK:{chunk_num}".encode('utf-8') not in ack:
                    print(f"Failed to get acknowledgment for chunk {chunk_num}, retrying...")
                    continue  # Retry this chunk
                
                sent_bytes += bytes_sent
                chunk_num += 1
                
                # Report progress less frequently
                if chunk_num % 10 == 0:
                    progress = (sent_bytes / img_size) * 100
                    print(f"Image upload: {progress:.1f}% complete")
            
            print("Image sent successfully")
            
            # Wait for final acknowledgment
            response = self.socket.recv(SOCKET_BUFFER_SIZE)
            if b"READY_FOR_CLASSIFICATION" in response:
                return True
            else:
                print(f"Unexpected response after image send: {response}")
                return False
                
        except Exception as e:
            print(f"Error sending image: {e}")
            self.connected = False
            self.health = max(2, self.health)  # At least minor issues in multiple subsystems
            return False
    
    def send_classification(self, classification_data, timestamp):
        """Send classification data to ground station"""
        if not self.connected or classification_data is None:
            print("Cannot send classification: not connected or no data")
            return False
        
        try:
            message = {
                'type': 'classification',
                'timestamp': timestamp,
                'data': classification_data.tolist() if isinstance(classification_data, np.ndarray) else classification_data
            }
            
            json_data = json.dumps(message).encode('utf-8')
            json_data += b"END_MESSAGE"
            
            self.socket.send(json_data)
            print("Classification data sent successfully")
            return True
        except Exception as e:
            print(f"Error sending classification data: {e}")
            self.connected = False
            self.health = max(1, self.health)  # At least minor issue
            return False
        
    # In mission.py: Add connection monitoring and reconnection
    def check_connection(self):
        """Check if connection is still active and try to reconnect if needed"""
        if not self.connected:
            print("Connection lost, attempting to reconnect...")
            retry_count = 0
            while retry_count < MAX_RETRIES:
                try:
                    self.connect_to_ground_station()
                    if self.connected:
                        print("Successfully reconnected to ground station")
                        break
                except Exception as e:
                    print(f"Reconnection attempt {retry_count+1} failed: {e}")
                
                retry_count += 1
                time.sleep(2 * retry_count)  # Increasing backoff
            
            if not self.connected:
                print("Failed to reconnect after multiple attempts")
        
        return self.connected
    
    def run_mission(self):
        """Run the main mission sequence"""
        print("Starting CubeSat mission...")
        self.mission_start_time = time.time()
        last_image_time = 0
        mission_end_time = self.mission_start_time + MISSION_DURATION
        
        # Send initial heartbeat
        self.send_heartbeat()
        
        # Main mission loop
        try:
            while time.time() < mission_end_time:
                current_time = time.time()
                elapsed = current_time - self.mission_start_time
                
                # In the run_mission method
                if int(elapsed) % 30 == 0:  # Every 30 seconds
                    self.check_connection()
                
                # Send heartbeat every 5 seconds
                if int(elapsed) % 5 == 0:
                    self.send_heartbeat()

                # Take image every IMAGE_INTERVAL seconds
                if current_time - last_image_time >= IMAGE_INTERVAL:
                    # Take photo
                    img_path, timestamp = self.take_photo()
                    
                    if img_path:
                        # Classify image
                        classification_data = self.classify_image(img_path)
                        
                        # Send image to ground station
                        if self.send_image(img_path, timestamp):
                            # Send classification data
                            self.send_classification(classification_data, timestamp)
                        
                        last_image_time = current_time
                
                # Small delay to prevent CPU overuse
                time.sleep(0.1)
                
                # Check if we've taken 10 images already
                if self.image_count >= 10:
                    print("Completed 10 images, mission complete!")
                    break
                
        except KeyboardInterrupt:
            print("Mission interrupted by user")
        except Exception as e:
            print(f"Mission error: {e}")
            self.health = max(4, self.health)  # Major issues in multiple subsystems
        finally:
            # Send final heartbeat
            self.send_heartbeat()
            
            # Clean up resources
            self.cleanup()
            
            print("Mission complete!")
    
    def cleanup(self):
        """Clean up resources"""
        try:
            # Stop camera
            self.picam2.stop()
            
            # Close Bluetooth connection
            if self.socket:
                self.socket.close()
            
            print("Resources cleaned up")
        except Exception as e:
            print(f"Error during cleanup: {e}")

# Main execution
if __name__ == "__main__":
    mission = CubeSatMission()
    mission.run_mission()