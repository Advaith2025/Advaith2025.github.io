import os
import time
import numpy as np
import socket
import json
import threading
import base64
from datetime import datetime
from PIL import Image
from io import BytesIO
import bluetooth
from flask import Flask, render_template, jsonify, Response, request, send_from_directory
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SOCKET_BUFFER_SIZE = 4096
DATA_DIR = "cubesat_data"
IMAGES_DIR = f"{DATA_DIR}/images"
TELEMETRY_FILE = f"{DATA_DIR}/telemetry.json"
MAX_MISSION_DURATION = 600  # 10 minutes

# Create directories if they don't exist
os.makedirs(IMAGES_DIR, exist_ok=True)

# Define data structures
@dataclass
class Heartbeat:
    time: float
    accelx: float
    accely: float
    accelz: float
    battery: float
    image_count: int
    health: int  # Health status codes as defined in CubeSat code
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            time=data.get('time', 0.0),
            accelx=data.get('accelx', 0.0),
            accely=data.get('accely', 0.0),
            accelz=data.get('accelz', 0.0),
            battery=data.get('battery', 0.0),
            image_count=data.get('image_count', 0),
            health=data.get('health', 0)
        )
    
    def to_dict(self):
        return {
            'time': self.time,
            'accelx': self.accelx,
            'accely': self.accely,
            'accelz': self.accelz,
            'battery': self.battery,
            'image_count': self.image_count,
            'health': self.health,
            'timestamp': datetime.fromtimestamp(self.time).strftime('%Y-%m-%d %H:%M:%S')
        }

@dataclass
class ImageData:
    timestamp: float
    image_path: str
    classification_data: np.ndarray
    
    def to_dict(self):
        return {
            'timestamp': self.timestamp,
            'image_path': self.image_path,
            'classification_data': self.classification_data.tolist() if self.classification_data is not None else None,
            'datetime': datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S')
        }

class GroundStation:
    def __init__(self):
        self.telemetry_data = []
        self.image_data = []
        self.latest_heartbeat = None
        self.latest_image = None
        self.connected = False
        self.socket = None
        self.client_socket = None
        self.bluetooth_thread = None
        self.lock = threading.Lock()
        
        # Load existing telemetry if available
        if os.path.exists(TELEMETRY_FILE):
            try:
                with open(TELEMETRY_FILE, 'r') as f:
                    data = json.load(f)
                    self.telemetry_data = [Heartbeat.from_dict(h) for h in data.get('telemetry', [])]
                    logger.info(f"Loaded {len(self.telemetry_data)} telemetry records")
            except Exception as e:
                logger.error(f"Failed to load telemetry data: {e}")
    
    # In mission.py: Add connection monitoring and reconnection
# def check_connection(self):
#     """Check if connection is still active and try to reconnect if needed"""
#     if not self.connected:
#         print("Connection lost, attempting to reconnect...")
#         retry_count = 0
#         while retry_count < MAX_RETRIES:
#             try:
#                 self.connect_to_ground_station()
#                 if self.connected:
#                     print("Successfully reconnected to ground station")
#                     break
#             except Exception as e:
#                 print(f"Reconnection attempt {retry_count+1} failed: {e}")
            
#             retry_count += 1
#             time.sleep(2 * retry_count)  # Increasing backoff
        
#         if not self.connected:
#             print("Failed to reconnect after multiple attempts")
    
#     return self.connected

    def start_bluetooth_server(self):
        """Set up and start the Bluetooth server"""
        try:
            self.socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            self.socket.bind(("", bluetooth.PORT_ANY))
            self.socket.listen(1)
            
            port = self.socket.getsockname()[1]
            uuid = "94f39d29-7d6d-437d-973b-fba39e49d4ee"
            
            bluetooth.advertise_service(
                self.socket, "CubeSatGroundStation",
                service_id=uuid,
                service_classes=[uuid, bluetooth.SERIAL_PORT_CLASS],
                profiles=[bluetooth.SERIAL_PORT_PROFILE]
            )
            
            logger.info(f"Waiting for Bluetooth connection on RFCOMM channel {port}")
            
            # Start a thread to accept connections
            self.bluetooth_thread = threading.Thread(target=self._bluetooth_listen)
            self.bluetooth_thread.daemon = True
            self.bluetooth_thread.start()
            
            return True
        except Exception as e:
            logger.error(f"Failed to start Bluetooth server: {e}")
            return False
    
    def _bluetooth_listen(self):
        """Listen for and handle Bluetooth connections"""
        try:
            while True:
                logger.info("Waiting for connection...")
                client_sock, client_info = self.socket.accept()
                logger.info(f"Accepted connection from {client_info}")
                
                with self.lock:
                    self.client_socket = client_sock
                    self.connected = True
                
                try:
                    self._handle_client(client_sock)
                except Exception as e:
                    logger.error(f"Error handling client: {e}")
                finally:
                    with self.lock:
                        self.connected = False
                        if self.client_socket:
                            self.client_socket.close()
                            self.client_socket = None
        except Exception as e:
            logger.error(f"Bluetooth listener error: {e}")
        finally:
            if self.socket:
                self.socket.close()
    
    # def _handle_client(self, client_sock):
    #     """Handle communication with a connected client"""
    #     buffer = b""
        
    #     while True:
    #         try:
    #             data = client_sock.recv(SOCKET_BUFFER_SIZE)
    #             if not data:
    #                 logger.info("Connection closed by client")
    #                 break
                
    #             buffer += data
                
    #             # Look for message delimiters
    #             while b"END_MESSAGE" in buffer:
    #                 msg, buffer = buffer.split(b"END_MESSAGE", 1)
    #                 try:
    #                     # Decode the message
    #                     msg_data = json.loads(msg.decode('utf-8'))
    #                     msg_type = msg_data.get('type')
                        
    #                     if msg_type == 'heartbeat':
    #                         self._process_heartbeat(msg_data.get('data', {}))
    #                     elif msg_type == 'image_start':
    #                         # Prepare to receive image data
    #                         img_timestamp = msg_data.get('timestamp')
    #                         img_size = msg_data.get('size')
    #                         img_name = f"{img_timestamp}.png"
    #                         img_path = os.path.join(IMAGES_DIR, img_name)
                            
    #                         # Receive the image data in chunks
    #                         # self._receive_image(client_sock, img_path, img_size)
                            
    #                         # Now expect classification data
    #                         client_sock.send(b"READY_FOR_CLASSIFICATION")
    #                     elif msg_type == 'classification':
    #                         # Process the classification data
    #                         self._process_classification(msg_data.get('timestamp'), msg_data.get('data'))
    #                 except json.JSONDecodeError:
    #                     logger.error(f"Failed to decode JSON message: {msg}")
    #                 except Exception as e:
    #                     logger.error(f"Error processing message: {e}")
            
    #         except Exception as e:
    #             logger.error(f"Error receiving data: {e}")
    #             break
    
    # def _receive_image(self, client_sock, img_path, expected_size):
    #     """Receive image data and save it to disk"""
    #     logger.info(f"Receiving image of size {expected_size} bytes")
        
    #     received_data = b""
    #     total_received = 0
        
    #     while total_received < expected_size:
    #         chunk = client_sock.recv(min(SOCKET_BUFFER_SIZE, expected_size - total_received))
    #         if not chunk:
    #             raise Exception("Connection closed during image transfer")
            
    #         received_data += chunk
    #         total_received += len(chunk)
            
    #         # Update progress
    #         if expected_size > 0:
    #             progress = (total_received / expected_size) * 100
    #             if total_received % (SOCKET_BUFFER_SIZE * 10) == 0:
    #                 logger.info(f"Image transfer: {progress:.1f}% complete")
        
    #     # Save the image
    #     with open(img_path, 'wb') as f:
    #         f.write(received_data)
        
    #     logger.info(f"Image saved to {img_path}")
        
    #     # Store the image info
    #     timestamp = float(os.path.basename(img_path).split('.')[0])
    #     img_data = ImageData(timestamp=timestamp, image_path=img_path, classification_data=None)
        
    #     with self.lock:
    #         self.latest_image = img_data
    #         self.image_data.append(img_data)

    

    # In ground_station.py: Modify the _handle_client method
    def _handle_client(self, client_sock):
        """Handle communication with a connected client"""
        buffer = b""
        
        while True:
            try:
                data = client_sock.recv(SOCKET_BUFFER_SIZE)
                if not data:
                    logger.info("Connection closed by client")
                    break
                
                buffer += data
                
                # Look for message delimiters
                while b"END_MESSAGE" in buffer:
                    msg, buffer = buffer.split(b"END_MESSAGE", 1)
                    try:
                        # Decode the message
                        msg_data = json.loads(msg.decode('utf-8'))
                        msg_type = msg_data.get('type')
                        
                        if msg_type == 'heartbeat':
                            self._process_heartbeat(msg_data.get('data', {}))
                        elif msg_type == 'image_start':
                            # Prepare to receive image data
                            img_timestamp = msg_data.get('timestamp')
                            img_size = msg_data.get('size')
                            img_name = f"{img_timestamp}.png"
                            img_path = os.path.join(IMAGES_DIR, img_name)
                            
                            # Tell client we're ready to receive chunks
                            client_sock.send(b"START_TRANSFER")
                            
                            # Receive the image data in acknowledged chunks
                            self._receive_image_chunked(client_sock, img_path, img_size)
                            
                            # Now expect classification data
                            client_sock.send(b"READY_FOR_CLASSIFICATION")
                        elif msg_type == 'classification':
                            # Process the classification data
                            self._process_classification(msg_data.get('timestamp'), msg_data.get('data'))
                    except json.JSONDecodeError:
                        logger.error(f"Failed to decode JSON message: {msg}")
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
            
            except Exception as e:
                logger.error(f"Error receiving data: {e}")
                break

    # Add a new method for chunked image reception
    def _receive_image_chunked(self, client_sock, img_path, expected_size):
        """Receive image data in chunks with acknowledgments and save it to disk"""
        logger.info(f"Receiving image of size {expected_size} bytes in chunks")
        
        received_data = bytearray(expected_size)
        total_received = 0
        
        while total_received < expected_size:
            # First read the chunk header
            header_data = b""
            while b":" not in header_data:
                chunk = client_sock.recv(1024)
                if not chunk:
                    raise Exception("Connection closed during image transfer")
                header_data += chunk
                
                if len(header_data) > 100:  # Sanity check - headers shouldn't be this long
                    logger.error("Malformed chunk header")
                    break
            
            if not header_data.startswith(b"CHUNK:"):
                logger.error(f"Invalid header received: {header_data[:50]}")
                continue
            
            # Parse the header
            header_parts = header_data.split(b":", 3)
            if len(header_parts) < 3:
                logger.error("Malformed chunk header")
                continue
                
            chunk_num = int(header_parts[1])
            chunk_size = int(header_parts[2])
            
            # Receive the chunk data
            chunk_data = b""
            bytes_left = chunk_size
            
            while len(chunk_data) < chunk_size:
                bytes_to_read = min(4096, bytes_left)
                chunk = client_sock.recv(bytes_to_read)
                if not chunk:
                    raise Exception("Connection closed during chunk transfer")
                
                chunk_data += chunk
                bytes_left -= len(chunk)
            
            # Insert data at the correct position
            position = chunk_num * 4096
            end_pos = min(position + len(chunk_data), expected_size)
            received_data[position:end_pos] = chunk_data[:end_pos-position]
            
            # Send acknowledgment
            client_sock.send(f"ACK:{chunk_num}".encode('utf-8'))
            
            # Update total received
            total_received += len(chunk_data)
            
            # Log progress less frequently
            if chunk_num % 10 == 0:
                progress = (total_received / expected_size) * 100
                logger.info(f"Image transfer: {progress:.1f}% complete")
        
        # Save the image
        with open(img_path, 'wb') as f:
            f.write(received_data)
        
        logger.info(f"Image saved to {img_path}")
        
        # Store the image info
        timestamp = float(os.path.basename(img_path).split('.')[0])
        img_data = ImageData(timestamp=timestamp, image_path=img_path, classification_data=None)
        
        with self.lock:
            self.latest_image = img_data
            self.image_data.append(img_data)
    
    def _process_heartbeat(self, data):
        """Process and store heartbeat data"""
        heartbeat = Heartbeat.from_dict(data)
        
        with self.lock:
            self.latest_heartbeat = heartbeat
            self.telemetry_data.append(heartbeat)
        
        # Save telemetry to disk
        self._save_telemetry()
        
        logger.info(f"Received heartbeat: time={heartbeat.time}, battery={heartbeat.battery}, health={heartbeat.health}")
    
    def _process_classification(self, timestamp, classification_data):
        """Process and store image classification data"""
        if not timestamp:
            logger.error("Received classification data without timestamp")
            return
        
        # Convert to numpy array
        classification_array = np.array(classification_data)
        
        # Find the corresponding image
        with self.lock:
            for img in self.image_data:
                if abs(img.timestamp - float(timestamp)) < 1.0:  # Allow small time difference
                    img.classification_data = classification_array
                    logger.info(f"Added classification data to image {img.image_path}")
                    break
    
    def _save_telemetry(self):
        """Save telemetry data to disk"""
        with self.lock:
            data = {
                'telemetry': [h.to_dict() for h in self.telemetry_data],
                'last_updated': time.time()
            }
        
        try:
            with open(TELEMETRY_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save telemetry data: {e}")
    
    def get_latest_telemetry(self):
        """Get the latest telemetry data"""
        with self.lock:
            if self.latest_heartbeat:
                return self.latest_heartbeat.to_dict()
            return None
    
    def get_telemetry_history(self, limit=100):
        """Get historical telemetry data"""
        with self.lock:
            return [h.to_dict() for h in self.telemetry_data[-limit:]]
    
    def get_latest_image(self):
        """Get the latest image data"""
        with self.lock:
            if self.latest_image:
                return self.latest_image.to_dict()
            return None
    
    def get_image_history(self, limit=10):
        """Get historical image data"""
        with self.lock:
            return [img.to_dict() for img in self.image_data[-limit:]]
    
    def get_connection_status(self):
        """Get the current connection status"""
        with self.lock:
            return {
                'connected': self.connected,
                'last_heartbeat': self.latest_heartbeat.time if self.latest_heartbeat else None,
                'image_count': len(self.image_data)
            }

# Initialize the ground station
ground_station = GroundStation()

# Create the Flask web application
app = Flask(__name__, static_folder=DATA_DIR)

@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('index.html')

@app.route('/api/status')
def status():
    """Return the connection status"""
    return jsonify(ground_station.get_connection_status())

@app.route('/api/telemetry/latest')
def latest_telemetry():
    """Return the latest telemetry data"""
    data = ground_station.get_latest_telemetry()
    return jsonify(data if data else {})

@app.route('/api/telemetry/history')
def telemetry_history():
    """Return historical telemetry data"""
    limit = request.args.get('limit', default=100, type=int)
    return jsonify(ground_station.get_telemetry_history(limit))

@app.route('/api/image/latest')
def latest_image():
    """Return the latest image data"""
    data = ground_station.get_latest_image()
    return jsonify(data if data else {})

@app.route('/api/image/history')
def image_history():
    """Return historical image data"""
    limit = request.args.get('limit', default=10, type=int)
    return jsonify(ground_station.get_image_history(limit))

@app.route('/images/<path:filename>')
def get_image(filename):
    """Serve image files"""
    return send_from_directory(IMAGES_DIR, filename)

@app.route('/api/mission/summary')
def mission_summary():
    """Return a summary of the mission data"""
    telemetry = ground_station.get_telemetry_history()
    images = ground_station.get_image_history()
    
    # Calculate some basic statistics
    total_images = len(images)
    avg_battery = sum(t['battery'] for t in telemetry) / len(telemetry) if telemetry else 0
    health_counts = {}
    for t in telemetry:
        health = t['health']
        health_counts[health] = health_counts.get(health, 0) + 1
    
    return jsonify({
        'mission_duration': time.time() - telemetry[0]['time'] if telemetry else 0,
        'total_images': total_images,
        'average_battery': avg_battery,
        'health_distribution': health_counts,
        'connection_status': ground_station.get_connection_status()
    })

def main():
    """Main entry point for the ground station application"""
    # Start the Bluetooth server
    if ground_station.start_bluetooth_server():
        logger.info("Bluetooth server started successfully")
    else:
        logger.error("Failed to start Bluetooth server")
    
    # Start the web interface
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

if __name__ == '__main__':
    main()