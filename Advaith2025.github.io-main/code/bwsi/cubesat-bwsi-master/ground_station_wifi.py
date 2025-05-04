import os
import time
import numpy as np
import socket
import json
import threading
import logging
from datetime import datetime
from dataclasses import dataclass
from flask import Flask, render_template, jsonify, Response, request, send_from_directory

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
TCP_IP = '0.0.0.0'  # Listen on all interfaces
TCP_PORT = 5555
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
        self.client_address = None
        self.socket = None
        self.server_thread = None
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
    
    def start_server(self):
        """Start the TCP server to listen for CubeSat connections"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((TCP_IP, TCP_PORT))
            self.socket.listen(1)
            
            logger.info(f"TCP Server started on {TCP_IP}:{TCP_PORT}")
            
            # Start a thread to accept connections
            self.server_thread = threading.Thread(target=self._server_listen)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            return True
        except Exception as e:
            logger.error(f"Failed to start TCP server: {e}")
            return False
    
    def _server_listen(self):
        """Listen for and handle TCP connections"""
        try:
            while True:
                logger.info("Waiting for connection...")
                client_sock, client_addr = self.socket.accept()
                logger.info(f"Accepted connection from {client_addr}")
                
                with self.lock:
                    self.client_address = client_addr
                    self.connected = True
                
                try:
                    self._handle_client(client_sock)
                except Exception as e:
                    logger.error(f"Error handling client: {e}")
                finally:
                    with self.lock:
                        self.connected = False
                        self.client_address = None
        except Exception as e:
            logger.error(f"Server listener error: {e}")
        finally:
            if self.socket:
                self.socket.close()
    
    def _receive_data_with_length_prefix(self, sock):
        """Receive data with length prefix for reliable transmission"""
        try:
            # First receive the length prefix (4 bytes)
            length_bytes = b''
            while len(length_bytes) < 4:
                chunk = sock.recv(4 - len(length_bytes))
                if not chunk:
                    raise ConnectionError("Connection closed while receiving length prefix")
                length_bytes += chunk
            
            # Convert length bytes to integer
            data_length = int.from_bytes(length_bytes, byteorder='big')
            
            # Now receive the actual data
            data = b''
            bytes_received = 0
            
            while bytes_received < data_length:
                bytes_to_receive = min(SOCKET_BUFFER_SIZE, data_length - bytes_received)
                chunk = sock.recv(bytes_to_receive)
                
                if not chunk:
                    raise ConnectionError(f"Connection closed after receiving {bytes_received}/{data_length} bytes")
                
                data += chunk
                bytes_received += len(chunk)
                
                # Log progress for large data transfers
                if data_length > 1000000 and bytes_received % 1000000 < SOCKET_BUFFER_SIZE:
                    progress = (bytes_received / data_length) * 100
                    logger.info(f"Receiving data: {progress:.1f}% complete ({bytes_received}/{data_length} bytes)")
            
            return data
        except Exception as e:
            logger.error(f"Error receiving data: {e}")
            raise
    
    def _handle_client(self, client_sock):
        """Handle communication with a connected client"""
        client_sock.settimeout(60)  # 60 second timeout
        
        try:
            while True:
                try:
                    # Receive message type - a single byte
                    type_byte = client_sock.recv(1)
                    if not type_byte:
                        logger.info("Connection closed by client (no data)")
                        break
                    
                    msg_type = type_byte[0]
                    
                    if msg_type == 1:  # Heartbeat
                        # Receive heartbeat data
                        data = self._receive_data_with_length_prefix(client_sock)
                        try:
                            heartbeat_data = json.loads(data.decode('utf-8'))
                            self._process_heartbeat(heartbeat_data)
                            # Send acknowledgment
                            client_sock.send(b'\x01')  # ACK code
                        except json.JSONDecodeError:
                            logger.error(f"Failed to decode heartbeat JSON")
                    
                    elif msg_type == 2:  # Image
                        # Receive image timestamp
                        data = self._receive_data_with_length_prefix(client_sock)
                        timestamp = float(data.decode('utf-8'))
                        
                        # Send acknowledgment for timestamp
                        client_sock.send(b'\x01')  # ACK code
                        
                        # Receive image data
                        image_data = self._receive_data_with_length_prefix(client_sock)

                        # Save image to file
                        img_name = f"{timestamp}.png"
                        img_path = os.path.join(IMAGES_DIR, img_name)
                        # img_path = img_name
                        
                        with open(img_path, 'wb') as f:
                            f.write(image_data)
                        
                        logger.info(f"Image saved to {img_path}")
                        
                        # Store image info
                        img_data = ImageData(timestamp=timestamp, image_path=img_name, classification_data=None)
                        
                        with self.lock:
                            self.latest_image = img_data
                            self.image_data.append(img_data)
                        
                        # Send acknowledgment for image data
                        client_sock.send(b'\x01')  # ACK code
                    
                    elif msg_type == 3:  # Classification data
                        # Receive timestamp
                        data = self._receive_data_with_length_prefix(client_sock)
                        timestamp = float(data.decode('utf-8'))
                        
                        # Send acknowledgment for timestamp
                        client_sock.send(b'\x01')  # ACK code
                        
                        # Receive classification data
                        data = self._receive_data_with_length_prefix(client_sock)
                        classification_data = json.loads(data.decode('utf-8'))
                        
                        # Process classification data
                        self._process_classification(timestamp, classification_data)
                        
                        # Send acknowledgment for classification data
                        client_sock.send(b'\x01')  # ACK code
                    
                    else:
                        logger.warning(f"Unknown message type: {msg_type}")
                
                except socket.timeout:
                    logger.debug("Socket timeout while waiting for data")
                    # Send a ping to check if connection is still alive
                    try:
                        client_sock.send(b'\x00')  # Ping
                    except:
                        logger.error("Failed to send ping, connection may be lost")
                        break
                
                except ConnectionError as e:
                    logger.error(f"Connection error: {e}")
                    break
                
                except Exception as e:
                    logger.error(f"Error receiving data: {e}", exc_info=True)
        
        except Exception as e:
            logger.error(f"Client handler fatal error: {e}", exc_info=True)
        
        finally:
            logger.info("Closing client connection")
            try:
                client_sock.close()
            except:
                pass

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
                'client': str(self.client_address) if self.client_address else None,
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
    # Start the TCP server
    if ground_station.start_server():
        logger.info("TCP server started successfully")
    else:
        logger.error("Failed to start TCP server")
    
    # Start the web interface
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

if __name__ == '__main__':
    main()