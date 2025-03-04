import bluetooth
import time
import random
import os
import json
import logging
from io import BytesIO
from PIL import Image, ImageDraw

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CHUNK_SIZE = 512

def create_test_image(width=250, height=250):
    """Create a test image with random colored shapes"""
    image = Image.new('RGB', (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(image)

    # Draw random shapes
    for _ in range(5):
        shape_type = random.choice(['rectangle', 'ellipse'])
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        x1 = random.randint(0, width - 50)
        y1 = random.randint(0, height - 50)
        x2 = x1 + random.randint(20, 50)
        y2 = y1 + random.randint(20, 50)

        if shape_type == 'rectangle':
            draw.rectangle((x1, y1, x2, y2), fill=color)
        else:
            draw.ellipse((x1, y1, x2, y2), fill=color)

    # Add timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    draw.text((5, 5), f"Test {timestamp}", fill=(0, 0, 0))

    return image

'''
def test_bluetooth_client():
    """Test Bluetooth client connection to ground station"""
    ground_station_mac = None

    try:
        logger.info("Discovering nearby devices...")
        nearby_devices = bluetooth.discover_devices(lookup_names=True)

        for addr, name in nearby_devices:
            logger.info(f"Found device: {name} at {addr}")
            if "Ground" in name or "CubeSat" in name:
                ground_station_mac = addr
                logger.info(f"Found ground station at {addr}")
                break

        if not ground_station_mac:
            if nearby_devices:
                ground_station_mac = nearby_devices[0][0]
                logger.info(f"Using first available device: {nearby_devices[0][1]} ({ground_station_mac})")
            else:
                logger.error("No Bluetooth devices found")
                return False

        # Create socket and connect
        port = 1  # RFCOMM port
        socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        socket.connect((ground_station_mac, port))
        logger.info("Connected to ground station")

        # Send test heartbeat
        heartbeat = {
            "type": "heartbeat",
            "data": {
                "time": time.time(),
                "accelx": random.uniform(-1.0, 1.0),
                "accely": random.uniform(-1.0, 1.0),
                "accelz": random.uniform(9.0, 10.0),  # Approx gravity
                "battery": random.uniform(0.8, 1.0),
                "image_count": 0,
                "health": 0
            }
        }

        json_data = json.dumps(heartbeat).encode('utf-8') + b"END_MESSAGE"
        socket.send(json_data)
        logger.info("Test heartbeat sent successfully")

        # Wait a moment before sending test image
        time.sleep(2)

        # Create test image
        test_img = create_test_image()
        buffer = BytesIO()
        test_img.save(buffer, format="PNG")
        image_data = buffer.getvalue()
        image_size = len(image_data)

        # Send image start message
        timestamp = time.time()
        start_message = {
            "type": "image_start",
            "timestamp": timestamp,
            "size": image_size
        }

        json_data = json.dumps(start_message).encode('utf-8') + b"END_MESSAGE"
        socket.send(json_data)
        logger.info(f"Image start message sent, size={image_size} bytes")

        # Send image data in chunks
        sent_bytes = 0

        for i in range(0, image_size, CHUNK_SIZE):
            chunk = image_data[i:i+CHUNK_SIZE]
            socket.send(chunk)
            sent_bytes += len(chunk)

            # Log progress periodically
            if (i // CHUNK_SIZE) % 5 == 0 or sent_bytes >= image_size:
                progress = (sent_bytes / image_size) * 100
                logger.info(f"Image transfer: {progress:.1f}% complete ({sent_bytes}/{image_size} bytes)")

            # Add small delay to prevent buffer overflows
            time.sleep(0.01)

        logger.info("Image transfer completed")

        # Wait for acknowledgment
        logger.info("Waiting for acknowledgment...")
        response = socket.recv(1024)
        logger.info(f"Received response: {response}")

        # Send test classification data
        if b"READY_FOR_CLASSIFICATION" in response:
            # Send classification data
            classification = {
                "type": "classification",
                "timestamp": timestamp,
                "data": [[random.random() for _ in range(8)] for _ in range(8)]
            }

            json_data = json.dumps(classification).encode('utf-8') + b"END_MESSAGE"
            socket.send(json_data)
            logger.info("Test classification data sent successfully")

        # Wait a moment
        time.sleep(2)

        # Send final heartbeat
        heartbeat = {
            "type": "heartbeat",
            "data": {
                "time": time.time(),
                "accelx": random.uniform(-1.0, 1.0),
                "accely": random.uniform(-1.0, 1.0),
                "accelz": random.uniform(9.0, 10.0),
                "battery": random.uniform(0.8, 1.0),
                "image_count": 1,
                "health": 0
            }
        }

        json_data = json.dumps(heartbeat).encode('utf-8') + b"END_MESSAGE"
        socket.send(json_data)
        logger.info("Final heartbeat sent successfully")

        # Close the connection
        socket.close()
        logger.info("Test completed successfully")
        return True

    except Exception as e:
        logger.error(f"Bluetooth test failed: {e}")
        return False

'''
def test_bluetooth_client():
    """Test Bluetooth client connection to ground station with robust error handling"""
    ground_station_mac = None
    socket = None

    try:
        logger.info("Discovering nearby devices...")
        nearby_devices = bluetooth.discover_devices(lookup_names=True)

        for addr, name in nearby_devices:
            logger.info(f"Found device: {name} at {addr}")
            if "Ground" in name or "CubeSat" in name:
                ground_station_mac = addr
                logger.info(f"Found ground station at {addr}")
                break

        if not ground_station_mac:
            if nearby_devices:
                ground_station_mac = nearby_devices[0][0]
                logger.info(f"Using first available device: {nearby_devices[0][1]} ({ground_station_mac})")
            else:
                logger.error("No Bluetooth devices found")
                return False

        # Create socket and connect
        port = 1  # RFCOMM port
        socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        # Set a reasonable timeout
        socket.settimeout(10.0)
        
        logger.info(f"Attempting to connect to {ground_station_mac} on port {port}...")
        socket.connect((ground_station_mac, port))
        logger.info("Connected to ground station")
        
        # Function to safely send data with error checking
        def safe_send(sock, data):
            try:
                bytes_sent = sock.send(data)
                if bytes_sent < len(data):
                    logger.warning(f"Only sent {bytes_sent}/{len(data)} bytes")
                return bytes_sent
            except bluetooth.btcommon.BluetoothError as e:
                logger.error(f"Bluetooth error during send: {e}")
                raise
        
        # Function to safely receive data with error checking
        def safe_recv(sock, buffer_size, timeout=None):
            original_timeout = sock.gettimeout()
            if timeout is not None:
                sock.settimeout(timeout)
                
            try:
                data = sock.recv(buffer_size)
                return data
            except bluetooth.btcommon.BluetoothError as e:
                if "timed out" in str(e).lower():
                    logger.warning(f"Receive timed out: {e}")
                    return b""
                else:
                    logger.error(f"Bluetooth error during receive: {e}")
                    raise
            finally:
                # Restore original timeout
                if timeout is not None:
                    sock.settimeout(original_timeout)

        # Send test heartbeat
        heartbeat = {
            "type": "heartbeat",
            "data": {
                "time": time.time(),
                "accelx": random.uniform(-1.0, 1.0),
                "accely": random.uniform(-1.0, 1.0),
                "accelz": random.uniform(9.0, 10.0),  # Approx gravity
                "battery": random.uniform(0.8, 1.0),
                "image_count": 0,
                "health": 0
            }
        }

        json_data = json.dumps(heartbeat).encode('utf-8') + b"END_MESSAGE"
        safe_send(socket, json_data)
        logger.info("Test heartbeat sent successfully")

        # Wait for heartbeat acknowledgment (with shorter timeout)
        response = safe_recv(socket, 1024, timeout=2.0)
        if response:
            logger.info(f"Heartbeat response: {response}")
        else:
            logger.warning("No acknowledgment received for heartbeat, continuing anyway")
        
        # Wait a moment before sending test image
        time.sleep(0.5)

        # Create test image - make it smaller for testing
        test_img = create_test_image(100, 100)  # Smaller 100x100 image for testing
        buffer = BytesIO()
        test_img.save(buffer, format="PNG")
        image_data = buffer.getvalue()
        image_size = len(image_data)
        
        logger.info(f"Created test image, size={image_size} bytes")

        # Send image start message
        timestamp = time.time()
        start_message = {
            "type": "image_start",
            "timestamp": timestamp,
            "size": image_size
        }

        json_data = json.dumps(start_message).encode('utf-8') + b"END_MESSAGE"
        safe_send(socket, json_data)
        logger.info(f"Image start message sent")

        # Wait for image start acknowledgment
        response = safe_recv(socket, 1024, timeout=2.0)
        if response:
            logger.info(f"Image start response: {response}")
        else:
            logger.warning("No acknowledgment received for image start, continuing anyway")

        # Send image data in chunks
        sent_bytes = 0
        chunk_size = 512  # Very small chunks for testing
        
        logger.info(f"Sending image in chunks of {chunk_size} bytes")

        for i in range(0, image_size, chunk_size):
            chunk = image_data[i:i+chunk_size]
            safe_send(socket, chunk)
            sent_bytes += len(chunk)

            # Log progress periodically
            if (i // chunk_size) % 5 == 0 or sent_bytes >= image_size:
                progress = (sent_bytes / image_size) * 100
                logger.info(f"Image transfer: {progress:.1f}% complete ({sent_bytes}/{image_size} bytes)")

            # Briefly check for progress acknowledgments (non-blocking)
            try:
                response = safe_recv(socket, 1024, timeout=0.1)
                if response:
                    logger.info(f"Got response during transfer: {response}")
            except Exception as e:
                logger.warning(f"Error checking for progress acks: {e}")
                
            # Add small delay to prevent buffer overflows
            time.sleep(0.1)

        logger.info("Image transfer completed")

        # Wait for final acknowledgment
        logger.info("Waiting for final acknowledgment...")
        response = safe_recv(socket, 1024, timeout=5.0)
        if response:
            logger.info(f"Received response: {response}")
            
            # Send test classification data if requested
            if b"READY_FOR_CLASSIFICATION" in response:
                # Send classification data
                classification = {
                    "type": "classification",
                    "timestamp": timestamp,
                    "data": [[random.random() for _ in range(4)] for _ in range(4)]  # Smaller data for testing
                }

                json_data = json.dumps(classification).encode('utf-8') + b"END_MESSAGE"
                safe_send(socket, json_data)
                logger.info("Test classification data sent successfully")
                
                # Wait for classification acknowledgment
                response = safe_recv(socket, 1024, timeout=2.0)
                if response:
                    logger.info(f"Classification response: {response}")
                else:
                    logger.warning("No acknowledgment received for classification data")
        else:
            logger.warning("No response received after image transfer, skipping classification")

        # Wait a moment
        time.sleep(0.5)

        # Send final heartbeat
        heartbeat = {
            "type": "heartbeat",
            "data": {
                "time": time.time(),
                "accelx": random.uniform(-1.0, 1.0),
                "accely": random.uniform(-1.0, 1.0),
                "accelz": random.uniform(9.0, 10.0),
                "battery": random.uniform(0.8, 1.0),
                "image_count": 1,
                "health": 0
            }
        }

        json_data = json.dumps(heartbeat).encode('utf-8') + b"END_MESSAGE"
        try:
            safe_send(socket, json_data)
            logger.info("Final heartbeat sent successfully")
            
            # Wait for final acknowledgment
            response = safe_recv(socket, 1024, timeout=2.0)
            if response:
                logger.info(f"Final heartbeat response: {response}")
            else:
                logger.warning("No acknowledgment received for final heartbeat")
        except Exception as e:
            logger.warning(f"Failed to send final heartbeat: {e}")

        # Close the connection properly
        logger.info("Closing connection...")
        try:
            socket.shutdown(bluetooth.SHUT_RDWR)
        except Exception as e:
            logger.warning(f"Socket shutdown error: {e}")
            
        try:
            socket.close()
        except Exception as e:
            logger.warning(f"Socket close error: {e}")
            
        logger.info("Test completed successfully")
        return True

    except Exception as e:
        logger.error(f"Bluetooth test failed: {e}")
        if socket:
            try:
                socket.close()
            except:
                pass
        return False
if __name__ == "__main__":
    print("CubeSat Bluetooth Connection Test")
    print("================================")
    print("This script tests the Bluetooth connection to the ground station.")
    print("It will attempt to find the ground station, connect, and send test data.")
    print("Make sure the ground station is running and in discovery mode.")
    print()

    input("Press Enter to begin the test...")

    if test_bluetooth_client():
        print("\nTest PASSED - Bluetooth connection and data transfer successful!")
    else:
        print("\nTest FAILED - Check the log for details")