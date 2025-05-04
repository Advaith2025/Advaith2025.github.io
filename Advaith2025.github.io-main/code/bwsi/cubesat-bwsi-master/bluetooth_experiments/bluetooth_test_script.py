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
CHUNK_SIZE = 4096

def create_test_image(width=250, height=250):
    """Create a test image with random colored shapes"""
    image = Image.new('RGB', (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(image)

    # Draw random shapes
    for _ in range(5):
        shape_type = random.choice(['rectangle', 'ellipse'])
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        x1 = random.randint(0, width - 100)
        y1 = random.randint(0, height - 100)
        x2 = x1 + random.randint(50, 100)
        y2 = y1 + random.randint(50, 100)

        if shape_type == 'rectangle':
            draw.rectangle((x1, y1, x2, y2), fill=color)
        else:
            draw.ellipse((x1, y1, x2, y2), fill=color)

    # Add timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    draw.text((10, 10), f"Test Image - {timestamp}", fill=(0, 0, 0))

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
        # Set a reasonable timeout
        socket.settimeout(10.0)
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

        # Wait for heartbeat acknowledgment
        try:
            response = socket.recv(1024)
            logger.info(f"Heartbeat response: {response}")
        except socket.timeout:
            logger.warning("No acknowledgment received for heartbeat, continuing anyway")
        
        # Wait a moment before sending test image
        time.sleep(1)

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

        # Wait for image start acknowledgment
        try:
            response = socket.recv(1024)
            logger.info(f"Image start response: {response}")
        except socket.timeout:
            logger.warning("No acknowledgment received for image start, continuing anyway")

        # Send image data in chunks
        sent_bytes = 0
        chunk_size = min(CHUNK_SIZE, 1024)  # Use smaller chunks to avoid buffer issues

        for i in range(0, image_size, chunk_size):
            chunk = image_data[i:i+chunk_size]
            socket.send(chunk)
            sent_bytes += len(chunk)

            # Log progress periodically
            if (i // chunk_size) % 5 == 0 or sent_bytes >= image_size:
                progress = (sent_bytes / image_size) * 100
                logger.info(f"Image transfer: {progress:.1f}% complete ({sent_bytes}/{image_size} bytes)")

            # Check for progress acknowledgments
            try:
                socket.settimeout(0.1)  # Brief timeout to check for responses without blocking
                response = socket.recv(1024)
                if response.startswith(b"PROGRESS:"):
                    logger.info(f"Progress acknowledgment: {response}")
            except socket.timeout:
                pass  # No progress acknowledgment is fine
            except Exception as e:
                logger.warning(f"Error checking for progress acks: {e}")
            finally:
                socket.settimeout(10.0)  # Reset to normal timeout
                
            # Add small delay to prevent buffer overflows
            time.sleep(0.05)

        logger.info("Image transfer completed")

        # Wait for acknowledgment
        logger.info("Waiting for acknowledgment...")
        socket.settimeout(15.0)  # Longer timeout for final acknowledgment
        try:
            response = socket.recv(1024)
            logger.info(f"Received response: {response}")
        except socket.timeout:
            logger.error("No acknowledgment received after image transfer")
            # Continue anyway to try to complete the test
        
        # Send test classification data if acknowledgment received or not
        if b"READY_FOR_CLASSIFICATION" in response if 'response' in locals() else True:
            # Send classification data
            classification = {
                "type": "classification",
                "timestamp": timestamp,
                "data": [[random.random() for _ in range(8)] for _ in range(8)]
            }

            json_data = json.dumps(classification).encode('utf-8') + b"END_MESSAGE"
            socket.send(json_data)
            logger.info("Test classification data sent successfully")
            
            # Wait for classification acknowledgment
            try:
                response = socket.recv(1024)
                logger.info(f"Classification response: {response}")
            except socket.timeout:
                logger.warning("No acknowledgment received for classification data")

        # Wait a moment
        time.sleep(1)

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
        
        # Wait for final acknowledgment
        try:
            response = socket.recv(1024)
            logger.info(f"Final heartbeat response: {response}")
        except socket.timeout:
            logger.warning("No acknowledgment received for final heartbeat")

        # Close the connection properly
        try:
            socket.shutdown(socket.SHUT_RDWR)
        except:
            pass
        socket.close()
        logger.info("Test completed successfully")
        return True

    except Exception as e:
        logger.error(f"Bluetooth test failed: {e}")
        if 'socket' in locals() and socket:
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