import bluetooth
import time
import random
import os
import json
import logging
import sys
from io import BytesIO
from PIL import Image, ImageDraw

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CHUNK_SIZE = 512  # Smaller chunks for better reliability

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

def find_ground_station():
    """Find the ground station using proper service discovery"""
    logger.info("Searching for CubeSat Ground Station service...")
    
    service_matches = bluetooth.find_service(name="CubeSatGroundStation", uuid="94f39d29-7d6d-437d-973b-fba39e49d4ee")
    
    if not service_matches:
        logger.info("No ground station service found. Searching for nearby devices...")
        nearby_devices = bluetooth.discover_devices(lookup_names=True, duration=8)
        
        for addr, name in nearby_devices:
            logger.info(f"Found device: {name} at {addr}")
            if "Ground" in name or "CubeSat" in name:
                logger.info(f"Found potential ground station: {name}")
                return addr, 1  # Default RFCOMM port
        
        if nearby_devices:
            # Ask user if they want to try the first device
            device_addr, device_name = nearby_devices[0]
            logger.info(f"No ground station found. Would you like to try connecting to {device_name} ({device_addr})?")
            response = input("Connect to this device? (y/n): ").strip().lower()
            if response == 'y':
                return device_addr, 1
            else:
                return None, None
        else:
            logger.error("No Bluetooth devices found")
            return None, None
    else:
        # Found the service
        service = service_matches[0]
        logger.info(f"Found ground station service on {service['host']} port {service['port']}")
        return service['host'], service['port']

def test_bluetooth_client():
    """Test Bluetooth client connection to ground station with robust error handling"""
    socket = None

    try:
        # Find the ground station
        addr, port = find_ground_station()
        if not addr:
            logger.error("Could not find a ground station to connect to")
            return False
            
        logger.info(f"Attempting to connect to {addr} on port {port}...")
        
        # Create socket and connect
        socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        socket.settimeout(10.0)
        
        try:
            socket.connect((addr, port))
            logger.info("Connected to ground station")
        except bluetooth.btcommon.BluetoothError as e:
            logger.error(f"Connection failed: {e}")
            # Try one more time with a different port if the first attempt fails
            if "Connection refused" in str(e) and port == 1:
                logger.info("Trying alternative port (port 1)...")
                try:
                    socket.close()
                    socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
                    socket.settimeout(10.0)
                    socket.connect((addr, 1))
                    port = 1
                    logger.info("Connected to ground station on alternative port")
                except bluetooth.btcommon.BluetoothError as e2:
                    logger.error(f"Second connection attempt failed: {e2}")
                    return False
            else:
                return False
        
        # Function to safely send data with error checking and retry
        def safe_send(sock, data, retries=3):
            for attempt in range(retries):
                try:
                    bytes_sent = sock.send(data)
                    if bytes_sent < len(data):
                        logger.warning(f"Only sent {bytes_sent}/{len(data)} bytes, retrying...")
                        # Send the remainder
                        return bytes_sent + safe_send(sock, data[bytes_sent:], retries-1)
                    return bytes_sent
                except bluetooth.btcommon.BluetoothError as e:
                    if attempt < retries - 1:
                        logger.warning(f"Send error (attempt {attempt+1}/{retries}): {e}")
                        time.sleep(0.5)
                    else:
                        logger.error(f"Failed to send data after {retries} attempts: {e}")
                        raise
        
        # Function to safely receive data with error checking
        def safe_recv(sock, buffer_size, timeout=None, expected_response=None, max_attempts=5):
            original_timeout = sock.gettimeout()
            if timeout is not None:
                sock.settimeout(timeout)
                
            data = b""
            attempts = 0
            
            try:
                while attempts < max_attempts:
                    try:
                        chunk = sock.recv(buffer_size)
                        if chunk:
                            data += chunk
                            logger.debug(f"Received {len(chunk)} bytes")
                            
                            # If we have a specific response we're looking for, check if we got it
                            if expected_response and expected_response in data:
                                return data
                            
                            # If no specific response expected, just return what we got
                            if not expected_response:
                                return data
                                
                        else:
                            logger.warning("Received empty data")
                            attempts += 1
                            time.sleep(0.5)
                            
                    except bluetooth.btcommon.BluetoothError as e:
                        if "timed out" in str(e).lower():
                            logger.warning(f"Receive timed out (attempt {attempts+1}/{max_attempts})")
                            attempts += 1
                            if attempts >= max_attempts:
                                logger.error("Maximum receive attempts reached")
                                return data if data else b""
                        else:
                            logger.error(f"Bluetooth error during receive: {e}")
                            raise
                
                # If we got here without returning, we've hit max attempts
                return data if data else b""
                
            finally:
                # Restore original timeout
                if timeout is not None and sock:
                    try:
                        sock.settimeout(original_timeout)
                    except:
                        pass

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

        logger.info("Sending test heartbeat...")
        json_data = json.dumps(heartbeat).encode('utf-8') + b"END_MESSAGE"
        safe_send(socket, json_data)
        logger.info("Test heartbeat sent successfully")

        # Wait for heartbeat acknowledgment
        logger.info("Waiting for heartbeat acknowledgment...")
        response = safe_recv(socket, 1024, timeout=5.0, expected_response=b"HEARTBEAT_ACK")
        if response:
            logger.info(f"Heartbeat response: {response}")
        else:
            logger.warning("No acknowledgment received for heartbeat, continuing anyway")
        
        # Wait a moment before sending test image
        time.sleep(1.0)

        # Create test image - use a very small image for testing
        test_img = create_test_image(100, 100)
        buffer = BytesIO()
        test_img.save(buffer, format="JPEG", quality=70)  # Use JPEG with lower quality for smaller size
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

        logger.info("Sending image start message...")
        json_data = json.dumps(start_message).encode('utf-8') + b"END_MESSAGE"
        safe_send(socket, json_data)
        logger.info(f"Image start message sent")

        # Wait for image start acknowledgment
        logger.info("Waiting for image start acknowledgment...")
        response = safe_recv(socket, 1024, timeout=5.0, expected_response=b"IMAGE_START_ACK")
        if response:
            logger.info(f"Image start response: {response}")
        else:
            logger.warning("No acknowledgment received for image start, continuing anyway")

        # Send image data in chunks
        sent_bytes = 0
        chunk_size = CHUNK_SIZE  # Use smaller chunks for more reliable transfer
        
        logger.info(f"Sending image in chunks of {chunk_size} bytes...")

        # Add a small delay before starting image transfer
        time.sleep(0.5)

        for i in range(0, image_size, chunk_size):
            chunk = image_data[i:i+chunk_size]
            bytes_sent = safe_send(socket, chunk)
            sent_bytes += bytes_sent

            # Log progress periodically
            if (i // chunk_size) % 5 == 0 or sent_bytes >= image_size:
                progress = (sent_bytes / image_size) * 100
                logger.info(f"Image transfer: {progress:.1f}% complete ({sent_bytes}/{image_size} bytes)")

            # Briefly check for progress acknowledgments (non-blocking)
            try:
                response = safe_recv(socket, 1024, timeout=0.1)
                if response and b"PROGRESS" in response:
                    logger.info(f"Got progress update: {response}")
            except Exception as e:
                logger.debug(f"No progress ack received: {e}")
                
            # Add small delay between chunks to prevent buffer overflows
            time.sleep(0.2)

        logger.info("Image transfer completed")

        # Wait for final acknowledgment
        logger.info("Waiting for final acknowledgment...")
        response = safe_recv(socket, 1024, timeout=10.0, expected_response=b"READY_FOR_CLASSIFICATION")
        if response:
            logger.info(f"Received response: {response}")
            
            # Send test classification data if requested
            if b"READY_FOR_CLASSIFICATION" in response:
                time.sleep(0.5)  # Brief pause before sending classification
                
                # Send classification data
                classification = {
                    "type": "classification",
                    "timestamp": timestamp,
                    "data": [[random.random() for _ in range(4)] for _ in range(4)]
                }

                logger.info("Sending classification data...")
                json_data = json.dumps(classification).encode('utf-8') + b"END_MESSAGE"
                safe_send(socket, json_data)
                logger.info("Test classification data sent successfully")
                
                # Wait for classification acknowledgment
                response = safe_recv(socket, 1024, timeout=5.0, expected_response=b"CLASSIFICATION_ACK")
                if response:
                    logger.info(f"Classification response: {response}")
                else:
                    logger.warning("No acknowledgment received for classification data")
        else:
            logger.warning("No response received after image transfer, skipping classification")

        # Wait a moment
        time.sleep(1.0)

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

        logger.info("Sending final heartbeat...")
        json_data = json.dumps(heartbeat).encode('utf-8') + b"END_MESSAGE"
        try:
            safe_send(socket, json_data)
            logger.info("Final heartbeat sent successfully")
            
            # Wait for final acknowledgment
            response = safe_recv(socket, 1024, timeout=5.0)
            if response:
                logger.info(f"Final heartbeat response: {response}")
            else:
                logger.warning("No acknowledgment received for final heartbeat")
        except Exception as e:
            logger.warning(f"Failed to send final heartbeat: {e}")

        # Close the connection properly
        logger.info("Closing connection...")
        try:
            # Send a graceful disconnect message
            socket.send(b"DISCONNECT")
            time.sleep(0.5)
        except:
            pass
            
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