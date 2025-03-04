"""
The Python code you will write for this module should read
acceleration data from the IMU. When a reading comes in that surpasses
an acceleration threshold (indicating a shake), your Pi should pause,
trigger the camera to take a picture, then save the image with a
descriptive filename. You may use GitHub to upload your images automatically,
but for this activity it is not required.

The provided functions are only for reference, you do not need to use them. 
You will need to complete the take_photo() function and configure the VARIABLES section
"""

#AUTHOR: 
#DATE:

#import libraries
import time
import board
from adafruit_lsm6ds.lsm6dsox import LSM6DSOX as LSM6DS
from adafruit_lis3mdl import LIS3MDL
from git import Repo
from picamera2 import Picamera2, Preview

#VARIABLES
THRESHOLD = 2.0      #Any desired value from the accelerometer
REPO_PATH = "/home/cubesat/CubeSat"     #Your github repo path: ex. /home/pi/FlatSatChallenge
FOLDER_PATH = "/Images"   #Your image folder path in your GitHub repo: ex. /Images
CAPTURE_COOLDOWN = 5.0

#imu and camera initialization
i2c = board.I2C()
accel_gyro = LSM6DS(i2c)
mag = LIS3MDL(i2c)
picam2 = Picamera2()

picam2.start_preview(Preview.DRM)
capture_config = picam2.create_still_configuration()
picam2.configure(capture_config)
picam2.start()

time.sleep(2)

def git_push():
    """
    This function is complete. Stages, commits, and pushes new images to your GitHub repo.
    """
    try:
        repo = Repo(REPO_PATH)
        origin = repo.remote('origin')
        print('added remote')
        origin.pull()
        print('pulled changes')
        repo.git.add(REPO_PATH + FOLDER_PATH)
        repo.index.commit('New Photo')
        print('made the commit')
        origin.push()
        print('pushed changes')
    except Exception as e:
        print('Couldn\'t upload to git', e)


def img_gen(name):
    """
    This function is complete. Generates a new image name.

    Parameters:
        name (str): your name ex. MasonM
    """
    t = time.strftime("_%Y-%m-%d-%H%M%S")
    imgname = (f'{REPO_PATH}/{FOLDER_PATH}/{name}{t}.jpg')
    return imgname


def take_photo():
    """
    This function is NOT complete. Takes a photo when the FlatSat is shaken.
    Replace psuedocode with your own code.
    """
    last_captured = 0
    while True:
        try:
            accelx, accely, accelz = accel_gyro.acceleration

            current_time = time.time()
            #CHECKS IF READINGS ARE ABOVE THRESHOLD
            # check if squared acceleration is greater than squared threshold
            print(f'accelx: {accelx}, accely: {accely}, accelz: {accelz}')

            if current_time - last_captured > CAPTURE_COOLDOWN:
                print("capturing image...")
                #PAUSE
                name = "HariA"     #First Name, Last Initial  ex. MasonM
                img_path = img_gen(name)
                #TAKE PHOTO
                print("before capture")
                picam2.capture_file(img_path)
                print(f"After capture with path {img_path}")
                #PUSH PHOTO TO GITHUB
                git_push()
                print("After push")
                last_captured = current_time
            #PAUSE
            time.sleep(0.1)

        except KeyboardInterrupt:
            print('Interrupted. Exiting...')
            picam2.stop()
            break
        except Exception as e:
            print("[ERROR] ",e)
            picam2.stop()
            break


def main():
    take_photo()


if __name__ == '__main__':
    main()

