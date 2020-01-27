import io
import socket
import struct
import time
import picamera
from picamera import PiCamera
import numpy as np
import sys

#CHANGE IP ADDRESS HERE
HOST='192.168.43.103'
PORT=8000

sys.tracebacklimit = 0

client_socket = socket.socket()
client_socket.connect((HOST,PORT))

connection = client_socket.makefile('wb')
try:
    cam_width = 1280
    cam_height = 480

    # Final image capture settings
    scale_ratio = 0.5

    # Camera resolution height must be dividable by 16, and width by 32
    cam_width = int((cam_width+31)/32)*32
    cam_height = int((cam_height+15)/16)*16
    print ("\nUsed camera resolution: "+str(cam_width)+" x "+str(cam_height))

    img_width = int (cam_width * scale_ratio)
    img_height = int (cam_height * scale_ratio)
    print ("Scaled image resolution: "+str(img_width)+" x "+str(img_height)+"\n")

    camera = PiCamera(stereo_mode='side-by-side',stereo_decimate=False)
    camera.resolution=(cam_width, cam_height)
    camera.framerate = 20
    camera.hflip = False
    camera.rotation=0
    
    camera.start_preview()
    time.sleep(2)

    start = time.time()
    stream = io.BytesIO()
    
    for foo in camera.capture_continuous(stream, format="jpeg", use_video_port=True, resize=(img_width,img_height)):
        
        connection.write(struct.pack('<L', stream.tell()))
        connection.flush()
        
        stream.seek(0)
        connection.write(stream.read())
        
        stream.seek(0)
        stream.truncate()
        
    connection.write(struct.pack('<L', 0))

except socket.error as msg:
    print("The connection to the server was lost (if it was not done by pressing Q, check your connectivity).\n")   

finally:
    client_socket.close()
    
