import socket
import sys
import cv2
import pickle
import numpy as np
import struct
import io
import socket
import struct
import subprocess

#CHANGE IP ADDRESS HERE
HOST='192.168.43.103'
PORT=8000


print("Welcome to the PyDepth streaming interface\n")
print("You can quit the stream at any moment by pushing on the Q key of your keyboard\n")

print("We are now logging into PyDepth:")

server_socket = socket.socket()
server_socket.bind((HOST,PORT))
server_socket.listen(0)

connection = server_socket.accept()[0].makefile('rb')

cv2.namedWindow("left")
cv2.moveWindow("left", 850,100)
cv2.namedWindow("right")
cv2.moveWindow("right", 450,100)

try:
	x = 0
	while True:
		image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
		if not image_len:
			break
			
		image_stream = io.BytesIO()
		image_stream.write(connection.read(image_len))
		
		image_stream.seek(0)
		
		#Way of converting from BytesIO to a cv2 understandable format: (depracated)
		#img = cv2.imdecode(np.fromstring(image_stream.read(), np.uint8), 1)
		
		#Another way of converting from BytesIO to a cv2 understandable format: 
		file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
		img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
		
		height, width = img.shape[:2]
		
		imgRight = img [0:height,0:int(width/2)]
		imgLeft = img [0:height,int(width/2):width]
		
		cv2.imshow("left",imgLeft)
		cv2.imshow("right",imgRight)
		key = cv2.waitKey(1) & 0xFF
			
		if x == 10:
			np.save("Saved_Images/streamLeft.npy",imgLeft)
			np.save("Saved_Images/streamRight.npy",imgRight)
			x = 0
		x+=1
		
		if key == ord("q"):
			break
		
		
finally:
    connection.close()
    server_socket.close()
