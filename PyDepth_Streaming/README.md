# PyDepth Project Streaming Interface
The whole project relies on the stereo images broadcasted by the StereoPi device. The scripts here open a server-client communication stream where the StereoPi is the client and broadcasts images to the computer.

The scripts work for Windows (the .bat file), MacOS and Linux (the .sh file). There is no need to run the .py files, everything is taken care of by the shell scripts. The clientSocket.py file is already in the StereoPi. It is important to have a folder called Saved_Images in the same directory as the code, server-side.

The scripts work for Windows (the .bat file), MacOS and Linux (the .sh file). There is no need to run the .py files, everything is taken care of by the shell scripts. The clientSocket.py file is already in the StereoPi.
It is very important to have the folder Saved_Images in the same directory as the server programme and the shell scripts. Every ten images, an image is going to be saved there in the format of .npy, ready to be sent through the neural network.

## Prerequirements
To run these scripts you will need to install OpenCV-python by running ```pip install opencv-python``` in the command line. 
On windows specifically you need to install PuTTy, which comes with an additionnal module called plink. Be certain that the PATH variables are well set (other necessary library is numpy, but ).
For Linux and MacOS, just need to have ssh installed.

Before using the stereopi, we need to make sure that every device is connected to the same local network. When this is done we have to find the ip address of the computer used as server. On linux the easiest is to type ```hostname -I``` into the terminal. For macOS the command is ```ifconfig``` and Windows you will have to use ```ipconfig```. With the ip address we can now configure the addresses hardcoded in the python scripts both on server and client side.

For the one in the StereoPi, you will need to ssh in with ```ssh pi@pydepth.local``` (or ```putty.exe -ssh pi@pydepth.local``` on windows) and modify the hardcoded ip address in the script. It is located in the directory ```~/PyDepth_Client/``` . You can use nano to modify the file.

That is all there is to do!
