# PyDepth Project Streaming Interface
The whole project relies on the stereo images broadcasted by the StereoPi device. The scripts here open a server-client communication stream where the StereoPi is the client and broadcasts images to the computer.

The scripts work for Windows (the .bat file), MacOS and Linux (the .sh file). There is no need to run the .py files, everything is taken care of by the shell scripts. The clientSocket.py file is already in the StereoPi. It is important to have a folder called Saved_Images in the same directory as the code, server-side.

The scripts work for Windows (the .bat file), MacOS and Linux (the .sh file). There is no need to run the .py files, everything is taken care of by the shell scripts. The clientSocket.py file is already in the StereoPi.
It is very important to have the folder Saved_Images in the same directory as the server programme and the shell scripts. Every ten images, an image is going to be saved there in the format of .npy, ready to be sent through the neural network.

## Prerequirements
To run these scripts you will need to install OpenCV-python by running ```pip install opencv-python``` in the command line. 
On windows specifically you need to install PuTTy, which comes with an additionnal module called plink. Be certain that the PATH variables are well set (other necessary library is numpy, but ).
For Linux and MacOS, just need to have ssh installed.

Before using the stereopi, we need to make sure that every device is connected to the same local network. When this is done we have to find the ip address (IPv4) of the computer used as server. On linux the easiest is to type ```hostname -I``` into the terminal. For macOS the command is ```ifconfig``` and Windows you will have to use ```ipconfig```. With the ip address we can now configure the addresses hardcoded in the python scripts both on server and client side.

For the one in the StereoPi, you will need to ssh in with ```ssh pi@pydepth.local``` (or ```putty.exe -ssh pi@pydepth.local``` on windows; if this doesn't work, see at the end of this README for more info) and modify the hardcoded ip address in the script. It is located in the directory ```~/PyDepth_Client/``` . You can use nano which is a lightweight text editor to modify the file. Just type ```nano ~/PyDepth_Streaming/PyDepth_clientSocket.py```, use the arrows to find a variable called HOST and modify it to match you own IP address. Do the same with the python script (PyDepth_serverSocket.py) that is saved on you own computer. 

That is all there is to do!

Now you just need to run the shell scripts (```./PyDepth_Run.sh``` for Linux, ```sh PyDepth_Run.sh``` for MacOS and you can just run the batch file on windows by double clicking on the icon) and it will ask to enter a password to log into the StereoPi.

## Setting the SSH to work on windows
Sometimes PuTTY on windows doesn't allow mDNS resolution. This means you cannot use the simple pi@pydepth.local address to communicate, but the actual IP address of the StereoPi (so using something like ```putty.exe -ssh pi@192.168.0.1``` where 192.168.0.1 is the ip address). There are multiple ways of finding this address but an easy one is downloading the smartphone app Fing available on both iOS and Android devices and connecting to the local network. Another efficient way is to use nmap.

Once this address is found, the only thing to do is modify the .bat files ```plink pi@pydepth.local python ~/PyDepth_Client/PyDepth_clientSocket.py``` line by modifying *pydepth.local* into the found address. After that everything should run smoothly.
