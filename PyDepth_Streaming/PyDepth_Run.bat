@ECHO OFF
ECHO You will be able to enter the password in 7 seconds.
ECHO
start /B python PyDepth_serverSocket.py
ECHO
TIMEOUT 7
ECHO
plink pi@pydepth.local python ~/PyDepth_Client/PyDepth_clientSocket.py
