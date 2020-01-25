start /B python PyDepth_serverSocket.py 
SLEEP 3 
plink pi@pydepth.local python ~/PyDepth_Client/PyDepth_clientSocket.py