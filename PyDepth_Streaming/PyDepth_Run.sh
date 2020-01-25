python PyDepth_serverSocket.py &
sleep 3
ssh -t pi@pydepth.local 'python ~/PyDepth_Client/PyDepth_clientSocket.py'
