echo "You will be able to enter the password in 7 seconds."
echo ""
python PyDepth_serverSocket.py &
sleep 7
ssh -t pi@pydepth.local 'python ~/PyDepth_Client/PyDepth_clientSocket.py'
