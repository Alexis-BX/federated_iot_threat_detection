#!/bin/bash

python3 server.py &

sleep 5

python3 client.py 0 > /dev/null &
python3 client.py 1 > /dev/null &

time python3 client.py 2 > /dev/null
