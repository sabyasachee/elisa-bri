#!/bin/bash

BASEDIR='/services'

cd $BASEDIR
python server.py &

echo "=========================================================================================
Welcome to the BRI Opinion Frame
Send any questions to Sabyasachee Baruah, sbaruah@usc.edu
===============================================================================================
PLEASE ALLOW A FEW MINUTES FOR ALL MODELS TO BE LOADED
==============================================================================================="

while :
do
	sleep 500
done
