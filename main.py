from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
from threading import Thread
import threading
from notify_run import Notify
import datetime
import pyglet
import cv2
import eel
eel.init('web')
eel.browsers.set_path('chrome', 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe')

protopath = "MobileNetSSD_deploy.prototxt.txt"
modelpath = "MobileNetSSD_deploy.caffemodel"
check_send = 0
check_end = 0
check_time = 0
mode = 0
link = ""
def check_camera(mode,link):
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]
	COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
	print("[INFO] loading model...")
	net = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
	print("[INFO] starting video stream...")
	if link == "0":
		vs = VideoStream(0).start()
	else:
		vs = VideoStream(link).start()
	fps_start_time = datetime.datetime.now()
	fps = 0
	total_frames = 0
	while True:
		global check_send,check_end
		check_send = 0
		frame = vs.read()
		total_frames = total_frames + 1
		frame = imutils.resize(frame, width=800)
		if mode == 1:
			(h, w) = frame.shape[:2]
			blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
				0.007843, (300, 300), 127.5)
			net.setInput(blob)
			detections = net.forward()
			for i in np.arange(0, detections.shape[2]):
				confidence = detections[0, 0, i, 2]
				if confidence > 0.5:
					idx = int(detections[0, 0, i, 1])
					if CLASSES[idx] != "person":
						continue
					check_send = 1
					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")
					cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
					cv2.putText(frame, "Canh bao dot nhap!", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0 , 255), 2)

		fps_end_time = datetime.datetime.now()
		time_diff = fps_end_time - fps_start_time
		if time_diff.seconds == 0:
			fps = 0.0
		else:
			fps = (total_frames / time_diff.seconds)
		fps_text = "FPS: {:.2f}".format(fps)
		cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF


		if key == ord("q"):
			break
	cv2.destroyAllWindows()
	check_end = 1
	vs.stop()
def checktime():
	global check_time
	while check_end < 1:
		for x in range(1, 6):
			check_time = x
			time.sleep(1)

def send_notice():
	global check_send,check_end,check_time
	while check_end < 1:
		if check_send == 1:
			if check_time>2:
				music = pyglet.resource.media('police.wav')
				music.play()
				notify = Notify()
				notify.send('Hi there!')
@eel.expose
def main(mode,link):
	try:
		t1 = threading.Thread(target=check_camera, args=(mode,link))
		t2 = threading.Thread(target=send_notice)
		t3 = threading.Thread(target=checktime)
		t1.start()
		t2.start()
		t3.start()
		t1.join()
		t2.join()
		t3.join()
	except:
		print("Lỗi")
eel.start('index.html', size=(1000, 600))
