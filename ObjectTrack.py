from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import numpy
import torch
import cv2
import numpy as np
import csv


    # initialize a dictionary that maps strings to their corresponding
    # OpenCV object tracker implementations
  # grab the appropriate object tracker using our dictionary of
    # OpenCV object tracker objects
pts_src=np.array([[1,1],[2,1],[2,2],[1,2]])
dst_src=np.array([[240,796],[447,796],[447,597],[248,590]])
h, status = cv2.findHomography(pts_src, dst_src)

print(cv2.perspectiveTransform(pts_src, h))
print(cv2.perspectiveTransform(dst_src, np.linalg.inv(h)))

print(h)
print(np.linalg.inv(h))
tracker = cv2.legacy_TrackerBoosting.create();
# initialize the bounding box coordinates of the object we are going
# to track
outList=[]
initBB = None
fps=None
vs = cv2.VideoCapture("/Users/jules/downloads/video_for_huaner/MVI_0848.MP4")
i=0;
while True:
    frame = vs.read()
    frame = frame[1] #if args.get("video", False) else frame
    # check to see if we have reached the end of the stream
    #if frame is None:
    #    break
    # resize the frame (so we can process it faster) and grab the
    # frame dimensions
    if frame is None:
        break

    frame = imutils.resize(frame, width=500)
   # newF=numpy.array(frame)
  #  (H, W) = frame.shape[:2]
    i+=1;
    H=300
    W=500
    if initBB is not None:
        #outList.append([initBB[1],initBB[0]])
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)
        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                (0, 255, 0), 2)
        
            transformed = cv2.perspectiveTransform(np.float32([[[x+w/2, y+h/2]]]), np.linalg.inv(h))
          
            print(transformed)
            outList.append([i,x+w/2,y+h/2])
           # outList.append([i,x+w/2,y+h/2,transformed[0,0,0],transformed[0,0,1]])
        fps.update()
        fps.stop()
        # initialize the set of information we'll be displaying on
        # the frame
        info = [
            ("Tracker", cv2.legacy_TrackerBoosting.create()),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
        ]
        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        initBB = cv2.selectROI("Frame", frame, fromCenter=False,
        showCrosshair=True)
        print("here")
       # outList.append([initBB[1],initBB[0]])
        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        tracker.init(frame, initBB)
        fps = FPS().start()
    elif key == ord("q"):
         break
vs.release()
cv2.destroyAllWindows()
filename='/Users/jules/downloads/index2.csv'
with open(filename,'w') as f:
   write=csv.writer(f)
   write.writerow({'VideoX','VideoY','transformedX','transfomedY'})
   write.writerows(outList)

