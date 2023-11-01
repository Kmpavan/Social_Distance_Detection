from mylib import config, thread
from mylib.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse, imutils, cv2, os
from flask import Flask,render_template,Response,request
import pyperclip

app=Flask(__name__,template_folder='templates')

#----------------------------Parse req. arguments------------------------------#
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
args = vars(ap.parse_args())
#------------------------------------------------------------------------------#

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
if config.USE_GPU:
	# set CUDA as the preferable backend and target
	print("")
	print("[INFO] Looking for GPU")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

def generate_frames(path):
    Video_Path = path

    print("[INFO] Starting the video..")
    vs = cv2.VideoCapture(Video_Path)
    if config.Thread:
            cap = thread.ThreadingClass(Video_Path)

    writer = None

    while True:
            
        ## read the video frame
        success,frame=vs.read()
        if not success:
            break
        else:
            # ============= #
            # resize the frame and then detect people (and only people) in it
            frame = imutils.resize(frame, width=700)
            results = detect_people(frame, net, ln,
                personIdx=LABELS.index("person"))

            # initialize the set of indexes that violate the max/min social distance limits
            serious = set()
            abnormal = set()

            # ensure there are *at least* two people detections (required in
            # order to compute our pairwise distance maps)
            if len(results) >= 2:
                # extract all centroids from the results and compute the
                # Euclidean distances between all pairs of the centroids
                centroids = np.array([r[2] for r in results])
                D = dist.cdist(centroids, centroids, metric="euclidean")

                # loop over the upper triangular of the distance matrix
                for i in range(0, D.shape[0]):
                    for j in range(i + 1, D.shape[1]):
                        # check to see if the distance between any two
                        # centroid pairs is less than the configured number of pixels
                        if D[i, j] < config.MIN_DISTANCE:
                            # update our violation set with the indexes of the centroid pairs
                            serious.add(i)
                            serious.add(j)
                        # update our abnormal set if the centroid distance is below max distance limit
                        if (D[i, j] < config.MAX_DISTANCE) and not serious:
                            abnormal.add(i)
                            abnormal.add(j)

            # loop over the results
            for (i, (prob, bbox, centroid)) in enumerate(results):
                # extract the bounding box and centroid coordinates, then
                # initialize the color of the annotation
                (startX, startY, endX, endY) = bbox
                (cX, cY) = centroid
                color = (0, 255, 0)

                # if the index pair exists within the violation/abnormal sets, then update the color
                if i in serious:
                    color = (0, 0, 255)
                elif i in abnormal:
                    color = (0, 255, 255) #orange = (0, 165, 255)

                # draw (1) a bounding box around the person and (2) the
                # centroid coordinates of the person,
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.circle(frame, (cX, cY), 5, color, 2)

            # draw some of the parameters
            Safe_Distance = "Safe distance: >{} px".format(config.MAX_DISTANCE)
            cv2.putText(frame, Safe_Distance, (470, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)
            Threshold = "Threshold limit: {}".format(config.Threshold)
            cv2.putText(frame, Threshold, (470, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)

            # draw the total number of social distancing violations on the output frame
            text = "Total serious violations: {}".format(len(serious))
            cv2.putText(frame, text, (10, frame.shape[0] - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)

            hreext1 = "Total abnormal violations: {}".format(len(abnormal))
            cv2.putText(frame, text, (10, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)

        #------------------------------Alert function----------------------------------#
            if len(serious) >= config.Threshold:
                cv2.putText(frame, "-ALERT: Violations over limit-", (10, frame.shape[0] - 80),
                    cv2.FONT_HERSHEY_COMPLEX, 0.60, (0, 0, 255), 2)
                if config.ALERT:
                    print("")
                    print('[INFO] Sending mail...')
                   # Mailer().send(config.MAIL)
                    print('[INFO] Mail sent')
                #config.ALERT = False
        #------------------------------------------------------------------------------#

            # if an output video file path has been supplied and the video
            # writer has not been initialized, do so now
            if args["output"] != "" and writer is None:
                # initialize our video writer
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(args["output"], fourcc, 25,
                    (frame.shape[1], frame.shape[0]), True)

            # if the video writer is not None, write the frame to the output video file
            if writer is not None:
                writer.write(frame)
            # ============= #
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/')
def homepage():
    return render_template('homepage.html')
@app.route('/live', methods=['POST', 'GET'])
def live():
    return Response(generate_frames(0),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video', methods=['POST', 'GET'])
def video():
        path = pyperclip.paste()
        print(path)
        return Response(generate_frames(path),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)