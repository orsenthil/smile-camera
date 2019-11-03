import subprocess
from scipy.spatial import distance as dist
from imutils.video import VideoStream, FPS
from imutils import face_utils
import time
import dlib
import cv2
import datetime


def smile(mouth):
    A = dist.euclidean(mouth[3], mouth[9])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[4], mouth[8])
    avg = (A+B+C)/3
    D = dist.euclidean(mouth[0], mouth[6])
    # mouth aspect ratio
    # this is our measure of smile
    mar = avg / D
    return mar


COUNTER = 0
TOTAL = 0


def get_unique_timestamp():
    _now = datetime.datetime.now()
    return _now.strftime("%Y-%m-%d-%H-%M-%S")


SMILE_THRESHOLD = 0.30


def click():
    subprocess.call(["aplay", "click.wav"])


shape_predictor = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

print("[INFO] starting video stream thread...")

# 640, 480 can be tried too.
# , resolution=(800, 600)
vs = VideoStream(src=0, resolution=(640, 480)).start()
fileStream = False

time.sleep(1.0)

fps = FPS().start()

window_name = "Smile Camera"

cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)

while True:
    frame = vs.read()

    #frame = imutils.resize(frame, width=800)

    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_rects = detector(grayscale_frame, 0)
    num_faces = len(face_rects)

    face_smile_values = {face_id: 0 for face_id in range(num_faces)}

    for idx, face in enumerate(face_rects):
        shape = predictor(grayscale_frame, face)
        shape = face_utils.shape_to_np(shape)
        mouth = shape[mStart:mEnd]

        mar = smile(mouth)
        face_smile_values[idx] = mar
        mouthHull = cv2.convexHull(mouth)

        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        sum_of_smiles = sum(face_smile_values.values())
        total_smile_threshold = SMILE_THRESHOLD * num_faces

        if sum_of_smiles > total_smile_threshold:
            COUNTER += 1

        if COUNTER >= 6:
            TOTAL += 1
            time.sleep(0.2)
            click()

            cv2.putText(frame, "Click!", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            frame = vs.read()

            frame2 = frame.copy()

            img_name = "smile_photo_{total}_{ts}.png".format(total=TOTAL, ts=get_unique_timestamp())

            cv2.imwrite(img_name, frame2)
            print("{} written!".format(img_name))
            COUNTER = 0

        else:
            if COUNTER < 5:
                cv2.putText(frame, "Smile wide!", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "Smile Wider!", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

    cv2.imshow(window_name, frame)

    fps.update()

    key2 = cv2.waitKey(1) & 0xFF

    if key2 == ord('q'):
        break

fps.stop()

cv2.destroyAllWindows()
vs.stop()
