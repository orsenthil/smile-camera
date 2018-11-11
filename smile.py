import subprocess
from scipy.spatial import distance as dist
from imutils.video import VideoStream, FPS
from imutils import face_utils
import imutils
import numpy as np
import time
import dlib
import cv2


def smile(mouth):
    A = dist.euclidean(mouth[3], mouth[9])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[4], mouth[8])
    avg = (A+B+C)/3
    D = dist.euclidean(mouth[0], mouth[6])
    mar = avg / D
    return mar


COUNTER = 0
TOTAL = 0

SMILE_THRESHOLD = 0.30


def click():
    subprocess.call(["afplay", "/Users/senthil/facedetect/click.wav"])


shape_predictor = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

print("[INFO] starting video stream thread...")

vs = VideoStream(src=0).start()
fileStream = False

time.sleep(1.0)

fps = FPS().start()

cv2.namedWindow("smile detector.")

while True:
    frame = vs.read()

    frame = imutils.resize(frame, width=450)

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

        else:
            if COUNTER >= 5:
                TOTAL += 1
                time.sleep(0.2)
                click()

                cv2.putText(frame, "Click!", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                frame = vs.read()

                frame2 = frame.copy()

                img_name = "smile_photo_{}.png".format(TOTAL)

                cv2.imwrite(img_name, frame2)
                print("{} written!".format(img_name))

            else:
                if COUNTER < 3:
                    cv2.putText(frame, "Smile!", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 2)
                else:
                    cv2.putText(frame, "Smile Wider!", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 2)


            COUNTER = 0

    cv2.imshow("Smile Camera", frame)

    fps.update()

    key2 = cv2.waitKey(1) & 0xFF

    if key2 == ord('q'):
        break

fps.stop()

cv2.destroyAllWindows()
vs.stop()
