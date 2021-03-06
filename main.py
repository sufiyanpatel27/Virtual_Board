import mediapipe as mp
import cv2
import imutils
import pickle

model = pickle.load(open("ML_model/model.sav", 'rb'))

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture('ML_model/data/video_data/start_or_stop.mp4')
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()

        image = imutils.resize(frame, width=400)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(
                                      color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(
                                      color=(121, 44, 250), thickness=2, circle_radius=2)
                                  )

        try:
            keypoints = []
            x = []
            y = []
            z = []
            vis = []

            for data_point in results.pose_landmarks.landmark:
                keypoints.append({
                    'X': data_point.x,
                    'Y': data_point.y,
                    'Z': data_point.z,
                    'Visibility': data_point.visibility,
                })
                x.append(-data_point.x)
                x.append(-data_point.y)
                z.append(-data_point.z)
                vis.append(data_point.visibility)

            # print(x[:42])
            print(model.predict([x[:42]]))
        except:
            g = 10

        cv2.imshow('Raw Webcam Feed', image)
        key = cv2.waitKey(1) & 0xff
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()
