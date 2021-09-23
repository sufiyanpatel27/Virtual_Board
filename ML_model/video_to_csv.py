import mediapipe as mp
import cv2
import imutils
import pandas as pd
import os


for video_frame_name in os.listdir('data/video_data'):
    print(video_frame_name)
    name = []
    for i in range(21):
        name.append(str(i) + "_x")
        name.append(str(i) + "_y")
        name.append(str(i) + "_z")
        name.append(str(i) + "_vis")
    df = pd.DataFrame(columns=name)

    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    cap = cv2.VideoCapture('data/video_data/'+video_frame_name)
    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("saving data...")
                df.to_csv(r'C:/Users/sufiyan/Desktop/virtual board/final_project/ML_model/data/categorical_data/' +
                          video_frame_name+".csv", index=False, header=True)
                print("data captured successfully")
                break

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
                y.append(-data_point.y)
                z.append(-data_point.z)
                vis.append(data_point.visibility)

            data = {}
            for i in range(21):
                data[str(i) + "_x"] = x[i]
                data[str(i) + "_y"] = y[i]
                data[str(i) + "_z"] = z[i]
                data[str(i) + "_vis"] = vis[i]

            df = df.append(data, ignore_index=True)

            cv2.imshow('Raw Webcam Feed', image)
            key = cv2.waitKey(1) & 0xff
            if key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
