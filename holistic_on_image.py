import mediapipe as mp
import cv2
import imutils

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.imread('./sample/img.jpg')
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while True:
        image = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(
                                      color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(
                                      color=(121, 44, 250), thickness=2, circle_radius=2)
                                  )

        # 3. Left Hand
        '''mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(
                                      color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(
                                      color=(121, 44, 250), thickness=2, circle_radius=2)
                                  )'''
        # here
        # print(results.left_hand_landmarks)
        try:
            x = []
            y = []
            for data_point in results.left_hand_landmarks.landmark:
                x.append(data_point.x)
                y.append(data_point.y)

                for i in range(len(x)-15):
                    cv2.circle(
                        image, (int(x[i]*1300), int(y[i]*700)), 5, (0, 0, 255), 10)
        except:
            print("not visible")

        cv2.imshow('Raw Webcam Feed', image)
        key = cv2.waitKey(1) & 0xff
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()
