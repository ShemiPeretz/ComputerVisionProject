import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMash = mp.solutions.face_mesh
faceMash = mpFaceMash.FaceMesh()
drawSpec = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=2)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = faceMash.process(imgRGB)
    if result.multi_face_landmarks:
        for faceLms in result.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMash.FACE_CONNECTIONS, landmark_drawing_spec=drawSpec)
            for id, lm in enumerate(faceLms.landmark):
                ih, iw, ic = img.shape
                x, y = int(lm.x*iw), int(lm.y*ih)
                print(id, x, y)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)