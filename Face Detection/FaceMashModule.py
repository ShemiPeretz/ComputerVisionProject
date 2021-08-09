import cv2
import mediapipe as mp
import time


class FaceDetector:
    def __init__(self, mode=False, maxfaces=1, detection_confidence=0.5, tracking_confidence=0.5,
                 color=(0, 0, 255), thickness=1, circle_radius=2):
        self.mode = mode
        self.numFaces = maxfaces
        self.detectionCon = detection_confidence
        self. trackingCon = tracking_confidence
        self.color = color
        self.thickness = thickness
        self.radius = circle_radius

        self.mpFaceMash = mp.solutions.face_mesh
        self.faceMash = self.mpFaceMash.FaceMesh(self.mode, self.numFaces, self.detectionCon, self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(self.color, self.thickness, self.radius)

    def find_face(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMash.process(imgRGB)

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMash.FACE_CONNECTIONS,
                                               landmark_drawing_spec=self.drawSpec)
        return img

    def find_position(self, img, draw=True):
        lm_list = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    lm_list.append([id, x, y])
                    # if draw:
                    #     cv2.circle(img, (x, y), 8, (255, 0, 255), cv2.FILLED)

        return lm_list


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img1 = detector.find_face(img)
        lm_list = detector.find_position(img)
        if len(lm_list) != 0:
            print(lm_list[4])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()