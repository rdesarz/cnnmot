import cv2


def get_frame():
    webcam = cv2.VideoCapture(0)
    _, frame = webcam.read()
    webcam.release()
    return frame
