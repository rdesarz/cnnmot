import cv2
from keras.models import load_model
from yolo3.preprocessing import process_image
from yolo3.postprocessing import decode_netout, correct_yolo_boxes, do_nms, get_boxes
from yolo3.output import draw_boxes
from inputs import webcam
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model', action="store", dest="model_path", default='model.h5',
                    help='Path to the Keras model to load (default: model.h5)')
parser.add_argument('--threshold', action="store", dest="threshold", default=0.6,
                    help='Threshold for the filtering of prediction (default: 0.6)')
args = parser.parse_args()


def main():
    model = load_model(args.model_path)
    input_shape = (416, 416)
    threshold = args.threshold
    labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
              "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
              "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
              "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
              "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
    anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
    frame = webcam.get_frame()
    # preprocess image to input it in the network
    image, image_w, image_h = process_image(frame, input_shape)
    # make prediction
    yhat = model.predict(image)
    boxes = list()
    for i in range(len(yhat)):
        # decode the output of the network
        boxes += decode_netout(yhat[i][0], anchors[i], threshold, input_shape[0], input_shape[1])
    # correct the sizes of the bounding boxes for the shape of the image
    correct_yolo_boxes(boxes, image_h, image_w, input_shape[0], input_shape[1])
    # suppress non-maximal boxes
    do_nms(boxes, 0.5)
    # get the details of the detected objects
    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, threshold)
    # draw the resulting prediction
    draw_boxes(frame, v_boxes, v_labels, v_scores)
    cv2.waitKey()


if __name__ == "__main__":
    # execute only if run as a script
    main()
