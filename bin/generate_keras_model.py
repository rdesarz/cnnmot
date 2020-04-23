#!/usr/bin/env python3
from cnnmot.yolo.model import get_keras_model_from_weight
import argparse

parser = argparse.ArgumentParser(description='Load a weight file and generate a Keras Yolov3 model')
parser.add_argument('--weights', action="store", dest="weights_path", default='model.weights',
                    help='Path to the weights of the model (default: model.weights)')
parser.add_argument('--model_name', action="store", dest="model_name", default='model.h5',
                    help='name of the Keras model generated (default: model.h5)')
args = parser.parse_args()


def main():
    model = get_keras_model_from_weight(args.weights_path)
    # save the model to file
    model.save(args.model_name)


if __name__ == "__main__":
    main()
