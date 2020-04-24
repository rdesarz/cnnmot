# CNN-MOT
The goal of this project is to implement a multiple object tracker algorithm for computer vision applications. 

## Object detection
The object detection algorithm is based on a CNN pre-trained Yolo v3 model proposed in [this paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf). Some part of the code are used in this project as well as
part of this [tutorial](https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/)

## Installation
The project is written in Python (v3.6). You can install it using `pip`.

```bash
pip3 install git+https://github.com/rdesarz/cnnmot.git
```

### Keras model generation 
This project is using Keras however the and so the model needs to be translated to a Keras model before using any of the scripts. Run the following instructions in the root folder of the repo. 
It will download the model and translate it.
```bash
wget  https://pjreddie.com/media/files/yolov3.weights
./bin/generate_keras_model.py --weights yolov3.weights --model_name yolov3.h5
```

## Usage
To get a demo of the detection, a Jupyter notebook `yolov3_object_detection_colab_example` located in the notebook folder of the project is available. 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
