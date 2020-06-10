# gui_mtcnn_face_detection
Gui aplication for face detection from webcam using pre-trained MTCNN based on repo (https://github.com/ResByte/mtcnn-face-detect).

## Dependencies
* tensorflow
* opencv
* numpy

## Installation
```
git clone https://github.com/MashaSS/gui_mtcnn_face_detection.git
git clone https://github.com/ResByte/mtcnn-face-detect.git

cp gui_mtcnn_face_detection/webcam_extra.py mtcnn-face-detect/src
cd mtcnn-face-detec
mkdir saved_images/timer
mkdir saved_images/faces

python src/webcam_extra.py
```
