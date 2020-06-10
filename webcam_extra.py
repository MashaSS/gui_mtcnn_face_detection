import sys
import os
import numpy as np
import cv2
import datetime
import tensorflow as tf
import time

from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QPushButton, QLineEdit
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread

import detect_face

sys.path.append('.')
APP_FACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'saved_images', "faces")
APP_TIMER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'saved_images', "timer")

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        sess = tf.Session()
        minsize = 25  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        face_detected = False
        self.timer_started = False
        self.save_faces = False
        save_face_with_timer = False
        timer_start = 0
        sec = 0
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
            while True:
                ret, cv_img = cap.read()
                if not ret:
                    break
                img = cv_img[:, :, 0:3]
                boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                if boxes.shape[0] == 0:
                    face_detected = False
                else:
                    face_detected = True
                date_time = datetime.datetime.now()

                # timer part
                if self.timer_started:
                    timer_start = time.time()
                    self.timer_started = False
                if timer_start and not face_detected:
                    self.app.le.setText("FAIL")
                    timer_start = 0
                    self.timer_started = True
                if timer_start and (time.time() - timer_start) > sec:
                    self.app.le.setText("Wait for {} seconds...".format(5 - sec))
                    sec = sec + 1
                if timer_start and face_detected and (time.time() - timer_start) > 5.0:
                    timer_start = 0
                    save_face_with_timer = True

                for i in range(boxes.shape[0]):
                    pt1 = (int(boxes[i][0]), int(boxes[i][1]))
                    pt2 = (int(boxes[i][2]), int(boxes[i][3]))
                    crop_img = cv_img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
                    clean_img = cv_img
                    if self.save_faces:
                        img_path = os.path.join(APP_FACES_DIR, 'face_{}_date_{}-{}-{}-{}-{}.png'.format(i + 1,
                                                                                                        date_time.month,
                                                                                                        date_time.day,
                                                                                                        date_time.hour,
                                                                                                        date_time.minute,
                                                                                                        date_time.second))
                        cv2.imwrite(img_path, crop_img)
                        if i + 1 == boxes.shape[0]:
                            self.save_faces = False
                    if save_face_with_timer and not i:
                        img_path = os.path.join(APP_TIMER_DIR, 'timer_face_date_{}-{}-{}-{}-{}.png'.format(date_time.month,
                                                                                                           date_time.day,
                                                                                                           date_time.hour,
                                                                                                           date_time.minute,
                                                                                                           date_time.second))
                        cv2.imwrite(img_path, clean_img)
                        self.app.le.setText("SUCCESS")
                        sec = 0
                    cv2.rectangle(cv_img, pt1, pt2, color=(255, 255, 0))
                if ret:
                    self.change_pixmap_signal.emit(cv_img)
                    if save_face_with_timer:
                        time.sleep(1)
                        save_face_with_timer = False

    def get_app(self, app):
       self.app = app

    def start_timer(self):
        self.timer_started = True


    def save_all_faces(self):
        self.save_faces = True


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face detection")
        self.disply_width = 900
        self.display_height = 750
        self.setFixedSize(self.disply_width, self.display_height)
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        # create a text label
        self.textLabel = QLabel('Start timer to make photo: ')

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.get_app(self)
        self.thread.start()

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabel)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        self.btn1 = QPushButton("start timer", self)
        self.btn1.move(210, 700)
        self.btn1.clicked.connect(self.btn_start_timer)

        self.le = QLineEdit(self)
        self.le.move(320, 700)
        self.le.setReadOnly(True)
        self.le.resize(250, 30)

        self.btn2 = QPushButton("save all faces", self)
        self.btn2.move(700, 700)
        self.btn2.clicked.connect(self.btn_save_faces)

    def btn_start_timer(self):
        self.thread.start_timer()

    def btn_save_faces(self):
        self.thread.save_all_faces()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == "__main__":
    if not os.path.isdir(APP_FACES_DIR):
        raise FileExistsError("Error: {} does not exist. Please create this directory.".format(APP_FACES_DIR))
    if not os.path.isdir(APP_TIMER_DIR):
        raise FileExistsError("Error: {} does not exist. Please create this directory.".format(APP_TIMER_DIR))
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
