# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
import time
import cv2

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

class YOLO(object):
    _defaults = {
        # "model_path": 'model_data/yolo.h5',
        # "anchors_path": 'model_data/yolo_anchors.txt',
        # "model_path": 'model_data/yolo_tiny.h5',
        # "model_path": 'logs/000/trained_weights_final.h5',
        "model_path": 'logs/000/trained_weights_final#001.h5',
        "anchors_path": 'model_data/tiny_yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }
    last_out_scores = []
    last_out_classes = []
    last_out_boxes = np.array([])
    average_boxes_gap = 0
    i = 0
    predict_gap = 4
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognizedanch attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        # print(np.array(anchors).reshape(-1, 2))
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        is_tiny_version = num_anchors==6 # default setting

        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()
        start_timer = time.time()
        time.clock()
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        ''' 
        The tf.nn.conv_2d() op takes much longer to run on the first tf.Session.run() invocation because—by default—TensorFlow uses cuDNN's autotune facility to choose how to run subsequent convolutions as fast as possible
        https://stackoverflow.com/questions/45063489/first-tf-session-run-performs-dramatically-different-from-later-runs-why '''
        if self.i % self.predict_gap == 0:
            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.yolo_model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })
            '''print('frame : ', self.i, 'predict : ', out_boxes)
            if self.last_out_boxes.any():
                if self.last_out_boxes.shape == out_boxes.shape:
                    self.average_boxes_gap = (self.last_out_boxes - out_boxes) / self.predict_gap
                else:
                    print('not same shape')
                print('average_boxes_gap', self.average_boxes_gap)
            else:
                self.average_boxes_gap = 0'''
        else:
            out_boxes, out_scores, out_classes = self.last_out_boxes, self.last_out_scores, self.last_out_classes
            '''if self.i < 4:
                out_boxes, out_scores, out_classes = self.last_out_boxes, self.last_out_scores, self.last_out_classes
            else:
                out_boxes, out_scores, out_classes = self.last_out_boxes + self.average_boxes_gap, self.last_out_scores, self.last_out_classes'''

        self.last_out_boxes, self.last_out_scores, self.last_out_classes = out_boxes, out_scores, out_classes
        # print(self.last_out_boxes, ';', self.last_out_scores, ';', self.last_out_classes)
        self.i += 1
        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            # print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        # print('time for one frame in sec: ', end - start)
        return image, start_timer

    def close_session(self):
        self.sess.close()


def detect_video(yolo, video_path, output_path=""):
    model_path = yolo.get_defaults('model_path')
    directory = model_path.split('/')[2].split('#')[1].split('.')[0]
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    # video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # isOutput = True if output_path != "" else False
    '''if isOutput:
        # print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)'''
    ''' accum_time = 0
    curr_fps = 0
    fps = "FPS: ??" '''
    prev_time = timer()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    path = 'result/'+directory+'/'+video_path.split('/')[::-1][0]
    video = cv2.VideoWriter(path, fourcc, float(video_fps), video_size)
    start = None
    while True:
        time.clock()

        return_value, frame = vid.read()
        if return_value is False:
            break
        ''' 
            print(type(frame[::-1][0]))
            print(frame[::-1][0])
            exit()
        '''
        image = Image.fromarray(frame)
        if start is None:
            image, start = yolo.detect_image(image)
        else:
            image, _ = yolo.detect_image(image)

        result = np.asarray(image)
        ''' curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time'''
        ''' curr_fps = curr_fps + 1 '''
        ''' if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0 '''
        # cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale=0.50, color=(255, 0, 0), thickness=2)
        # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        # cv2.imshow("result", result)
        video.write(result)
        '''if isOutput:
            out.write(result)'''
        ''' if cv2.waitKey(1) & 0xFF == ord('q'):
            break '''
    end = time.time()
    print('Temps de traitement de la video : ', end-start, 'seconds')
    video.release()
    yolo.close_session()

