from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
# Disable TensorFlow 2.x behavior to use TensorFlow 1.x style
tf.compat.v1.disable_eager_execution()

from imutils.video import VideoStream

import argparse
import facenet
import imutils
import os
import sys
import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path of the video you want to test on.', default=0)
    args = parser.parse_args()

    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160
    VIDEO_PATH = args.path
    FACENET_MODEL_PATH = 'Models/20180402-114759.pb'
    
    # Cải thiện độ chính xác
    CONFIDENCE_THRESHOLD = 0.7   # Ngưỡng tin cậy
    MIN_FACE_SIZE = 40          # Kích thước mặt tối thiểu (pixels)

    # Load The Custom Classifier - ưu tiên model cải thiện
    classifier_paths = [
        'Models/improved_facemodel.pkl',  # Model mới
        'Models/facemodel.pkl'            # Model cũ backup
    ]
    
    model = None
    class_names = None
    
    for path in classifier_paths:
        try:
            with open(path, 'rb') as file:
                model, class_names = pickle.load(file)
            print(f"✅ Loaded classifier: {path}")
            print(f"📋 Available classes: {class_names}")
            break
        except FileNotFoundError:
            print(f"⚠️ Không tìm thấy: {path}")
        except Exception as e:
            print(f"❌ Lỗi load {path}: {e}")
    
    if model is None:
        print("❌ Không thể load classifier!")
        return

    with tf.Graph().as_default():

        # Cai dat GPU neu co
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)

            # Get input and output tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")

            people_detected = set()
            person_detected = collections.Counter()

            cap = VideoStream(src=0).start()

            while (True):
                frame = cap.read()
                if frame is None:
                    continue
                    
                frame = imutils.resize(frame, width=600)
                frame = cv2.flip(frame, 1)

                bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

                faces_found = bounding_boxes.shape[0]
                
                try:
                    if faces_found > 0:
                        det = bounding_boxes[:, 0:4]
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        
                        for i in range(faces_found):
                            # Tính toán bounding box
                            bb[i][0] = max(0, int(det[i][0]))
                            bb[i][1] = max(0, int(det[i][1]))
                            bb[i][2] = min(frame.shape[1], int(det[i][2]))
                            bb[i][3] = min(frame.shape[0], int(det[i][3]))
                            
                            # Kiểm tra kích thước khuôn mặt
                            face_height = bb[i][3] - bb[i][1]
                            face_width = bb[i][2] - bb[i][0]
                            
                            # Luôn vẽ bounding box
                            color = (0, 255, 0)  # Màu xanh lá mặc định
                            name = "Unknown"
                            confidence_text = ""
                            
                            # Chỉ thực hiện nhận diện nếu khuôn mặt đủ lớn
                            if face_width > MIN_FACE_SIZE and face_height > MIN_FACE_SIZE:
                                try:
                                    # Cắt và xử lý khuôn mặt
                                    cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                                    
                                    if cropped.size > 0:
                                        scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                          interpolation=cv2.INTER_CUBIC)
                                        scaled = facenet.prewhiten(scaled)
                                        scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                                        feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                                        emb_array = sess.run(embeddings, feed_dict=feed_dict)

                                        predictions = model.predict_proba(emb_array)
                                        best_class_indices = np.argmax(predictions, axis=1)
                                        best_class_probabilities = predictions[
                                            np.arange(len(best_class_indices)), best_class_indices]
                                        best_name = class_names[best_class_indices[0]]
                                        confidence = best_class_probabilities[0]
                                        
                                        print(f"🎯 Dự đoán: {best_name} (confidence: {confidence:.4f})")
                                        
                                        # Kiểm tra độ tin cậy
                                        if confidence > CONFIDENCE_THRESHOLD:
                                            name = best_name
                                            color = (0, 255, 0)  # Xanh lá cho nhận diện thành công
                                            confidence_text = f" ({confidence:.2f})"
                                            person_detected[best_name] += 1
                                        else:
                                            name = "Unknown"
                                            color = (0, 0, 255)  # Đỏ cho không nhận diện được
                                            confidence_text = f" ({confidence:.2f})"
                                            
                                except Exception as e:
                                    print(f"Lỗi nhận diện: {e}")
                                    name = "Error"
                                    color = (128, 128, 128)  # Xám cho lỗi
                            else:
                                name = "Too small"
                                color = (255, 0, 0)  # Xanh dương cho khuôn mặt quá nhỏ
                            
                            # VẼ BOUNDING BOX VÀ TÊN (LUÔN LUÔN)
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), color, 2)
                            
                            # Vẽ tên và độ tin cậy
                            text_x = bb[i][0]
                            text_y = bb[i][1] - 10 if bb[i][1] > 30 else bb[i][3] + 20
                            
                            cv2.putText(frame, name + confidence_text, (text_x, text_y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                            
                        # Hiển thị số lượng khuôn mặt phát hiện
                        cv2.putText(frame, f"Faces detected: {faces_found}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    else:
                        # Không phát hiện khuôn mặt nào
                        cv2.putText(frame, "No face detected", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                except Exception as e:
                    print(f"Error in face recognition: {e}")
                    cv2.putText(frame, f"Error: {str(e)}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # Hiển thị thông tin hệ thống
                cv2.putText(frame, f"Confidence threshold: {CONFIDENCE_THRESHOLD}", (10, frame.shape[0] - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow('Face Recognition - Improved', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.stop()
            cv2.destroyAllWindows()


main()
