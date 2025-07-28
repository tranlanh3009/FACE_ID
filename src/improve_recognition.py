#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script cải thiện độ chính xác nhận diện khuôn mặt
Khắc phục vấn đề nhận diện sai giữa Lanh và Việt
"""

import os
import cv2
import numpy as np
import pickle
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import facenet
import align.detect_face
from imutils.video import VideoStream
import imutils

tf.compat.v1.disable_eager_execution()

class ImprovedFaceRecognition:
    def __init__(self):
        self.MINSIZE = 20
        self.THRESHOLD = [0.6, 0.7, 0.7]
        self.FACTOR = 0.709
        self.IMAGE_SIZE = 182
        self.INPUT_IMAGE_SIZE = 160
        self.FACENET_MODEL_PATH = 'Models/20180402-114759.pb'
        
        # Tăng cường ngưỡng phân biệt
        self.CONFIDENCE_THRESHOLD = 0.9  # Tăng lên 0.9 để chặt chẽ hơn
        self.DISTANCE_THRESHOLD = 0.6    # Ngưỡng khoảng cách embeddings
        
        self.sess = None
        self.pnet = None
        self.rnet = None 
        self.onet = None
        self.model = None
        self.class_names = None
        self.embeddings_placeholder = None
        self.phase_train_placeholder = None
        self.embeddings_op = None
        
    def setup_tensorflow(self):
        """Khởi tạo TensorFlow session và models"""
        self.graph = tf.Graph()
        with self.graph.as_default():
            gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
            self.sess = tf.compat.v1.Session(
                config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
            )
            
            with self.sess.as_default():
                # Load FaceNet model
                facenet.load_model(self.FACENET_MODEL_PATH)
                
                # Get tensors
                self.embeddings_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings_op = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
                
                # Create MTCNN
                self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(self.sess, "src/align")
                
        print("✅ TensorFlow setup completed")
    
    def augment_image(self, image):
        """Tăng cường dữ liệu ảnh"""
        augmented = []
        
        # Ảnh gốc
        augmented.append(image)
        
        # Lật ngang
        augmented.append(cv2.flip(image, 1))
        
        # Thay đổi độ sáng
        bright = cv2.convertScaleAbs(image, alpha=1.2, beta=20)
        augmented.append(bright)
        
        dark = cv2.convertScaleAbs(image, alpha=0.8, beta=-20)
        augmented.append(dark)
        
        # Xoay nhẹ
        h, w = image.shape[:2]
        center = (w//2, h//2)
        
        # Xoay 5 độ
        M1 = cv2.getRotationMatrix2D(center, 5, 1.0)
        rotated1 = cv2.warpAffine(image, M1, (w, h))
        augmented.append(rotated1)
        
        # Xoay -5 độ
        M2 = cv2.getRotationMatrix2D(center, -5, 1.0)
        rotated2 = cv2.warpAffine(image, M2, (w, h))
        augmented.append(rotated2)
        
        return augmented
    
    def extract_embeddings(self, image_path):
        """Trích xuất embeddings từ ảnh"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Không thể đọc ảnh: {image_path}")
                return None
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            bounding_boxes, _ = align.detect_face.detect_face(
                image_rgb, self.MINSIZE, self.pnet, self.rnet, self.onet, 
                self.THRESHOLD, self.FACTOR
            )
            
            if len(bounding_boxes) == 0:
                print(f"❌ Không tìm thấy khuôn mặt trong: {image_path}")
                return None
                
            if len(bounding_boxes) > 1:
                print(f"⚠️ Tìm thấy {len(bounding_boxes)} khuôn mặt, chỉ lấy khuôn mặt đầu tiên: {image_path}")
            
            # Lấy khuôn mặt đầu tiên
            bb = bounding_boxes[0][:4]
            bb = [max(0, int(coord)) for coord in bb]
            
            face = image_rgb[bb[1]:bb[3], bb[0]:bb[2], :]
            
            if face.size == 0:
                print(f"❌ Khuôn mặt trống: {image_path}")
                return None
            
            # Augment data
            augmented_faces = self.augment_image(face)
            embeddings_list = []
            
            for aug_face in augmented_faces:
                # Resize và preprocess
                aligned = cv2.resize(aug_face, (self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE))
                prewhitened = facenet.prewhiten(aligned)
                prewhitened = prewhitened.reshape(-1, self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, 3)
                
                # Extract embeddings
                feed_dict = {
                    self.embeddings_placeholder: prewhitened, 
                    self.phase_train_placeholder: False
                }
                embedding = self.sess.run(self.embeddings_op, feed_dict=feed_dict)
                embeddings_list.append(embedding[0])
            
            return embeddings_list
            
        except Exception as e:
            print(f"❌ Lỗi khi xử lý {image_path}: {e}")
            return None
    
    def retrain_model(self):
        """Train lại model với dữ liệu cải thiện"""
        print("🔄 Bắt đầu train lại model...")
        
        # Prepare data
        X = []
        y = []
        
        raw_data_path = "DataSet/FaceData/raw"
        for person_name in os.listdir(raw_data_path):
            person_path = os.path.join(raw_data_path, person_name)
            if not os.path.isdir(person_path):
                continue
                
            print(f"📁 Xử lý dữ liệu cho: {person_name}")
            
            for image_file in os.listdir(person_path):
                if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                image_path = os.path.join(person_path, image_file)
                embeddings_list = self.extract_embeddings(image_path)
                
                if embeddings_list:
                    for embedding in embeddings_list:
                        X.append(embedding)
                        y.append(person_name)
                    print(f"  ✅ {image_file}: {len(embeddings_list)} embeddings")
                else:
                    print(f"  ❌ {image_file}: Thất bại")
        
        if len(X) == 0:
            print("❌ Không có dữ liệu training!")
            return False
            
        X = np.array(X)
        y = np.array(y)
        
        print(f"📊 Tổng số mẫu: {len(X)}")
        for class_name in np.unique(y):
            count = np.sum(y == class_name)
            print(f"  - {class_name}: {count} mẫu")
        
        # Train SVM với hyperparameter tuning
        print("🤖 Training SVM...")
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'poly']
        }
        
        svm = SVC(probability=True, random_state=42)
        grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y)
        
        self.model = grid_search.best_estimator_
        self.class_names = np.unique(y)
        
        print(f"✅ Training hoàn thành!")
        print(f"🎯 Best parameters: {grid_search.best_params_}")
        print(f"📈 Best score: {grid_search.best_score_:.3f}")
        
        # Save model
        model_path = 'Models/improved_facemodel.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump((self.model, self.class_names), f)
        print(f"💾 Đã lưu model: {model_path}")
        
        return True
    
    def analyze_confusion(self):
        """Phân tích confusion matrix để hiểu lỗi phân loại"""
        if self.model is None:
            print("❌ Model chưa được load!")
            return
            
        print("🔍 Phân tích confusion matrix...")
        
        # Load test data
        X_test = []
        y_test = []
        
        raw_data_path = "DataSet/FaceData/raw"
        for person_name in os.listdir(raw_data_path):
            person_path = os.path.join(raw_data_path, person_name)
            if not os.path.isdir(person_path):
                continue
                
            for image_file in os.listdir(person_path):
                if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                image_path = os.path.join(person_path, image_file)
                embeddings_list = self.extract_embeddings(image_path)
                
                if embeddings_list:
                    # Chỉ lấy embedding đầu tiên để test
                    X_test.append(embeddings_list[0])
                    y_test.append(person_name)
        
        if len(X_test) == 0:
            print("❌ Không có dữ liệu test!")
            return
            
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        # Predict
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Classification report
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=self.class_names)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Analyze misclassifications
        print("\n🔍 Phân tích lỗi phân loại:")
        for i, (true_label, pred_label, proba) in enumerate(zip(y_test, y_pred, y_pred_proba)):
            if true_label != pred_label:
                max_proba = np.max(proba)
                print(f"❌ Mẫu {i}: {true_label} → {pred_label} (confidence: {max_proba:.3f})")
    
    def test_live_recognition(self):
        """Test nhận diện trực tiếp"""
        if self.model is None:
            # Load model
            try:
                with open('Models/improved_facemodel.pkl', 'rb') as f:
                    self.model, self.class_names = pickle.load(f)
                print("✅ Đã load improved model")
            except:
                with open('Models/facemodel.pkl', 'rb') as f:
                    self.model, self.class_names = pickle.load(f)
                print("✅ Đã load original model")
        
        print("🎥 Bắt đầu test nhận diện trực tiếp...")
        print("📋 Classes:", self.class_names)
        
        cap = VideoStream(src=0).start()
        
        while True:
            frame = cap.read()
            if frame is None:
                continue
                
            frame = imutils.resize(frame, width=600)
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            bounding_boxes, _ = align.detect_face.detect_face(
                frame, self.MINSIZE, self.pnet, self.rnet, self.onet, 
                self.THRESHOLD, self.FACTOR
            )
            
            if len(bounding_boxes) > 0:
                for bb in bounding_boxes:
                    bb = [max(0, int(coord)) for coord in bb[:4]]
                    
                    # Extract face
                    face = frame[bb[1]:bb[3], bb[0]:bb[2], :]
                    if face.size == 0:
                        continue
                    
                    # Preprocess
                    aligned = cv2.resize(face, (self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE))
                    prewhitened = facenet.prewhiten(aligned)
                    prewhitened = prewhitened.reshape(-1, self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, 3)
                    
                    # Extract embeddings
                    feed_dict = {
                        self.embeddings_placeholder: prewhitened, 
                        self.phase_train_placeholder: False
                    }
                    embedding = self.sess.run(self.embeddings_op, feed_dict=feed_dict)
                    
                    # Predict
                    prediction = self.model.predict_proba(embedding)
                    best_class_idx = np.argmax(prediction[0])
                    confidence = prediction[0][best_class_idx]
                    predicted_name = self.class_names[best_class_idx]
                    
                    # Display all probabilities for debugging
                    print(f"\n🔍 Debug info:")
                    for i, (name, prob) in enumerate(zip(self.class_names, prediction[0])):
                        print(f"  {name}: {prob:.3f}")
                    
                    # Display result
                    color = (0, 255, 0) if confidence > self.CONFIDENCE_THRESHOLD else (0, 0, 255)
                    text = f"{predicted_name}: {confidence:.3f}" if confidence > self.CONFIDENCE_THRESHOLD else "Unknown"
                    
                    cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                    cv2.putText(frame, text, (bb[0], bb[3] + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.putText(frame, f"Threshold: {self.CONFIDENCE_THRESHOLD}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow('Improved Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.stop()
        cv2.destroyAllWindows()

def main():
    recognizer = ImprovedFaceRecognition()
    recognizer.setup_tensorflow()
    
    print("🚀 Improved Face Recognition Tool")
    print("=" * 50)
    print("1. Retrain model với data augmentation")
    print("2. Phân tích confusion matrix") 
    print("3. Test live recognition")
    print("4. Thoát")
    
    while True:
        choice = input("\n👉 Chọn chức năng (1-4): ").strip()
        
        if choice == '1':
            recognizer.retrain_model()
        elif choice == '2':
            recognizer.analyze_confusion()
        elif choice == '3':
            recognizer.test_live_recognition()
        elif choice == '4':
            print("👋 Tạm biệt!")
            break
        else:
            print("❌ Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main() 