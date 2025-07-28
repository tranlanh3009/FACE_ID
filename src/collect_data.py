#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script thu thập dữ liệu training để cải thiện nhận diện
Đặc biệt để phân biệt Lanh và Việt
"""

import cv2
import os
import time
from imutils.video import VideoStream
import imutils

def collect_face_data(person_name, num_images=30):
    """Thu thập dữ liệu khuôn mặt"""
    
    # Tạo thư mục lưu dữ liệu
    save_path = f"DataSet/FaceData/raw/{person_name}"
    os.makedirs(save_path, exist_ok=True)
    
    print(f"🎯 Thu thập {num_images} ảnh cho: {person_name}")
    print(f"📁 Lưu vào: {save_path}")
    print("📋 Hướng dẫn:")
    print("  - Nhìn thẳng vào camera")
    print("  - Xoay mặt sang trái/phải nhẹ")
    print("  - Thay đổi biểu cảm (cười, nghiêm túc)")
    print("  - Thay đổi ánh sáng nếu có thể")
    print("  - Nhấn SPACE để chụp, 'q' để thoát")
    
    # Khởi tạo camera
    cap = VideoStream(src=0).start()
    time.sleep(2.0)  # Để camera khởi động
    
    count = 0
    captured_images = []
    
    while count < num_images:
        frame = cap.read()
        if frame is None:
            continue
            
        frame = imutils.resize(frame, width=600)
        frame = cv2.flip(frame, 1)
        
        # Hiển thị hướng dẫn
        cv2.putText(frame, f"Thu thap du lieu cho: {person_name}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Da chup: {count}/{num_images}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE: Chup anh, Q: Thoat", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Khung hướng dẫn vị trí khuôn mặt
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (w//4, h//6), (3*w//4, 5*h//6), (255, 255, 0), 2)
        cv2.putText(frame, "Dat khuon mat vao khung", (w//4, h//6 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.imshow('Thu thap du lieu', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space để chụp
            # Lưu ảnh với timestamp
            timestamp = int(time.time() * 1000)
            filename = f"{person_name}_{count+1:03d}_{timestamp}.jpg"
            filepath = os.path.join(save_path, filename)
            
            cv2.imwrite(filepath, frame)
            captured_images.append(filepath)
            count += 1
            
            print(f"✅ Đã chụp {count}/{num_images}: {filename}")
            
            # Hiệu ứng flash
            flash_frame = frame.copy()
            cv2.rectangle(flash_frame, (0, 0), (w, h), (255, 255, 255), -1)
            cv2.imshow('Thu thap du lieu', flash_frame)
            cv2.waitKey(100)
            
        elif key == ord('q'):  # Thoát
            break
    
    cap.stop()
    cv2.destroyAllWindows()
    
    print(f"\n🎉 Hoàn thành thu thập dữ liệu!")
    print(f"📊 Đã thu thập: {len(captured_images)} ảnh")
    print(f"📁 Lưu tại: {save_path}")
    
    return captured_images

def main():
    print("🚀 Thu thập dữ liệu training")
    print("=" * 40)
    
    while True:
        print("\n👤 Chọn người cần thu thập dữ liệu:")
        print("1. Lanh")
        print("2. Việt") 
        print("3. Người khác (nhập tên)")
        print("4. Thoát")
        
        choice = input("\n👉 Lựa chọn (1-4): ").strip()
        
        if choice == '1':
            person_name = "Lanh"
        elif choice == '2':
            person_name = "Viet"
        elif choice == '3':
            person_name = input("👤 Nhập tên: ").strip()
            if not person_name:
                print("❌ Tên không hợp lệ!")
                continue
        elif choice == '4':
            print("👋 Tạm biệt!")
            break
        else:
            print("❌ Lựa chọn không hợp lệ!")
            continue
        
        # Hỏi số lượng ảnh
        try:
            num_images = int(input(f"📸 Số ảnh muốn chụp cho {person_name} (mặc định 30): ") or 30)
            if num_images <= 0:
                num_images = 30
        except ValueError:
            num_images = 30
        
        # Thu thập dữ liệu
        collected = collect_face_data(person_name, num_images)
        
        if collected:
            print(f"\n💡 Gợi ý tiếp theo:")
            print(f"1. Chạy script improve_recognition.py")
            print(f"2. Chọn '1. Retrain model với data augmentation'")
            print(f"3. Test lại bằng face_rec_cam.py")

if __name__ == "__main__":
    main() 