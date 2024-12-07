import cv2
import numpy as np
import pandas as pd
import os
import math
import operator

from src.color_recognition_api.knn_classifier import kNearestNeighbors


# Hàm lấy màu sắc gần nhất từ k-NN
def getColorName(R, G, B, csv, k=3):
    minimum = 10000
    cname = None
    training_feature_vector = []  # vector đặc trưng cho tập huấn luyện

    # Tạo tập huấn luyện từ CSV
    for i in range(len(csv)):
        r = int(csv.loc[i, "R"])
        g = int(csv.loc[i, "G"])
        b = int(csv.loc[i, "B"])
        color_name = csv.loc[i, "color_name"]
        training_feature_vector.append([r, g, b, color_name])

    test_instance = [R, G, B]  # vector đặc trưng của pixel cần nhận diện
    neighbors = kNearestNeighbors(training_feature_vector, test_instance, k)

    # Bỏ phiếu để chọn màu sắc
    all_possible_neighbors = {}
    for neighbor in neighbors:
        color_name = neighbor[-1]
        if color_name in all_possible_neighbors:
            all_possible_neighbors[color_name] += 1
        else:
            all_possible_neighbors[color_name] = 1

    sortedVotes = sorted(all_possible_neighbors.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]  # trả về màu sắc có số phiếu cao nhất


def draw_function(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b, g, r, xpos, ypos, clicked
        clicked = True
        xpos = x
        ypos = y
        b, g, r = img[y, x]
        b = int(b)
        g = int(g)
        r = int(r)



from sklearn.metrics import f1_score, precision_score, recall_score


def calculate_f1_score(test_data, csv, k=3):
    true_labels = []
    predicted_labels = []

    # Lặp qua tập kiểm tra
    for test_instance in test_data:
        R, G, B, true_color = test_instance
        true_labels.append(true_color)  # Lưu nhãn thực
        predicted_color = getColorName(R, G, B, csv, k)  # Dự đoán
        predicted_labels.append(predicted_color)  # Lưu nhãn dự đoán

    # Tính F1 Score
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')

    return f1, precision, recall


def main(img_path):
    global img, clicked, r, g, b, xpos, ypos

    image_path = img_path  # Đặt ảnh trong cùng thư mục với file code
    csv_path = "training.data"  # Đặt file CSV trong cùng thư mục với file code

    # Kiểm tra file ảnh
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found")
        return

    # Kiểm tra file CSV
    if not os.path.exists(csv_path):
        print(f"Error: CSV file {csv_path} not found")
        return

    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read the image")
        return

    # Khởi tạo biến
    clicked = False
    r = g = b = xpos = ypos = 0

    # Đọc file CSV
    try:
        index = ["R", "G", "B", "color_name"]
        csv = pd.read_csv(csv_path, names=index, header=None)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Tạo cửa sổ và thiết lập callback
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_function)

    try:
        while True:
            cv2.imshow("image", img)
            if clicked:
                cv2.rectangle(img, (20, 20), (750, 60), (b, g, r), -1)
                text = getColorName(r, g, b, csv, k=3) + ' R=' + str(r) + ' G=' + str(g) + ' B=' + str(b)
                cv2.putText(img, text, (50, 50), 2, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                if (r + g + b >= 600):
                    cv2.putText(img, text, (50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                clicked = False
            if cv2.waitKey(20) & 0xFF == 27:  # ESC key
                break
    except KeyboardInterrupt:
        print("Program interrupted by user")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
