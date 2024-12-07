import tkinter as tk
from tkinter import ttk, filedialog
import cv2
from PIL import Image, ImageTk

from src.click_image import main
from src.color_classification_webcam import detect_objects_and_classify_color


class ColorDetectionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Phân Biệt Màu Sắc Sử Dụng KNN")
        self.window.geometry("800x800")  # Kích thước cửa sổ lớn hơn

        # Tạo style cho giao diện
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Arial', 24, 'bold'))
        style.configure('Subtitle.TLabel', font=('Arial', 14))
        style.configure('Big.TButton', font=('Arial', 12))

        # Tạo frame chính
        self.main_frame = ttk.Frame(self.window, padding="20")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Thêm tiêu đề
        title = ttk.Label(self.main_frame, 
                         text="HỆ THỐNG NHẬN DẠNG MÀU SẮC", 
                         style='Title.TLabel')
        title.grid(row=0, column=0, columnspan=3, pady=20)

        # Thêm phụ đề
        subtitle = ttk.Label(self.main_frame,
                             text="Sử dụng thuật toán KNN và Histogram màu",
                             style='Subtitle.TLabel')
        subtitle.grid(row=1, column=0, columnspan=3, pady=10)

        # Frame hiển thị webcam
        self.video_frame = ttk.Frame(self.main_frame, padding="10")
        self.video_frame.grid(row=2, column=0, columnspan=3, pady=20)

        self.label = ttk.Label(self.video_frame)
        self.label.grid(row=0, column=0)

        # Frame chứa các nút
        control_frame = ttk.Frame(self.main_frame)
        control_frame.grid(row=3, column=0, columnspan=3, pady=20)

        # Nút nhập ảnh
        self.image_button = ttk.Button(control_frame,
                                     text="Nhập ảnh",
                                     command=self.img_load,
                                     style='Big.TButton',
                                     width=20)
        self.image_button.grid(row=0, column=0, padx=10)

        # Nút bật/tắt camera
        self.cam_button = ttk.Button(control_frame,
                                   text="Bật Camera",
                                   command=self.toggle_camera,
                                   style='Big.TButton',
                                   width=20)
        self.cam_button.grid(row=0, column=1, padx=10)

        # Nút thoát
        self.quit_button = ttk.Button(control_frame,
                                    text="Thoát",
                                    command=self.quit_app,
                                    style='Big.TButton',
                                    width=20)
        self.quit_button.grid(row=0, column=2, padx=10)

        # Thông tin thêm
        info_text = "Các màu có thể nhận dạng: Đỏ, Xanh lá, Xanh dương, Vàng, Cam, Trắng, Đen, Tím"
        info_label = ttk.Label(self.main_frame, text=info_text, style='Subtitle.TLabel')
        info_label.grid(row=4, column=0, columnspan=3, pady=20)

        # Webcam variables
        self.cap = None
        self.is_running = False

        # Cấu hình grid
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)

    def toggle_camera(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            self.is_running = True
            self.cam_button.config(text="Tắt Camera")
            self.update_frame()
        else:
            self.is_running = False
            self.cam_button.config(text="Bật Camera")
            if self.cap:
                self.cap.release()
                self.label.config(image='')

    def img_load(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if file_path:
            img = cv2.imread(file_path)
            if img is not None:
                main(file_path)

    def update_frame(self):
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                processed_frame, color_predictions = detect_objects_and_classify_color(frame)
                processed_frame = cv2.resize(processed_frame, (400, 300))
                
                img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.label.imgtk = imgtk
                self.label.configure(image=imgtk)

            self.window.after(10, self.update_frame)

    def quit_app(self):
        if self.cap:
            self.cap.release()
        self.window.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = ColorDetectionApp(root)
    root.mainloop()
