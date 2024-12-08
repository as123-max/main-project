import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog

import os
from datetime import datetime
import threading
import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk
from queue import Queue
from telegram import Bot
from io import BytesIO
import asyncio

THRESHOLD = 0.7
video_processing_thread = None
stop_event = threading.Event()
queue = Queue()  # Define the queue globally
model = None  # Define the YOLO model globally

def send_frame_to_telegram(bot_token, chat_id, frame_with_boxes):
    async def send_photo_async():
        bot = Bot(token=bot_token)
        try:
            # Convert frame to RGB format
            frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
            # Convert frame to Image object
            img = Image.fromarray(frame_rgb)
            # Save Image object to bytes
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            buffer.seek(0)
            # Send photo to Telegram
            await bot.send_photo(chat_id=chat_id, photo=buffer)
            print("Frame sent successfully to Telegram!")
        except Exception as e:
            print(f"Error occurred while sending frame to Telegram: {e}")

    asyncio.run(send_photo_async())

def draw_boxes_on_frame(frame_rgb, results):
    frame_with_boxes = frame_rgb.copy()
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > THRESHOLD:
            cv2.rectangle(frame_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 4)
            cv2.putText(frame_with_boxes, "violence detection".upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
    return frame_with_boxes

def process_video_frames(queue, bot_token, chat_id, canvas_violence, canvas_live):
    global model
    MODEL_PATH = "E:\Project\Pycharm/best.pt"

    cap = cv2.VideoCapture(0)  # Capture video from webcam
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return

    model = YOLO(MODEL_PATH)

    while not stop_event.is_set():

        ret, frame = cap.read()
        if ret:
            results = model(frame)[0]

            # Display the live video frame on canvas_live
            resized_frame_live = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame_live = resize_to_fit_canvas(resized_frame_live, canvas_live)
            img_live = Image.fromarray(resized_frame_live)
            img_tk_live = ImageTk.PhotoImage(image=img_live)
            canvas_live.create_image(0, 0, anchor=tk.NW, image=img_tk_live)
            canvas_live.image = img_tk_live

            frame_with_boxes = draw_boxes_on_frame(frame, results)

            if any(score > THRESHOLD for _, _, _, _, score, _ in results.boxes.data.tolist()):
                # Display the frame with bounding boxes on canvas_violence
                resized_frame_violence = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
                resized_frame_violence = resize_to_fit_canvas(resized_frame_violence, canvas_violence)
                img_violence = Image.fromarray(resized_frame_violence)
                img_tk_violence = ImageTk.PhotoImage(image=img_violence)
                canvas_violence.create_image(0, 0, anchor=tk.NW, image=img_tk_violence)
                canvas_violence.image = img_tk_violence

                # Send the frame with bounding boxes to Telegram
                send_frame_to_telegram(bot_token, chat_id, frame_with_boxes)

    cap.release()
    queue.put(None)

def resize_to_fit_canvas(frame, canvas):
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    frame_height, frame_width, _ = frame.shape
    scale_factor_width = canvas_width / frame_width
    scale_factor_height = canvas_height / frame_height

    if scale_factor_width < scale_factor_height:
        resized_frame_height = int(frame_height * scale_factor_width)
        resized_frame_width = canvas_width
    else:
        resized_frame_width = int(frame_width * scale_factor_height)
        resized_frame_height = canvas_height

    resized_frame = cv2.resize(frame, (resized_frame_width, resized_frame_height))
    return resized_frame

def start_live_video(canvas_violence, canvas_live):
    global video_processing_thread
    global stop_event
    stop_event.clear()
    video_processing_thread = threading.Thread(target=process_video_frames, args=(queue, bot_token, chat_id, canvas_violence, canvas_live), daemon=True)
    video_processing_thread.start()

def stop_live_video():
    global stop_event
    stop_event.set()

def browse_video_file(canvas_violence):
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
    if file_path:
        process_browsed_video(file_path, canvas_violence, video_canvas_live)

def process_browsed_video(file_path, canvas_violence, canvas_live):
    global model

    # Load video file
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {file_path}")
        return

    model = YOLO("E:\Project\Pycharm/best.pt")

    # Process video frames
    frame_count = 0
    violence_count = 0  # Counter for the number of frames with violence detected
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        frame_with_boxes = draw_boxes_on_frame(frame, results)
        # Display the frame on canvas_live
        resized_frame_live = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame_live = resize_to_fit_canvas(resized_frame_live, canvas_live)
        img_live = Image.fromarray(resized_frame_live)
        img_tk_live = ImageTk.PhotoImage(image=img_live)
        canvas_live.create_image(0, 0, anchor=tk.NW, image=img_tk_live)
        canvas_live.image = img_tk_live

        if any(score > THRESHOLD for _, _, _, _, score, _ in results.boxes.data.tolist()):
            # Increment violence count
            violence_count += 1

            if violence_count <= 2:  # Send only the first two frames with violence to Telegram
                # Display the frame with bounding boxes on canvas_violence
                resized_frame_violence = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
                resized_frame_violence = resize_to_fit_canvas(resized_frame_violence, canvas_violence)
                img_violence = Image.fromarray(resized_frame_violence)
                img_tk_violence = ImageTk.PhotoImage(image=img_violence)
                canvas_violence.create_image(0, 0, anchor=tk.NW, image=img_tk_violence)
                canvas_violence.image = img_tk_violence

                # Send the frame with bounding boxes to Telegram
                send_frame_to_telegram(bot_token, chat_id, frame_with_boxes)

        # Increment frame count
        frame_count += 1

        # Break the loop if first two frames with violence have been sent
        if violence_count >= 2:
            break

    cap.release()

def stop_browsed_video():
    pass  # Placeholder for stopping browsed video

def run_code():
    start_live_video(video_canvas_violence, video_canvas_live)

def update_clock():
    now = datetime.now()
    date_label.configure(text=now.strftime("%B %d, %Y"))
    day_label.configure(text=now.strftime("%A"))
    time_label.configure(text=now.strftime("%I:%M:%S %p"))
    root.after(1000, update_clock)

root = ctk.CTk()

root.title("Violence Detection and Alert in Real-time")
root.geometry("800x550")
def on_resize(event):
    global video_canvas_violence, video_canvas_live
    # Adjust the canvas size on window resize
    canvas_width = (root.winfo_width() - 60) // 2
    canvas_height = (root.winfo_height() - 220)
    video_canvas_violence.config(width=canvas_width, height=canvas_height)
    video_canvas_live.config(width=canvas_width, height=canvas_height)

root.bind('<Configure>', on_resize)

root.configure(bg="#F0F0F0")


top_frame = ctk.CTkFrame(root)
top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=15, pady=(0, 20))


bottom_frame = ctk.CTkFrame(root)
bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=(10, 10))

video_canvas_violence = ctk.CTkCanvas(root, width=450, height=350, bg="black", highlightthickness=0)
video_canvas_violence.pack(side=tk.LEFT, padx=(26, 10))

label_violence = ctk.CTkLabel(video_canvas_violence, text="   DETECTION   ", font=("Arial", 12, "bold"), fg_color="transparent")
label_violence.place(relx=0.5, rely=0.9, anchor=tk.CENTER)

video_canvas_live = ctk.CTkCanvas(root, width=450, height=350, bg="black", highlightthickness=0)
video_canvas_live.pack(side=tk.LEFT, padx=(15, 26))

label_live = ctk.CTkLabel(video_canvas_live, text="   LIVE   ", font=("Arial", 12, "bold"),fg_color='transparent')
label_live.place(relx=0.5, rely=0.9, anchor=tk.CENTER)

date_label = ctk.CTkLabel(top_frame, text="", font=("Eccentric Std", 20))
date_label.pack(side=tk.TOP,pady=(10, 0))

day_label = ctk.CTkLabel(top_frame, text="", font=("Arial", 16))
day_label.pack(side=tk.TOP)

time_label = ctk.CTkLabel(top_frame, text="", font=("Arial", 17))
time_label.pack(side=tk.TOP,pady=(0, 10))

update_clock()

button_start_live_video = ctk.CTkButton(bottom_frame, text="Start Live",fg_color="#5d7c1f",
                                     command=lambda: start_live_video(video_canvas_violence, video_canvas_live))
button_start_live_video.pack(side=tk.LEFT, padx=(150, 10), pady=(10,10), fill=tk.X, expand=True)

button_stop_live_video = ctk.CTkButton(bottom_frame, text="Stop Live",fg_color="#9a2929", command=stop_live_video)
button_stop_live_video.pack(side=tk.LEFT, padx=(10, 10), pady=(10,10), fill=tk.X, expand=True)

button_browse_video = ctk.CTkButton(bottom_frame, text="Browse Video",fg_color="#296380", command=lambda: browse_video_file(video_canvas_violence))
button_browse_video.pack(side=tk.LEFT, padx=(15, 150), pady=(10,10), fill=tk.X, expand=True)



# Replace 'YOUR_BOT_TOKEN' and 'YOUR_CHAT_ID' with your actual bot token and chat ID
bot_token = '5937101927:AAH7xCb57Wb3V79MNbHOEOH7gnGGMZrnfro'
chat_id = '-1002105316717'

root.mainloop()
