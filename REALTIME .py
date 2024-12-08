import tkinter as tk
from tkinter import filedialog, ttk
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

def send_frame_to_telegram(bot_token, chat_id, frame_bgr, frame_count):
    async def send_photo_async():
        bot = Bot(token=bot_token)
        try:
            if frame_count < 2:
                # Convert frame to RGB format
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
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
            cv2.rectangle(frame_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame_with_boxes, "violence detection".upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    return frame_with_boxes

def process_video_frames(queue, bot_token, chat_id):
    MODEL_PATH = "E:\Project\Pycharm/best.pt"

    cap = cv2.VideoCapture(0)  # Capture video from webcam
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return

    model = YOLO(MODEL_PATH)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if ret:
            results = model(frame)[0]
            if any(score > THRESHOLD for _, _, _, _, score, _ in results.boxes.data.tolist()):
                queue.put(frame)
                send_frame_to_telegram(bot_token, chat_id, frame, frame_count)
                frame_count += 1
                if frame_count > 2:
                    break

    cap.release()
    queue.put(None)


def update_canvas(canvas, queue):
    while True:
        item = queue.get()
        if item is None:
            break
        frame = item  # Assuming frame is the only item in the queue
        resized_frame = resize_to_fit_canvas(frame, canvas)
        img = Image.fromarray(resized_frame)
        img_tk = ImageTk.PhotoImage(image=img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.image = img_tk


def resize_to_fit_canvas(frame, canvas):
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    frame_height, frame_width, _ = frame.shape
    scale_factor = min(canvas_width / frame_width, canvas_height / frame_height)
    resized_frame = cv2.resize(frame, (int(frame_width * scale_factor), int(frame_height * scale_factor)))
    return resized_frame

def run_video_processing(canvas, bot_token, chat_id):
    queue = Queue()
    threading.Thread(target=process_video_frames, args=(queue, bot_token, chat_id), daemon=True).start()
    threading.Thread(target=update_canvas, args=(canvas, queue), daemon=True).start()

def run_code():
    video_canvas.delete("all")
    run_video_processing(video_canvas, bot_token, chat_id)

def update_clock():
    now = datetime.now()
    date_label.config(text=now.strftime("%B %d, %Y"))
    day_label.config(text=now.strftime("%A"))
    time_label.config(text=now.strftime("%I:%M:%S %p"))
    root.after(1000, update_clock)

root = tk.Tk()
root.title("Video Processing Interface")
root.geometry("800x600")

root.configure(bg="#F0F0F0")
heading_label = tk.Label(root, text="VIOLENCE DETECTION", font=("Arial", 20), bg="#F0F0F0", fg="black")
heading_label.pack(pady=20)

top_frame = ttk.Frame(root, style='LightFrame.TFrame')
top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

date_label = ttk.Label(top_frame, text="", font=("Arial", 16), foreground="black")
date_label.pack(side=tk.TOP)

day_label = ttk.Label(top_frame, text="", font=("Arial", 16), foreground="black")
day_label.pack(side=tk.TOP)

time_label = ttk.Label(top_frame, text="", font=("Arial", 16), foreground="black")
time_label.pack(side=tk.TOP)

update_clock()

bottom_frame = ttk.Frame(root, style='LightFrame.TFrame')
bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=20, pady=(20, 0))

video_canvas = tk.Canvas(root, width=800, height=400, bg="black")
video_canvas.pack(pady=10)

button_run = ttk.Button(bottom_frame, text="Run Code", command=run_code)
button_run.grid(row=0, column=3)

# Replace 'YOUR_BOT_TOKEN' and 'YOUR_CHAT_ID' with your actual bot token and chat ID
bot_token = '5937101927:AAGF6fwFV5mva4jIUei5VX7moh0FReqDTAA'
chat_id = '-1342691955'
root.mainloop()
