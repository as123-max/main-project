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
            cv2.rectangle(frame_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame_with_boxes, "violence detection".upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    return frame_with_boxes

def process_video_frames(queue, bot_token, chat_id, canvas_violence, canvas_live):
    MODEL_PATH = "D:\College\PRJ\FINAL PRJCT\Human-Violence-Detection-master/best.pt"

    cap = cv2.VideoCapture(0)  # Capture video from webcam
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return

    model = YOLO(MODEL_PATH)

    while True:
        ret, frame = cap.read()
        if ret:
            results = model(frame)[0]
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



            else:
                # Display the live video frame on canvas_live
                resized_frame_live = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized_frame_live = resize_to_fit_canvas(resized_frame_live, canvas_live)
                img_live = Image.fromarray(resized_frame_live)
                img_tk_live = ImageTk.PhotoImage(image=img_live)
                canvas_live.create_image(0, 0, anchor=tk.NW, image=img_tk_live)
                canvas_live.image = img_tk_live

    cap.release()
    queue.put(None)

def resize_to_fit_canvas(frame, canvas):
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    frame_height, frame_width, _ = frame.shape
    scale_factor = min(canvas_width / frame_width, canvas_height / frame_height)
    resized_frame = cv2.resize(frame, (int(frame_width * scale_factor), int(frame_height * scale_factor)))
    return resized_frame

def run_video_processing(canvas_violence, canvas_live, bot_token, chat_id):
    queue = Queue()
    threading.Thread(target=process_video_frames, args=(queue, bot_token, chat_id, canvas_violence, canvas_live), daemon=True).start()

    def run_code():
        video_canvas_violence.delete("all")
        video_canvas_live.delete("all")
        run_video_processing(video_canvas_violence, video_canvas_live, bot_token, chat_id)

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

    video_canvas_violence = tk.Canvas(root, width=400, height=400, bg="black")
    video_canvas_violence.pack(side=tk.LEFT, padx=(0, 10))

    video_canvas_live = tk.Canvas(root, width=400, height=400, bg="black")
    video_canvas_live.pack(side=tk.LEFT, padx=(10, 0))

    button_run = ttk.Button(bottom_frame, text="Run Code", command=run_code)
    button_run.grid(row=0, column=3)

    # Replace 'YOUR_BOT_TOKEN' and 'YOUR_CHAT_ID' with your actual bot token and chat ID
    bot_token = '5937101927:AAH7xCb57Wb3V79MNbHOEOH7gnGGMZrnfro'
    chat_id = '-1342691955'

    root.mainloop()