import os
import cv2
import numpy as np
import math
from cvzone.ColorModule import ColorFinder
import cvzone
import argparse
import importlib.util
from tkinter import *
from tkinter import filedialog
import PIL
from PIL import ImageTk

class WelcomePage:
    def __init__(self, root):
        self.root = root
        self.root.title("Ball Pathway Prediction Application")

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        x = (screen_width - 800) // 2
        y = (screen_height - 600) // 2

        self.root.geometry(f"800x600+{x}+{y}")
        self.root.configure(bg="white")

        Frame_1stPage = Frame(self.root, bg="white")
        Frame_1stPage.place(relx=0.5, rely=0.5, anchor=CENTER)

        title = Label(Frame_1stPage, text="Ball Pathway Prediction Application", font=("Impact", 35, "bold"), fg="#6162FF", bg="white")
        title.pack(pady=10)
        subtitle = Label(Frame_1stPage, text="Practice Make Perfect", font=("Goudy old style", 15), bg="#E7E6E6")
        subtitle.pack(pady=10)

        get_started_button = Button(Frame_1stPage, text="Get Start", font=("Arial", 14), bg="blue", fg="white", padx=20, pady=10, command=self.open_main_page)
        get_started_button.pack(pady=20)

        subtitle = Label(Frame_1stPage,
                         text="The ball path prediction software is a piece of software that simulates and displays a ball's path. \n"
                              "For coaches, students, and athletes, this app is helpful in physics and mechanics. \n"
                              "Sports including tennis, golf, basketball, football, and others are encouraged. \n"
                              "It gives a realistic 2D representation of the ball's route, which helps the user to understand \n"
                              "how different conditions affect the ball's path. This project includes a basketball ball path prediction program. \n"
                              "In particular, the Ball Path Prediction Application is a very useful application for basketball \n",
                         font=("Goudy old style", 12), bg="#E7E6E6")
        subtitle.pack(pady=10)

    def open_main_page(self):
        self.root.destroy()
        next_window = Tk()
        MainPage(next_window)

class MainPage:
    def __init__(self, root):
        self.root = root
        self.root.title("Prediction")

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        x = (screen_width - 800) // 2
        y = (screen_height - 600) // 2

        self.root.geometry(f"800x600+{x}+{y}")
        self.root.configure(bg="white")
        self.root.resizable(False, False)

        source_label = Label(self.root, text="Source:", font=("Arial", 16), bg="white")
        source_label.place(x=50, y=50)

        self.file_path = StringVar()
        input_field = Entry(self.root, textvariable=self.file_path, width=50, font=("Arial", 12))
        input_field.place(x=150, y=50)

        self.error_label = Label(self.root, text="", font=("Arial", 12), fg="red", bg="white")
        self.error_label.place(x=150, y=80)

        browse_button = Button(self.root, text="Browse", font=("Arial", 12), command=self.browse_file)
        browse_button.place(x=620, y=45)

        play_button = Button(self.root, text="Replay", font=("Arial", 12), command=self.play_video)
        play_button.place(x=700, y=45)

        self.prediction_button = Button(self.root, text="Predict", font=("Arial", 14), bg="#FF5733", fg="white", padx=20, pady=10, command=self.predict)
        self.prediction_button.place(x=350, y=100)
        # self.prediction_button.pack(pady=100)

        self.video_label = Label(self.root, bg="white")
        self.video_label.place(x=80, y=150)

    def browse_file(self):
        self.error_label.config(text="")
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_path.set(file_path)
            self.play_video()
        else:
            self.error_label.config(text="Please select a file.")

    def play_video(self):
        file_path = self.file_path.get()
        if file_path:
            self.file_path.set(file_path)

            cap = cv2.VideoCapture(file_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 480))
                photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
                self.video_label.config(image=photo)
                self.video_label.image = photo
                self.root.update_idletasks()
            cap.release()
        else:
            self.error_label.config(text="Please select a file.")

    def predict(self):
        file_path = self.file_path.get()

        if file_path:
            # Function to initialize object detection
            def initialize_object_detection(modeldir='custom_model_lite', graph='detect.tflite', labels='labelmap.txt',
                                            threshold=0.5, edgetpu=False):
                # Import TensorFlow libraries
                pkg = importlib.util.find_spec('tflite_runtime')
                if pkg:
                    from tflite_runtime.interpreter import Interpreter

                    if edgetpu:
                        from tflite_runtime.interpreter import load_delegate
                else:
                    from tensorflow.lite.python.interpreter import Interpreter

                    if edgetpu:
                        from tensorflow.lite.python.interpreter import load_delegate

                # If using Edge TPU, assign filename for Edge TPU model
                if edgetpu:
                    if graph == 'detect.tflite':
                        graph = 'edgetpu.tflite'

                # Get path to current working directory
                CWD_PATH = os.getcwd()

                # Path to .tflite file, which contains the model that is used for object detection
                PATH_TO_CKPT = os.path.join(CWD_PATH, modeldir, graph)

                # Path to label map file
                PATH_TO_LABELS = os.path.join(CWD_PATH, modeldir, labels)

                # Load the label map
                with open(PATH_TO_LABELS, 'r') as f:
                    labels = [line.strip() for line in f.readlines()]

                # Load the Tensorflow Lite model.
                # If using Edge TPU, use special load_delegate argument
                if edgetpu:
                    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
                else:
                    interpreter = Interpreter(model_path=PATH_TO_CKPT)

                interpreter.allocate_tensors()

                # Get model details
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                height = input_details[0]['shape'][1]
                width = input_details[0]['shape'][2]

                floating_model = (input_details[0]['dtype'] == np.float32)

                input_mean = 127.5
                input_std = 127.5

                # Check output layer name to determine if this model was created with TF2 or TF1
                outname = output_details[0]['name']
                if 'StatefulPartitionedCall' in outname:  # This is a TF2 model
                    boxes_idx, classes_idx, scores_idx = 1, 3, 0
                else:  # This is a TF1 model
                    boxes_idx, classes_idx, scores_idx = 0, 1, 2

                return interpreter, input_details, output_details, labels, height, width, floating_model, input_mean, input_std, boxes_idx, classes_idx, scores_idx

            # Function to detect objects in a frame
            def detect_objects(frame, interpreter, input_details, output_details, labels, height, width, floating_model,
                               input_mean, input_std, boxes_idx, classes_idx, scores_idx, imH, imW):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (width, height))
                input_data = np.expand_dims(frame_resized, axis=0)

                if floating_model:
                    input_data = (np.float32(input_data) - input_mean) / input_std

                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
                classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
                scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

                max_score_idx = np.argmax(scores)

                ymin = int(max(1, (boxes[max_score_idx][0] * imH)))
                xmin = int(max(1, (boxes[max_score_idx][1] * imW)))
                ymax = int(min(imH, (boxes[max_score_idx][2] * imH)))
                xmax = int(min(imW, (boxes[max_score_idx][3] * imW)))

                object_name = labels[int(classes[max_score_idx])]
                confidence = scores[max_score_idx]

                return xmin, ymin, xmax, ymax, object_name, confidence

            # Function to initialize video capture
            def initialize_video_capture(video_path):
                video = cv2.VideoCapture(video_path)
                imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
                imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
                return video, imW, imH

            # Define and parse input arguments
            parser = argparse.ArgumentParser()
            parser.add_argument('--modeldir', default='custom_model_lite', help='Folder the .tflite file is located in',
                                required=False)
            parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                                default='detect.tflite')
            parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                                default='labelmap.txt')
            parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                                default=0.5)
            parser.add_argument('--video', help='Name of the video file',
                                default=file_path)
            parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                                action='store_true')
            args = parser.parse_args()

            # Initialize object detection
            interpreter, input_details, output_details, labels, height, width, floating_model, input_mean, input_std, boxes_idx, classes_idx, scores_idx = initialize_object_detection(
                args.modeldir, args.graph, args.labels, float(args.threshold), args.edgetpu)

            # Initialize video capture
            video, imW, imH = initialize_video_capture(args.video)

            # Create the color Finder object
            myColorFinder = ColorFinder(False)

            # Variables
            posListX, posListY = [], []
            xList = [item for item in range(0, 1300)]
            prediction = False

            while video.isOpened():
                # Acquire frame and resize to expected shape [1xHxWx3]
                ret, frame = video.read()
                if not ret:
                    print('Reached the end of the video!')
                    break

                # Perform object detection
                xmin, ymin, xmax, ymax, object_name, confidence = detect_objects(frame, interpreter, input_details,
                                                                                 output_details, labels, height, width,
                                                                                 floating_model, input_mean, input_std,
                                                                                 boxes_idx, classes_idx, scores_idx,
                                                                                 imH, imW)

                # Draw bounding box and label
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 4)
                label = '%s: %.2f' % (object_name, confidence)
                cv2.putText(frame, label, (xmin, ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                # Image Processing for color detection
                img = frame[0:900, :]
                imgColor, mask = myColorFinder.update(img, {'hmin': 8, 'smin': 96, 'vmin': 115, 'hmax': 14, 'smax': 255,
                                                            'vmax': 255})
                imgContours, contours = cvzone.findContours(img, mask, minArea=500)

                # Contour Analysis
                if contours:
                    posListX.append(contours[0]['center'][0])
                    posListY.append(contours[0]['center'][1])

                # Polynomial Regression and Visualization
                if posListX:
                    A, B, C = np.polyfit(posListX, posListY, 2)

                    for i, (posX, posY) in enumerate(zip(posListX, posListY)):
                        pos = (posX, posY)
                        cv2.circle(imgContours, pos, 10, (0, 255, 0), cv2.FILLED)
                        if i == 0:
                            cv2.line(imgContours, pos, pos, (0, 255, 0), 5)
                        else:
                            cv2.line(imgContours, pos, (posListX[i - 1], posListY[i - 1]), (0, 255, 0), 5)

                    for x in xList:
                        y = int(A * x ** 2 + B * x + C)
                        cv2.circle(imgContours, (x, y), 2, (255, 0, 255), cv2.FILLED)

                    if len(posListX) < 10:
                        a = A
                        b = B
                        c = C - 590
                        x = int((-b - math.sqrt(b ** 2 - (4 * a * c))) / (2 * a))
                        prediction = 330 < x < 430

                # Result Display
                if prediction:
                    cvzone.putTextRect(imgContours, "Basket", (50, 100),
                                       scale=5, thickness=5, colorR=(0, 200, 0), offset=20)
                    print("Basket")
                else:
                    cvzone.putTextRect(imgContours, "No Basket", (50, 100),
                                       scale=5, thickness=5, colorR=(0, 0, 200), offset=20)
                    print("No Basket")

                # Display the frame with bounding box and color detection
                imgContours = cv2.resize(imgContours, (0, 0), None, 0.7, 0.7)
                cv2.imshow('Prediction', imgContours)

                # Press 'q' to quit
                if cv2.waitKey(1) == ord('q'):
                    break

            # Clean up
            video.release()
            # cv2.destroyAllWindows()
        else:
            self.error_label.config(text="Please select a file.")

root = Tk()
obj = WelcomePage(root)
root.mainloop()