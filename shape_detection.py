import cv2
import numpy as np
from tkinter import Tk, Button, Label, Frame, Text, Scrollbar, VERTICAL, RIGHT, Y

frameWidth = 640
frameHeight = 480

def open_camera():
    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)

    def empty(a):
        pass

    cv2.namedWindow("Parameters")
    cv2.resizeWindow("Parameters", 640, 240)
    cv2.createTrackbar("Threshold1", "Parameters", 23, 255, empty)
    cv2.createTrackbar("Threshold2", "Parameters", 20, 255, empty)
    cv2.createTrackbar("Kernel", "Parameters", 1, 20, empty)
    cv2.createTrackbar("Area", "Parameters", 5000, 30000, empty)

    def stackImages(scale, imgArray):
        rows = len(imgArray)
        cols = len(imgArray[0])
        rowsAvailable = isinstance(imgArray[0], list)
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]
        if rowsAvailable:
            for x in range(0, rows):
                for y in range(0, cols):
                    if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                    else:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                    if len(imgArray[x][y].shape) == 2: 
                        imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height, width, 3), np.uint8)
            hor = [imageBlank] * rows
            for x in range(0, rows):
                hor[x] = np.hstack(imgArray[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                    imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
                else:
                    imgArray[x] = cv2.resize(imgArray[x], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x].shape) == 2: 
                    imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
            hor = np.hstack(imgArray)
            ver = hor
        return ver

    def getContours(img, imgContour):
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            areaMin = cv2.getTrackbarPos("Area", "Parameters")
            if area > areaMin:
                cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                vertices = len(approx)
                x, y, w, h = cv2.boundingRect(approx)
                aspectRatio = float(w) / float(h)

                shapeType = "Unidentified"
                if vertices == 3:
                    shapeType = "Triangle"
                elif vertices == 4:
                    if 0.95 < aspectRatio < 1.05:
                        shapeType = "Square"
                    else:
                        shapeType = "Rectangle"
                elif vertices > 4:
                    if 0.8 < aspectRatio < 1.2 and peri / (2 * np.pi * (area / peri)) > 0.9:
                        shapeType = "Circle"
                    else:
                        shapeType = "Unidentified"

                print(f"Vertices: {vertices}, Shape: {shapeType}, Area: {int(area)}")

                cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)
                cv2.putText(imgContour, f"Shape: {shapeType}", (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(imgContour, f"Vertices: {vertices}", (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(imgContour, f"Area: {int(area)}", (x + w + 20, y + 70), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from camera.")
            break

        imgContour = img.copy()
        imgBlur = cv2.bilateralFilter(img, 9, 75, 75)
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
        
        threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
        threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
        kernelSize = cv2.getTrackbarPos("Kernel", "Parameters")
        kernelSize = max(1, kernelSize) | 1
        kernel = np.ones((kernelSize, kernelSize), np.uint8)

        imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
        imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
        getContours(imgDil, imgContour)
        
        imgStack = stackImages(0.8, ([img, imgCanny, imgGray],
                                     [imgDil, imgContour, np.zeros_like(img)]))

        cv2.imshow("Result", imgStack)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

root = Tk()
root.title("Shape Detection Project")
root.geometry("900x600")
root.configure(bg="#f0f0f0")

frame = Frame(root, bg="#f0f0f0", padx=20, pady=20)
frame.pack(expand=True)

label_title = Label(frame, text="Shape Detection Project", font=("Arial", 24, "bold"), bg="#f0f0f0", fg="#333")
label_title.pack(pady=10)

scrollbar = Scrollbar(frame, orient=VERTICAL)
readme_text = Text(frame, wrap="word", yscrollcommand=scrollbar.set, height=20, font=("Arial", 12), bg="#fff", padx=10, pady=10)
scrollbar.config(command=readme_text.yview)
scrollbar.pack(side=RIGHT, fill=Y)
readme_text.pack(side="left", fill="both", expand=True)

readme_content = """
Shape Detection Project

This project demonstrates real-time shape detection using OpenCV. By processing live webcam feeds, it identifies shapes such as triangles, rectangles, squares, and circles.

Features
- Real-time shape detection and classification.
- Adjustable edge detection thresholds and shape area constraints.
- Dynamic visual feedback with labeled contours.

Technologies Used
- OpenCV for computer vision tasks.
- Tkinter for graphical user interface.

How It Works
1. Grayscale conversion simplifies image processing.
2. Bilateral filtering reduces noise while preserving edges.
3. Canny edge detection identifies object boundaries.
4. Dilation enhances edge visibility for contour detection.
5. Contours are analyzed to classify shapes based on vertices and aspect ratio.

How to Use
1. Click the "Detect Shapes in Real Time" button.
2. Use sliders in the Parameters window to adjust settings.
3. Observe shapes and classifications displayed in real time.
"""

readme_text.insert("1.0", readme_content)
readme_text.config(state="disabled")

# Button to open the camera
button = Button(frame, text="Detect Shapes in Real Time", command=open_camera, font=("Arial", 16),
                bg="#0078d7", fg="white", padx=20, pady=10)
button.pack(pady=20)

root.mainloop()
