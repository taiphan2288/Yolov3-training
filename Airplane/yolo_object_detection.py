import cv2
import numpy as np
import glob
import random

from numpy.core.fromnumeric import mean, size


# Load Yolo
net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")

# Name custom object
classes = ["Airplane"]

# Images path
images_path = glob.glob(r"D:\OpenCV\VScode\YOLOV3 train\Airplane\images\*.jpg")


# Get the name of all layers of the network
layer_names = net.getLayerNames()
# Get the index of the output layers.
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Insert here the path of your images
random.shuffle(images_path)
# loop through all the images
for img_path in images_path:
    # Loading image
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)

    # set the input to the pre-trained deep learning network and obtain
    # the output predicted probabilities for each of the 1,000 ImageNet
    # classes
    #->grab the predictions
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []

    # for each detetion from each output layer get the confidence, class id, 
    # bounding box params and ignore weak detections
    # loop over each of the layer outputs
    for out in outs:
        # loop over each of the object detections
        for detection in out:
            # extract the class id (label) and confidence (as a probability) of the current object detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # discard weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0.5:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                # Object detected
                print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    ## perform the non maximum suppression given the scores defined before
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 2, color, 2)


    cv2.imshow("Image", img)
    key = cv2.waitKey(0)
    if key == 27:
        break

cv2.destroyAllWindows()

#Note:
#1: cv2.dnn.blobFromImage -> creates 4-dimensional blob from image. Optionally resizes and crops image from center, subtract mean values, scales values by scalefactor, swap Blue and Red channels.
