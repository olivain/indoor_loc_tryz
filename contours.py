# Detect a triangle using opencv findcontours

import cv2
import imutils

def detect(c):
	# initialize the shape name and approximate the contour
	shape = "unidentified"
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.04 * peri, True)

	if len(approx) == 3:
		shape = "triangle"
	else:
		shape = ""
	return shape

cam = cv2.VideoCapture(0)

while True:
	ret, image = cam.read()
	resized = imutils.resize(image, width=300)
	ratio = image.shape[0] / float(resized.shape[0])

	gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	thresh = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)[1]
	# find contours in the thresholded image and initialize the
	# shape detector
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	for c in cnts:
		# compute the center of the contour, then detect the name of the
		# shape using only the contour
		M = cv2.moments(c)

		if int(M["m00"]) > 0:
			cX = int((M["m10"] / M["m00"]) * ratio)
			cY = int((M["m01"] / M["m00"]) * ratio)
			print(cX)
			print(cY)
			shape = detect(c)

			# multiply the contour (x, y)-coordinates by the resize ratio,
			# then draw the contours and the name of the shape on the image
			c = c.astype("float")
			c *= ratio
			c = c.astype("int")
			cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
			cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 2)

	# show the output image
	cv2.imshow("Image", image)
	cv2.imshow("Ithresh", thresh)
	cv2.imshow("blr", blurred)
	cv2.waitKey(10)

cam.release()
