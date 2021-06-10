from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
from sklearn.cluster import KMeans
from itertools import combinations
from math import sin, cos, atan


def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
	[0, 0],
	[maxWidth - 1, 0],
	[maxWidth - 1, maxHeight - 1],
	[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped


def _get_intersections(img, lines):
    """Finds the intersections between groups of lines."""
    intersections = []
    group_lines = combinations(range(len(lines)), 2)
    x_in_range = lambda x: 0 <= x <= img.shape[1]
    y_in_range = lambda y: 0 <= y <= img.shape[0]
    for i, j in group_lines:
      line_i, line_j = lines[i][0], lines[j][0]
      if 80.0 < _get_angle_between_lines(line_i, line_j) < 100.0:
          int_point = _intersection(line_i, line_j)
          if x_in_range(int_point[0][0]) and y_in_range(int_point[0][1]):
              intersections.append(int_point)
    return intersections


def _get_angle_between_lines(line_1, line_2):
    rho1, theta1 = line_1
    rho2, theta2 = line_2
    # x * cos(theta) + y * sin(theta) = rho
    # y * sin(theta) = x * (- cos(theta)) + rho
    # y = x * (-cos(theta) / sin(theta)) + rho
    m1 = -(np.cos(theta1) / np.sin(theta1))
    m2 = -(np.cos(theta2) / np.sin(theta2))
    return abs(atan(abs(m2-m1) / (1 + m2 * m1))) * (180 / np.pi)


def _intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.
    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
      [np.cos(theta1), np.sin(theta1)],
      [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]

def _find_quadrilaterals(intersections):
    X = np.array([[point[0][0], point[0][1]] for point in intersections])
    kmeans = KMeans(
      n_clusters = 4,
      init = 'k-means++',
      max_iter = 100,
      n_init = 10,
      random_state = 0
	  ).fit(X)

    return  [[center.tolist()] for center in kmeans.cluster_centers_]



def get_scan(image, show=False):
	# load the image and compute the ratio of the old height
	# to the new height, clone it, and resize it
	ratio = image.shape[0] / 1000.0
	orig = image.copy()
	resized = imutils.resize(image, height = 1000)

	print("STEP 1: Edge Detection")
	denoise = cv2.fastNlMeansDenoising(resized, h = 7)
	gray = cv2.cvtColor(denoise, cv2.COLOR_BGR2GRAY)
	# gray = cv2.equalizeHist(gray)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
	close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations = 5)
	edge = cv2.Canny(close, 75, 150, apertureSize = 3)
	edge = cv2.copyMakeBorder(edge, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, 255)


	print("STEP 2: Find contours of document")
	lines = cv2.HoughLines(edge, 1, np.pi / 180, 100)
	intersections = _get_intersections(edge, lines)
	quad = _find_quadrilaterals(intersections)

	# show the original image and the edge detected image

	if show:
		cv2.imwrite("./img_debug/1_gray.jpg", gray)
		cv2.imwrite("./img_debug/2__close.jpg", close)
		cv2.imwrite("./img_debug/3_edge.jpg", edge)
		contour = resized.copy()
		umat = np.float32(quad)
		xSorted = sorted(umat, key = lambda x: x[0][0])
		leftMost = xSorted[:2]
		rightMost = xSorted[2:]
		(tl , bl) = sorted(leftMost, key= lambda x: x[0][1])
		(tr, br) = sorted(rightMost, key = lambda x: x[0][1])
		print([tl, tr, br, bl])
		umat = np.float32([tl, tr, br, bl])
		poly = cv2.approxPolyDP(umat, 0.03 * cv2.arcLength(umat, True), True)
		ctr = np.array(umat).reshape((-1,1,2)).astype(np.int32)
		cv2.drawContours(contour, [ctr], -1, (0,0,255), 5)
		cv2.imwrite("./img_debug/3_contours.jpg", contour)


	# apply the four point transform to obtain a top-down
	# view of the original image
	warped = four_point_transform(orig, np.array(quad).reshape(4, 2) * ratio)
	# convert the warped image to grayscale, then threshold it
	# to give it that 'black and white' paper effect
	warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	T = threshold_local(warped, 25, offset = 10, method = "gaussian")
	warped = (warped > T).astype("uint8") * 255
	x,y = warped.shape
	# some padding to erase black pixels of image borders
	warped = warped[75:x-75, 25:y-25]
	warped = cv2.medianBlur(warped, 5)
	print("STEP 3: Apply perspective transform")
	if show:
		cv2.imwrite("./img_debug/4_scan.jpg", warped)

	return warped
