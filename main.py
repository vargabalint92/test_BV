import lane_detection
from find_lane_left_right_test import *


lane_left = Lane()
lane_right = Lane()
lane_middle = Lane()

font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture('output.avi')
# cap = cv2.VideoCapture(0)
ret, img = cap.read()
rows, cols, ch = img.shape

# Perspective Transformation
side = 0  # on the original image we have a recatngle not a trapez
under = 50
oben = 285

L = 190  # height of the img
H = 400  # width of the image
side_birdseye = 140

# Rectange on the original Pic
pts0 = np.array([[side, oben], [cols - side, oben], [cols, rows - under], [0, rows - under]], np.int32)

# Transformation parameters
pts1 = np.float32([[side, oben], [cols - side, oben], [cols, rows - under], [0, rows - under]])
pts2 = np.float32([[0, 0], [H, 0], [H - side_birdseye, L], [side_birdseye, L]])
M = cv2.getPerspectiveTransform(pts1, pts2)

# This part must be integrated in ROS
# it must be called periodically
# the parameters could be placed in separted scritp
throtle_des = 0.1  # throtles from driver
steering_actual = 0
j = 850
j = 220
while cap.isOpened():
    j += 1
    cap.set(1, j)
    ret, img = cap.read()
    imgOrg = img
    img = cv2.warpPerspective(img, M, (H, L), flags=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
    cv2.imshow('Gray',gray)
    cv2.imshow('Equ',equ)

    ret, thresh = cv2.threshold(equ, 245, 255, cv2.THRESH_BINARY)
    ###########################################
    # Now we have the edges
    # Fit poly and draw on the original pic
    ###########################################

    pts_left, pts_right, pts_middle = find_double_lane_points(thresh, lane_left, lane_right, lane_middle)

    print("Next frame:", cap.get(1))
    # trajectory planning:
    planning_trajectory(lane_middle, throtle_des)
    steering_gradient, throtle_des = mpc_controller(lane_middle, throtle_des, steering_actual)
    steering_actual = steering_gradient  # *(1/12.5)


    visualize(pts_left, pts_right, pts_middle, imgOrg, img, lane_middle, lane_left, lane_right, steering_actual)
print("End of script")
cap.released()
cv2.destroyAllWindow()
