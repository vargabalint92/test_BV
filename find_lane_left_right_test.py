import cv2
from mpc_controller import *
import matplotlib.pyplot as plt

# from scipy import signal
#
# signal.find_peaks_cwt()


class Lane:
    def __init__(self):
        self.isLeft = False
        # The starting points to the the actuall position
        self.x_start = []
        self.y_start = []

        self.n = 5  # Averaging with the last n frames
        self.detected = False

        # Polynomial coefficients: y = A*x^2 + B*x + C
        # Each of A, B, C is a "list-queue" with max length n
        # List of old coeffitients
        self.A = []
        self.B = []
        self.C = []

        # Average of above of the coeffs
        self.A_avg = 0.
        self.B_avg = 0.
        self.C_avg = 0.

        # The actual coeffitients and the fitted points for plotting
        self.coeff = [0, 0, 0]  # np.array[0, 0, 0]
        self.fitx = 0
        self.fity = 0

        # Errors, the state variables:
        self.delta_theta_queue = []  # used only for middle lane, as it does not haev detection issue
        self.d_error_queue = []  # used only for middle lane, as it does not haev detection issue

        self.delta_theta = 0
        self.d_error = 0
        self.predicted_curvature = [0., 0., 0., 0., 0.]

    def fit_line(self, x, y, y_act, height, base, x_after_base, step):
        # print("The x values: ",x)
        if np.any(x[:] != 0):
            # if there is some usefull points, make the fit
            fit, res, _, _, _ = np.polyfit(y, x, 2, full=True)
            self.detected = True
        else:
            # well there is no really fit...
            fit = np.array([0, 0, 0])  # To see if we have the error here or by finding the lines
            self.detected = False
        fity = np.linspace(y_act - height, y_act - 1, height)
        fitx = fit[0] * fity ** 2 + fit[1] * fity + fit[2]
        # Compute orientation of the lane
        # only actual velaus are used, beause by dissapearing one or two lane
        # there is a problem iwth lane left-right -> it is quite complex to define wich is left and wich is roght
        # just see the find_lane_no_class.py
        # print("X_base is: ", self.x_start)
        # when the number of windows changing,
        # it is computed from window_height = np.int(binary_warped.shape[0] / nwindows)
        # print("Delta is: ", delta)

        # No queue, matching issues

        #     # Tricky point
        #     delta = [self.x_start[1] - self.x_start[0], step + 15]  # 15 is height of one step, hard coded
        #     # step variable is the actual step, the actual y position,
        # else:
        #     # but only, if we have enough data
        #     delta = [0, 0]
        # check the dokumenation on the np.acrtan2 func, it is not soo evident

        delta = [base - x_after_base, 2*15]  # 15 means the heights of one window, hard coded, must be changed
        # self.x_start = base
        if self.detected:
            self.delta_theta = np.arctan2(delta[0], delta[1])  # - np.pi / 2
            self.x_start.append(base)
            if len(self.x_start) > 5:
                _ = self.x_start.pop(0)

        # Wirte to class var
        self.coeff = fit
        self.A = fit[0]
        self.B = fit[1]
        self.C = fit[2]
        self.fitx = fitx
        self.fity = fity
        return

    def avg_coeff(self, coeff):
        q_full = len(self.A) >= self.n
        self.A.append(coeff[0])
        self.B.append(coeff[1])
        self.C.append(coeff[2])

        # Pop from index 0 if full
        # implementation of the fifo queue
        if q_full:
            _ = self.A.pop(0)
            _ = self.B.pop(0)
            _ = self.C.pop(0)

        # Simple average of line coefficients
        self.A_avg = np.mean(self.A)
        self.B_avg = np.mean(self.B)
        self.C_avg = np.mean(self.C)

        return  # self.A_avg, self.B_avg, self.C_avg # no need for return


def get_second_lane(x_actual, i_thresh_left, i_thresh_right, binary_warped, out_img, step, max_dist, lane):
    y_actuall = binary_warped.shape[0] - step

    if len(i_thresh_left) > 2:  # decide if left or right
        base = np.int(np.mean(i_thresh_left))
        if np.abs(x_actual - base) > max_dist:
            # new line found
            x, y, height, x_after_base = find_single_lane(binary_warped, out_img, base, step)
            lane.fit_line(x, y, y_actuall, height, base, x_after_base, step)
            lane.isLeft = True
            return

    if len(i_thresh_right) > 2:  # decide if left or right
        base = np.int(np.mean(i_thresh_right))
        if np.abs(x_actual - base) > max_dist:
            # new line found
            x, y, height, x_after_base = find_single_lane(binary_warped, out_img,base, step)
            lane.fit_line(x, y, y_actuall, height, base, x_after_base, step)
            lane.isLeft = False
            return


def get_x_act(y, coeff):
    return coeff[0] * y ** 2 + coeff[1] * y + coeff[2]


def get_middle_lane_pos(lane1, lane2, lane3):

    d_full = len(lane3.delta_theta_queue) >= 2
    foo = (lane1.delta_theta + lane2.delta_theta) / 2

    # print("actual avg delta_theta", foo)
    lane3.delta_theta_queue.append(foo)
    if d_full:
        _ = lane3.delta_theta_queue.pop(0)
    lane3.delta_theta = np.mean(lane3.delta_theta_queue)

    # d_error computing:
    x_avg = np.sum(lane1.x_start + lane2.x_start) / (len(lane1.x_start) + len(lane2.x_start))
    lane3.d_error_queue.append(x_avg)
    if d_full:
        _ = lane3.d_error_queue.pop(0)
    lane3.d_error = 200 - np.mean(lane3.d_error_queue)  # f___ing hard coded... 200 is the middle of the pic....
    return  # no need for return


def avoid_lane_jump(lane):
    lane.avg_coeff(lane.coeff)
    # if the new frame gives about 30% error (variation, difference)
    jump = np.abs(lane.A_avg - lane.coeff[0]) > (np.abs(lane.A_avg) * 0.3)
    jump = jump or np.abs(lane.B_avg - lane.coeff[1]) > (np.abs(lane.B_avg) * 0.3)
    jump = jump or np.abs(lane.C_avg - lane.coeff[0]) > (np.abs(lane.C_avg) * 0.3)
    if jump:
        lane.coeff = [lane.A_avg, lane.B_avg, lane.C_avg]

    return  # no need for return, write direct class variable


def find_single_lane(binary_warped, out_img, x_base, step):
    nwindows = 12  # max number of windows from the begin
    windows_width = 35  # the width of the windows in pixel, if it is larger minpix should be also more
    minpix = 25  # the minimum pixel which must be found to accept the window, trying different values is welcome
    # Sliding windows size (hight = PicHigh/nwindows)
    # Define windows and their size
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    # get from histogramm
    x_current = x_base
    lane_inds = []
    E = 0  # counting error

    # out_img = binary_warped
    starter_point = binary_warped.shape[0] - step
    sum_height = 0
    x_second = 0
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)

        win_y_low = starter_point - (window + 1) * window_height
        win_y_high = starter_point - window * window_height
        win_x_low = x_current - windows_width
        win_x_high = x_current + windows_width
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 140, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (
            nonzerox < win_x_high)).nonzero()[0]
        # Append these indices to the lists
        lane_inds.append(good_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_inds) > minpix:
            x_current = np.int(np.mean(nonzerox[good_inds]))
            E = 0
            # print("++")
        # else:
        elif (len(good_inds) < minpix / 2) or E == 2:  # no pixel at all or error two wins after
            break
        else:
            E += 1
            # Visaulize debugging...
        if sum_height == 2*window_height:  # The second rectangle
            x_second = x_current
        sum_height += window_height
        # cv2.imshow('Debug', out_img)
        # cv2.waitKey(0)

    cv2.imshow('Debug', out_img)
    # cv2.waitKey(0)
    lane_inds = np.concatenate(lane_inds)
    x = nonzerox[lane_inds]
    # print('len of x indeces: ', x)
    y = nonzeroy[lane_inds]
    # print('len of y indeces: ', y)
    return x, y, sum_height, x_second


def find_double_lane_points(binary_warped, lane1, lane2, lane_middle):

    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    lane1.detected = False
    lane2.detected = False
    step = -15  # starting step, watch out advance indexing,
    max_dist = 50  # for the matching if the histogram gives a new line or an already founded one
    hist_thresh = 255*5  # 5 black pixel in one pixel row
    # they must be updated anyway
    # lane1.fitx, lane1.fity = 0, 0
    # lane2.fitx, lane2.fity = 0, 0
    # lane1.A, lane1.B, lane1.C = 0, 0, 0
    # lane2.A, lane2.B, lane2.C = 0, 0, 0

    while step < binary_warped.shape[0]:
        step += 15
        histogram = np.sum(binary_warped[-15:], axis=0) if step == 0 \
            else np.sum(binary_warped[-15 - step:-step], axis=0)
        midpoint = np.int(histogram.shape[0] / 2)
        # print('Peaks: ', signal.find_peaks_cwt(histogram, np.arange(10, 200)))
        #  it does not work, too slow...
        i_thresh = np.where(histogram > hist_thresh)
        i_thresh = i_thresh[0]
        i_thresh_left = i_thresh[i_thresh < midpoint]
        i_thresh_right = i_thresh[i_thresh > midpoint]
        y_actuall = binary_warped.shape[0] - step
        # plot for debugging
        plt.plot(histogram)
        plt.title('histo')
        #plt.show()
        # the necessray number on the x points, which are above the threshhold
        left_start = len(i_thresh_left) > 2
        right_start = len(i_thresh_right) > 2
        if not left_start and not right_start:
            continue
        # else:  # Adding popping new starting points of the

        if not lane1.detected and not lane2.detected:
            if step >= 75:  # binary_warped.shape[0]:
                #  searching line starting points only until 5 steps  (15*5 = 75)
                print('step: ', step)
                break  # no line at all
            if left_start and right_start:
                # compute the starting point of the new lane
                base = np.int(np.mean(i_thresh_left))
                x, y, height, x_after_base = find_single_lane(binary_warped, out_img, base, step)
                lane1.fit_line(x, y, y_actuall, height, base, x_after_base, step)
                lane1.isLeft = True

                base = np.int(np.mean(i_thresh_right))
                x, y, height, x_after_base = find_single_lane(binary_warped, out_img, base, step)
                lane2.fit_line(x, y, y_actuall, height, base, x_after_base, step)
                lane2.isLeft = False

                break
            else:  # decide if left or right
                if left_start:
                    base = np.int(np.mean(i_thresh_left))
                    x, y, height, x_after_base = find_single_lane(binary_warped, out_img, base, step)
                    lane1.fit_line(x, y, y_actuall, height, base, x_after_base, step)
                elif right_start:
                    base = np.int(np.mean(i_thresh_right))
                    x, y, height, x_after_base = find_single_lane(binary_warped, out_img, base, step)
                    lane2.fit_line(x, y, y_actuall, height, base, x_after_base, step)
                else:
                    continue

        if (lane1.detected and not lane2.detected) or (not lane1.detected and lane2.detected):
            if step >= 105:  # binary_warped.shape[0]:  # Max search high
                #  searching line starting points only until 7 steps  (15*7 = 105)
                print('step: ', step)
                break  # no additinoal line
            if lane1.detected:
                x_actual = get_x_act(y_actuall, lane1.coeff)
                get_second_lane(x_actual, i_thresh_left, i_thresh_right, binary_warped, out_img, step, max_dist, lane2)
                print("second by lane1")
                if lane2.detected:
                    break
            elif lane2.detected:
                x_actual = get_x_act(y_actuall, lane2.coeff)
                get_second_lane(x_actual, i_thresh_left, i_thresh_right, binary_warped, out_img, step, max_dist, lane1)
                print("second by lane2")
                if lane1.detected:
                    break

        # print('step: ', step)

    # End of the while

    left_fitx  = lane1.fitx
    left_fity  = lane1.fity
    right_fitx = lane2.fitx
    right_fity = lane2.fity
    print("lane 1 is left?: ", lane1.isLeft, ',  lane 2 is left?:', lane2.isLeft)
    # it is ok ?

    lane_middle.coeff = (lane1.coeff + lane2.coeff) / 2
    avoid_lane_jump(lane_middle)

    # good...
    y = np.linspace(5, binary_warped.shape[0]-1, binary_warped.shape[0] - 5)
    lane_middle.avg_coeff(lane_middle.coeff)
    # avoid_lane_jump(lane_middle)
    middlex = get_x_act(y, [lane_middle.A_avg, lane_middle.B_avg, lane_middle.C_avg])
    get_middle_lane_pos(lane1, lane2, lane_middle)
    # print("X pos:", lane_middle.d_error)
    pts_left_ = np.array(np.transpose(np.vstack([left_fitx, left_fity])))
    pts_right_ = np.array(np.transpose(np.vstack([right_fitx, right_fity])))
    pts_middle_ = np.array(np.transpose(np.vstack([middlex, y])))

    return pts_left_, pts_right_, pts_middle_


def planning_trajectory(lane, throtle_des):
    # computing the future curvature for the MPC
    T = 1/12.5  # sampling time

    # look up table for throtle - pix/sec function
    v = 17  # pixel / sec....

    s = v * T  # The passed lenght of arc in T

    # The Formel for computing the arc of lenght with integral:
    # s = int sqrt(1 + d^2) dy from a to b
    # Where d = diff(A*y^2 + B*y + C, y)
    # The integrate iwth help of taylor
    # in closed from:
    # (log(B + h + 2A) +  h*(B + 2Ay) )/ (4A)
    # Where h = sqrt( (B + 2Ay)^2 + 1 )
    # we have to solve s = F(end) - F(start)
    # where end (in y) is the end of the arc segment
    # and start is the begin of the arc segment
    # be care that the we start from the "top"
    # y coordinate shows "downwards"...
    A = lane.A_avg
    B = lane.B_avg
    C = lane.C_avg
    curvature = 2 * A
    y = 240  # starting from the 240, height of the pic
    x = 200  # Middle of the pic
    for i in range(0, 5):
        curvature /= (1 + (2 * A * y + B) ** 2) ** 1.5
        # print("The predictied curvature is: ", curvature/0.004, ' m')
        alpha = s * curvature
        x += (1/curvature) * (1 - np.cos(alpha))
        y -= (1/curvature) * np.sin(alpha)
        lane.predicted_curvature[i] = curvature
        # foo = 4 * A ** 2 * y_act ** 2 + 4 * A * B * y_act + B ** 2 + 1
        # alpha1 = foo ** (1 / 2)  - (y_act * (8 * y_act * A ** 2 + 4 * B * A)) / (2 * foo ** (1 / 2))
        # alpha2 = (y_act * (8 * y_act * A ** 2 + 4 * B * A)) / (2 * foo ** (1 / 2)) * 0.5

        # end = - alpha1 + np.sqrt(alpha1 ** 2 + 4 * (s + alpha2*y_act**2 + alpha1 * y_act)*alpha2)
        # end /= (2*alpha2)

    # must be done...

    return

font = cv2.FONT_HERSHEY_SIMPLEX
def visualize(pts_left, pts_right, pts_middle, imgOrg, img, lane_middle, lane_right, lane_left, steering_actual):

    cv2.rectangle(imgOrg, (0, 0), (400, 60), (255, 255, 255), -1)
    # pos = 'Coeffizients right Lane:  '
    # pos += '%.3f, ' % lane_right.coeff[0]
    # pos += '%.1f, ' % lane_right.coeff[1]
    # pos += '%.1f' % lane_right.coeff[2]
    # cv2.putText(imgOrg, pos, (20, 20), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    # pos = 'Coeffizients left Lane:  '
    # pos += '%.3f, ' % lane_left.coeff[0]
    # pos += '%.1f, ' % lane_left.coeff[1]
    # pos += '%.1f' % lane_left.coeff[2]
    # cv2.putText(imgOrg, pos, (20, 50), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    #
    pos = 'Errors:  '
    pos += 'd_error = %.1f [pixel],  ' % lane_middle.d_error
    pos += 'd_theta = %.3f [rad]' % lane_middle.delta_theta
    cv2.putText(imgOrg, pos, (20, 20), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    pos = 'steering = %.3f [rad]' % steering_actual
    cv2.putText(imgOrg, pos, (20, 40), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    for k in range(0, pts_left.shape[0] - 1):
        x1 = int(pts_left[k, 0])
        y1 = int(pts_left[k, 1])
        x2 = int(pts_left[k + 1, 0])
        y2 = int(pts_left[k + 1, 1])
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 100), 5, cv2.LINE_AA)

    for k in range(0, pts_right.shape[0] - 1):
        x1 = int(pts_right[k, 0])
        y1 = int(pts_right[k, 1])
        x2 = int(pts_right[k + 1, 0])
        y2 = int(pts_right[k + 1, 1])
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 100), 5, cv2.LINE_AA)
    if lane_left.detected and lane_right.detected:
        for k in range(0, pts_middle.shape[0] - 1):
            x1 = int(pts_middle[k, 0])
            y1 = int(pts_middle[k, 1])
            x2 = int(pts_middle[k + 1, 0])
            y2 = int(pts_middle[k + 1, 1])
            cv2.line(img, (x1, y1), (x2, y2), (100, 200, 100), 5, cv2.LINE_AA)
    elif (not lane_left.detected and lane_right.detected) or (lane_left.detected and not lane_right.detected):

        # cv2.putText(img, "Passen Sie auf !", (75, 180), font, 0.75,(0,255,255),2,cv2.LINE_AA)
        for k in range(0, pts_middle.shape[0] - 1):
            x1 = int(pts_middle[k, 0])
            y1 = int(pts_middle[k, 1])
            x2 = int(pts_middle[k + 1, 0])
            y2 = int(pts_middle[k + 1, 1])
            cv2.line(img, (x1, y1), (x2, y2), (100, 200, 100), 5, cv2.LINE_AA)
    else:
        cv2.putText(imgOrg, "kein Fahrerassistenzsystem", (120, 180), font, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)  # Draw the text

    cv2.imshow('video', img)
    cv2.imshow('Original Video', imgOrg)
    cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return

    return
