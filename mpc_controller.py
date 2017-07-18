import numpy as np


def mpc_controller(lane_middle, throtle_des, steering_actual):
    # x0 = np.array([lane_middle.d_error, lane_middle.delta_theta, steering_actual])
    x0 = np.array([lane_middle.d_error / 250, lane_middle.delta_theta])  # m and radian
    curv = lane_middle.predicted_curvature[1] * 250  # pixel / m
    print("Radius of the curve: ", 1/curv, ' [m]')
    Gd = 5  # BestrafungAbstandsfehler
    GTheta = 7.5  # Bestrafung Ausrichtungsfehler
    Gu = 0.5  # Bestrafung Lenkwinkelaenderung
    T = 1/12.5  # sec, smapling time

    L = 0.25  # m * pix/m
    v = 1.5  # m / sec insetad of pix/sec

    # simple controller, just 1 step and no gradient:
    h = Gu + (GTheta * T ** 2 * v ** 2) / L ** 2 + (Gd * T ** 4 * v ** 4) / (4 * L ** 2)
    f = [(Gd*T**2*v**2)/(2*L)], [(Gd*T**3*v**3)/(2*L) + (GTheta*T*v)/L]
    f = np.array(f)
    g = - (GTheta*T**2*v**2)/L - (Gd*T**4*v**4)/(4*L)
    steering_gradient = -(np.vdot(f, x0) + curv * g)
    steering_gradient /= h * 2
    print("Steering act:", steering_actual)
    return steering_gradient, throtle_des
