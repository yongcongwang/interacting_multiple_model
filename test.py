#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# By yongcong.wang @ 13/10/2020
import math
import numpy as np

from kalman_filter import KalmanFilter
from imm import Imm
import data
from plot import *

dt = 0.1

def kf_cv():
    A = np.array([
        [1., dt, 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., dt],
        [0., 0., 0., 1.]
    ])
    H = np.array([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]
    ])
    return KalmanFilter(A, H)

def kf_ca():
    A = np.array([
        [1., dt, 0.5 * dt**2, 0., 0., 0.],
        [0., 1., dt, 0., 0., 0.],
        [0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 1., dt, 0.5 * dt**2],
        [0., 0., 0., 0., 1., dt],
        [0., 0., 1., 0., 0., 1.]
    ])
    H = np.array([
        [1., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 1., 0.]
    ])
    return KalmanFilter(A, H)

def kf_ct():
    dtheta = math.pi / 180 * 15
    theta = dtheta * dt
    A = np.array([
         [1., math.sin(theta)/dtheta, 0., -(1 - math.cos(theta))/dtheta, 0.],
         [0., math.cos(theta), 0., -math.sin(theta), 0.],
         [0., (1 - math.cos(theta)) / dtheta, 1., math.sin(theta)/dtheta, 0.],
         [0., math.sin(theta), 0., math.cos(theta), 0.],
         [0., 0., 0., 0., 1.],
         ])
    H = np.array([
        [1., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0.],
        [0., 0., 1., 0., 0.],
        [0., 0., 0., 1., 0.]
        ])
    return KalmanFilter(A, H)

def imm_cvat():
    P_trans = np.array([
        [0.98, 0.01, 0.01],
        [0.01, 0.98, 0.01],
        [0.01, 0.01, 0.98]
    ])
    U_prob = np.array([0.8, 0.1, 0.1]).reshape((-1, 1))

    models = [kf_cv(), kf_ca(), kf_ct()]
    r = np.array([
        [5.],
        [2.],
        [5.],
        [2.5]
    ])
    for model in models:
        model.R *= r

    T12 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    T23 = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0]
    ])
    model_trans = [
        [np.eye(models[0].A.shape[0]), T12.T, np.dot(T12.T, T23.T)],
        [T12, np.eye(models[1].A.shape[0]), T23.T],
        [np.dot(T23, T12), T23, np.eye(models[2].A.shape[0])]
    ]

    return Imm(models, model_trans, P_trans, U_prob)

def imm_cvt():
    P_trans = np.array([
        [0.98, 0.02],
        [0.02, 0.98]
    ])
    U_prob = np.array([0.5, 0.5]).reshape((-1, 1))

    models = [kf_cv(), kf_ct()]
    r = np.array([
        [10.],
        [1.],
        [10.],
        [1.]
    ])
    for model in models:
        model.R *= r

    T12 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    T23 = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0]
    ])
    T13 = np.dot(T23, T12)
    model_trans = [
        [np.eye(models[0].A.shape[0]), T13.T],
        [T13, np.eye(models[1].A.shape[0])]
    ]

    return Imm(models, model_trans, P_trans, U_prob)

def z_data():
    cnt = 100
    z_std = data.cv_z(0., 10., 0., 10., dt, cnt)
    z_std += data.ct_z(z_std[-1][0,0], z_std[-1][1,0],
                       z_std[-1][2,0], z_std[-1][3,0], math.pi/180*25, dt, cnt)
    z_std += data.ca_z(z_std[-1][0,0], z_std[-1][1,0], 6.,
                       z_std[-1][2,0], z_std[-1][3,0], 8., dt, cnt)

    return z_std

def test_cvt():
    z_std = z_data()
    z_noise = data.add_noise(z_std, np.array([
        [5.],
        [.2],
        [5.],
        [.2]
    ]))

    imm = imm_cvt();
    z0 = z_noise[0]
    imm.models[0].X = np.array([
        [z0[0, 0]],
        [z0[1, 0]],
        [z0[2, 0]],
        [z0[3, 0]]
        ])
    imm.models[1].X = np.array([
        [z0[0, 0]],
        [z0[1, 0]],
        [z0[2, 0]],
        [z0[3, 0]],
        [0.]
        ])

    prob = []
    z_filt = []
    for z in z_noise:
        prob.append(np.copy(imm.filt(z)))
        # merge
        x = np.zeros(imm.models[0].X.shape)
        for i in range(len(imm.models)):
            x += np.dot(imm.model_trans[0][i], imm.models[i].X) * prob[-1][i]
        z_filt.append(x)

    plot_position(
        [z[0,0] for z in z_std],
        [z[2,0] for z in z_std],
        [z[0,0] for z in z_noise],
        [z[2,0] for z in z_noise],
        [z[0,0] for z in z_filt],
        [z[2,0] for z in z_filt]
    )
    plot_speed(
        [z[1,0] for z in z_std],
        [z[3,0] for z in z_std],
        [z[1,0] for z in z_noise],
        [z[3,0] for z in z_noise],
        [z[1,0] for z in z_filt],
        [z[3,0] for z in z_filt]
    )
    plot_prob(
        [p[0,0] for p in prob],
        [p[1,0] for p in prob],
        [p[1,0] for p in prob],
    )
    plot_show()
#test_cvt()

def test_cvat():
    z_std = z_data()
    z_noise = data.add_noise(z_std, np.array([
        [5.],
        [2],
        [5.],
        [2]
    ]))

    imm = imm_cvat();
    z0 = z_noise[0]
    imm.models[0].X = np.array([
        [z0[0, 0]],
        [z0[1, 0]],
        [z0[2, 0]],
        [z0[3, 0]]
    ])
    imm.models[1].X = np.array([
        [z0[0, 0]],
        [z0[1, 0]],
        [0.],
        [z0[2, 0]],
        [z0[3, 0]],
        [0.]
    ])
    imm.models[2].X = np.array([
        [z0[0, 0]],
        [z0[1, 0]],
        [z0[2, 0]],
        [z0[3, 0]],
        [0.]
    ])

    prob = []
    z_filt = []
    for z in z_noise:
        prob.append(np.copy(imm.filt(z)))
        # merge
        x = np.zeros(imm.models[0].X.shape)
        for i in range(len(imm.models)):
            x += np.dot(imm.model_trans[0][i], imm.models[i].X) * prob[-1][i]
        z_filt.append(x)

    plot_position(
        [z[0,0] for z in z_std],
        [z[2,0] for z in z_std],
        [z[0,0] for z in z_noise],
        [z[2,0] for z in z_noise],
        [z[0,0] for z in z_filt],
        [z[2,0] for z in z_filt]
    )
    plot_speed(
        [z[1,0] for z in z_std],
        [z[3,0] for z in z_std],
        [z[1,0] for z in z_noise],
        [z[3,0] for z in z_noise],
        [z[1,0] for z in z_filt],
        [z[3,0] for z in z_filt]
    )
    plot_prob(
        [p[0,0] for p in prob],
        [p[1,0] for p in prob],
        [p[2,0] for p in prob],
    )
    plot_show()

#test_cvat()

def test_imm_veh():
    z_noise = data.veh_z_mia()

    imm = imm_cvat();
    z0 = z_noise[0]
    imm.models[0].X = np.array([
        [z0[0, 0]],
        [z0[1, 0]],
        [z0[2, 0]],
        [z0[3, 0]]
    ])
    imm.models[1].X = np.array([
        [z0[0, 0]],
        [z0[1, 0]],
        [0.],
        [z0[2, 0]],
        [z0[3, 0]],
        [0.]
    ])
    imm.models[2].X = np.array([
        [z0[0, 0]],
        [z0[1, 0]],
        [z0[2, 0]],
        [z0[3, 0]],
        [0.]
    ])

    prob = []
    z_filt = []
    for z in z_noise:
        prob.append(np.copy(imm.filt(z)))
        # merge
        x = np.zeros(imm.models[0].X.shape)
        for i in range(len(imm.models)):
            x += np.dot(imm.model_trans[0][i], imm.models[i].X) * prob[-1][i]
        z_filt.append(x)
        #return

    plot_position(
        [z[0,0] for z in z_noise],
        [z[2,0] for z in z_noise],
        [z[0,0] for z in z_noise],
        [z[2,0] for z in z_noise],
        [z[0,0] for z in z_filt],
        [z[2,0] for z in z_filt]
    )
    plot_speed(
        [z[1,0] for z in z_noise],
        [z[3,0] for z in z_noise],
        [z[1,0] for z in z_noise],
        [z[3,0] for z in z_noise],
        [z[1,0] for z in z_filt],
        [z[3,0] for z in z_filt]
    )
    plot_prob(
        [p[0,0] for p in prob],
        [p[1,0] for p in prob],
        [p[2,0] for p in prob],
    )
    plot_show()

def test_imm_veh_pred():
    z_noise = data.veh_z_mia()

    imm = imm_cvat();
    z0 = z_noise[0]
    imm.models[0].X = np.array([
        [z0[0, 0]],
        [z0[1, 0]],
        [z0[2, 0]],
        [z0[3, 0]]
    ])
    imm.models[1].X = np.array([
        [z0[0, 0]],
        [z0[1, 0]],
        [0.],
        [z0[2, 0]],
        [z0[3, 0]],
        [0.]
    ])
    imm.models[2].X = np.array([
        [z0[0, 0]],
        [z0[1, 0]],
        [z0[2, 0]],
        [z0[3, 0]],
        [0.]
    ])

    prob = []
    z_filt = []
    pred_z = []
    for z in z_noise:
        prob.append(np.copy(imm.filt(z)))
        # merge
        x = np.zeros(imm.models[0].X.shape)
        for i in range(len(imm.models)):
            x += np.dot(imm.model_trans[0][i], imm.models[i].X) * prob[-1][i]
        z_filt.append(x)

        # predict trajectory
        states = [imm.models[0].X.copy(),
                  imm.models[1].X.copy(),
                  imm.models[2].X.copy()]
        pred_step = []
        for i in range(50): # predict 5s
            for i in range(len(states)): # each model predict
                states[i] = np.dot(imm.models[i].A, states[i])
            x_step = np.zeros(x.shape)
            for i in range(len(imm.models)): # merge predict
                x_step += np.dot(imm.model_trans[0][i], states[i]) * prob[-1][i]
            pred_step.append(x_step.copy())
        pred_z.append(pred_step)
            

    plot_position(
        [z[0,0] for z in z_noise],
        [z[2,0] for z in z_noise],
        [z[0,0] for z in z_noise],
        [z[2,0] for z in z_noise],
        [z[0,0] for z in z_filt],
        [z[2,0] for z in z_filt]
    )
    plot_speed(
        [z[1,0] for z in z_noise],
        [z[3,0] for z in z_noise],
        [z[1,0] for z in z_noise],
        [z[3,0] for z in z_noise],
        [z[1,0] for z in z_filt],
        [z[3,0] for z in z_filt]
    )
    plot_prob(
        [p[0,0] for p in prob],
        [p[1,0] for p in prob],
        [p[2,0] for p in prob],
    )
    pred_x = []
    pred_y = []
    for step_z in pred_z:
        curr_x = [z[0,0] for z in step_z]
        pred_x.append(curr_x)
        curr_y = [z[2,0] for z in step_z]
        pred_y.append(curr_y)
    plot_prediction(
        [z[0,0] for z in z_noise],
        [z[2,0] for z in z_noise],
        pred_x,
        pred_y
    )

    plot_show()

test_imm_veh_pred()
