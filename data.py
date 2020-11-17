#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# By yongcong.wang @ 13/10/2020

import math
import numpy as np

def cv_z(x0, dx, y0, dy, dt, cnt):
    Z = [np.array([
            [x0],
            [dx],
            [y0],
            [dy]
        ])]
    for i in np.arange(1, cnt):
        Z.append(np.array([
            [Z[i-1][0, 0] + dx * dt],
            [dx],
            [Z[i-1][2, 0] + dy * dt],
            [dy]
            ]))

    return Z

def ca_z(x0, dx, ddx, y0, dy, ddy, dt, cnt):
    Z = [np.array([
            [x0],
            [dx],
            [y0],
            [dy]
        ])]
    for i in np.arange(1, cnt):
        Z.append(np.array([
            [Z[i-1][0,0] + Z[i-1][1,0] * dt + 0.5 * ddx * dt**2],
            [Z[i-1][1,0]+ ddx * dt],
            [Z[i-1][2,0] + Z[i-1][3,0] * dt + 0.5 * ddy * dt**2],
            [Z[i-1][3,0]+ ddy * dt]
            ]))

    return Z

def ct_z(x0, dx, y0, dy, dtheta, dt, cnt):
    Z = [np.array([
            [x0],
            [dx],
            [y0],
            [dy]
        ])]
    theta = math.atan2(dy, dx)
    v = math.hypot(dx, dy)
    for i in np.arange(1, cnt):
        theta += dtheta * dt
        Z.append(np.array([
            [Z[i-1][0, 0] + v * dt * math.cos(theta)],
            [v * math.cos(theta)],
            [Z[i-1][2, 0] + v * dt * math.sin(theta)],
            [v * math.sin(theta)]
            ]))

    return Z

def add_noise(Z, gain):
    return [np.copy(z) + gain * (np.random.random(z.shape) - 0.5) for z in Z]
