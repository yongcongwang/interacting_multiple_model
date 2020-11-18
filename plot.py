#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# By yongcong.wang @ 13/10/2020

import matplotlib.pyplot as plt
import numpy as np

def plot_prediction(noise_x, noise_y, pred_x, pred_y):
    plt.figure(2)
    plt.plot(noise_x, noise_y, color='r', linewidth=4, label='pos_real')
    for i in range(len(pred_x)):
        plt.plot(pred_x[i], pred_y[i], color='g', alpha=0.6)
    plt.legend()

def plot_position(std_x, std_y, noise_x, noise_y, filt_x, filt_y):
    plt.figure(1)

    plt.subplot(321)
    plt.plot(std_x, std_y, color='r', label='pos_std')
    plt.plot(noise_x, noise_y, color='b', label='pos_noise')
    plt.plot(filt_x, filt_y, color='g', label='pos_filt')
    plt.legend()

    plt.subplot(322)
    t = np.arange(0, len(std_x))
    plt.plot(t, [noise_x[i] - std_x[i] for i in range(len(std_x))],
             color='r', label='pos_err_noise_x')
    plt.plot(t, [noise_y[i] - std_y[i] for i in range(len(std_y))],
             color='b', label='pos_err_noise_y')
    plt.plot(t, [filt_x[i] - std_x[i] for i in range(len(std_x))],
             color='g', label='pos_err_filt_x')
    plt.plot(t, [filt_y[i] - std_y[i] for i in range(len(std_y))],
             color='c', label='pos_err_filt_y')
    plt.legend()

def plot_speed(std_dx, std_dy, noise_dx, noise_dy, filt_dx, filt_dy):
    plt.figure(1)

    plt.subplot(323)
    plt.plot(std_dx, std_dy, color='r', label='spd_std')
    plt.plot(noise_dx, noise_dy, color='b', label='spd_noise')
    plt.plot(filt_dx, filt_dy, color='g', label='spd_filt')
    plt.legend()

    plt.subplot(324)
    t = np.arange(0, len(std_dx))
    plt.plot(t, [noise_dx[i] - std_dx[i] for i in range(len(std_dx))],
             color='r', label='spd_err_noise_dx')
    plt.plot(t, [noise_dy[i] - std_dy[i] for i in range(len(std_dy))],
             color='b', label='spd_err_noise_dy')
    plt.plot(t, [filt_dx[i] - std_dx[i] for i in range(len(std_dx))],
             color='g', label='spd_err_filt_dx')
    plt.plot(t, [filt_dy[i] - std_dy[i] for i in range(len(std_dy))],
             color='c', label='spd_err_filt_dy')
    plt.legend()

def plot_prob(cv, ca, ct):
    plt.figure(1)

    plt.subplot(313)
    t = np.arange(0, len(cv))
    plt.plot(t, cv, color='r', label='prob_cv')
    plt.plot(t, ca, color='b', label='prob_ca')
    plt.plot(t, ct, color='g', label='prob_ct')
    plt.legend()

def plot_show():
    plt.show()
