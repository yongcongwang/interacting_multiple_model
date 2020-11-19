#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# By yongcong.wang @ 13/10/2020

import math
import numpy as np

class Imm:
    def __init__(self, models, model_trans, P_trans, U_prob):
        self.models = models
        self.P_trans = P_trans
        self.U_prob = U_prob
        self.model_trans = model_trans

        self.mode_cnt = len(models)
        self.dim = models[0].A.shape[0]

    def filt(self, Z):
        # setp1: input mix
        u = np.dot(self.P_trans.T, self.U_prob)
        mu = np.zeros(self.P_trans.shape)
        for i in range(self.mode_cnt):
            for j in range(self.mode_cnt):
                mu[i, j] = self.P_trans[i, j] * self.U_prob[i, 0] / u[j, 0];

        X_mix = [np.zeros(model.X.shape) for model in self.models]

        for j in range(self.mode_cnt):
            for i in range(self.mode_cnt):
                X_mix[j] += np.dot(self.model_trans[j][i],
                                   self.models[i].X) * mu[i, j]

        P_mix = [np.zeros(model.P.shape) for model in self.models]
        for j in range(self.mode_cnt):
            for i in range(self.mode_cnt):
                P = self.models[i].P + np.dot((self.models[i].X - X_mix[i]),
                                              (self.models[i].X - X_mix[i]).T)
                P_mix[j] += mu[i, j] * np.dot(np.dot(self.model_trans[j][i], P),
                                              self.model_trans[j][i].T)
        ## step2: filt
        for j in range(self.mode_cnt):
            self.models[j].X = X_mix[j]
            self.models[j].P = P_mix[j]
            self.models[j].filt(Z)

        ### step3: update probability
        for j in range(self.mode_cnt):
            mode = self.models[j]
            D = Z - np.dot(mode.H, mode.X_pre)
            S = np.dot(np.dot(mode.H, mode.P_pre), mode.H.T) + mode.R

            Lambda = (np.linalg.det(2 * math.pi * S)) ** (-0.5) * \
                     np.exp(-0.5 * np.dot(np.dot(D.T, np.linalg.inv(S)), D))
            self.U_prob[j, 0] = Lambda * u[j, 0]
        self.U_prob = self.U_prob / np.sum(self.U_prob)

        return self.U_prob
