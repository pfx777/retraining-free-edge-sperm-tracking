#!/usr/bin/env python3
"""Kalman-filter DeepOCSORT tracks for the VISEM tracking pipeline."""

import numpy as np

import run_tracking_deepocsort as rtd


class SimpleKalmanXYWH:
    """Self-contained constant-velocity Kalman filter for (cx, cy, w, h)."""

    def __init__(self, weight_position=1 / 20, weight_velocity=1 / 160):
        self._std_weight_position = weight_position
        self._std_weight_velocity = weight_velocity
        self._motion_mat = np.eye(8, dtype=np.float32)
        for i in range(4):
            self._motion_mat[i, i + 4] = 1.0
        self._update_mat = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self._update_mat[i, i] = 1.0

    def initiate(self, measurement):
        mean_pos = np.asarray(measurement, dtype=np.float32)
        mean = np.r_[mean_pos, np.zeros_like(mean_pos)]
        std = np.array([
            2 * self._std_weight_position * mean_pos[2],
            2 * self._std_weight_position * mean_pos[3],
            2 * self._std_weight_position * mean_pos[2],
            2 * self._std_weight_position * mean_pos[3],
            10 * self._std_weight_velocity * mean_pos[2],
            10 * self._std_weight_velocity * mean_pos[3],
            10 * self._std_weight_velocity * mean_pos[2],
            10 * self._std_weight_velocity * mean_pos[3],
        ], dtype=np.float32)
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = np.array([
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ], dtype=np.float32)
        std_vel = np.array([
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ], dtype=np.float32)
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def update(self, mean, covariance, measurement):
        std = np.array([
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ], dtype=np.float32)
        innovation_cov = np.diag(np.square(std))
        projected_mean = self._update_mat @ mean
        projected_cov = self._update_mat @ covariance @ self._update_mat.T + innovation_cov
        gain = covariance @ self._update_mat.T @ np.linalg.inv(projected_cov)
        innovation = np.asarray(measurement, dtype=np.float32) - projected_mean
        mean = mean + gain @ innovation
        covariance = covariance - gain @ projected_cov @ gain.T
        return mean, covariance


class KalmanTrack:
    def __init__(self, tid, xywh, feat, conf):
        self.id = tid
        self.kf = SimpleKalmanXYWH()
        self.mean, self.covariance = self.kf.initiate(np.asarray(xywh, dtype=np.float32))
        self.features = [feat] if feat is not None else []
        self.conf = conf
        self.hits = 1
        self.age = 0

    @property
    def xywh(self):
        return self.mean[:4].copy()

    def predict(self):
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1

    def update(self, xywh, feat, conf):
        self.mean, self.covariance = self.kf.update(
            self.mean,
            self.covariance,
            np.asarray(xywh, dtype=np.float32),
        )
        self.conf = conf
        if feat is not None:
            self.features.append(feat)
            if len(self.features) > 30:
                self.features.pop(0)
        self.hits += 1
        self.age = 0

    def appearance(self):
        if not self.features:
            return None
        return np.mean(np.stack(self.features, axis=0), axis=0)

    def appearance_consistency(self):
        if len(self.features) < 2:
            return 1.0
        f = np.stack(self.features, axis=0)
        f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-6)
        sim = f @ f.T
        triu = np.triu_indices(len(self.features), 1)
        if triu[0].size == 0:
            return 1.0
        return float(sim[triu].mean())


class KalmanTracker(rtd.Tracker):
    def __init__(self):
        self.next_id = 1
        self.tracks = []

    def update(self, detections):
        for t in self.tracks:
            t.predict()

        high = [d for d in detections if d["conf"] >= rtd.TRACK_HIGH]
        low = [d for d in detections if rtd.TRACK_LOW <= d["conf"] < rtd.TRACK_HIGH]

        matched, unr, unc = self.match(self.tracks, high, use_app=True, thresh=rtd.MATCH_IOU)
        for r, c in matched:
            self.tracks[r].update(high[c]["xywh"], high[c]["feat"], high[c]["conf"])

        rem_tracks = [self.tracks[i] for i in unr]
        matched2, unr2, unc2 = self.match(rem_tracks, low, use_app=False, thresh=rtd.LOW_MATCH_IOU)
        for r, c in matched2:
            rem_tracks[r].update(low[c]["xywh"], low[c]["feat"], low[c]["conf"])

        for j in unc:
            if high[j]["conf"] >= rtd.NEW_TRACK:
                self.tracks.append(KalmanTrack(self.next_id, high[j]["xywh"], high[j]["feat"], high[j]["conf"]))
                self.next_id += 1

        live = []
        for t in self.tracks:
            if t.age > rtd.MAX_AGE:
                continue
            if t.hits >= rtd.MIN_HITS and t.age == 0:
                if rtd.APP_CONSIST > 0 and t.appearance_consistency() < rtd.APP_CONSIST:
                    continue
                live.append(t)
        self.tracks = [t for t in self.tracks if t.age <= rtd.MAX_AGE]
        return live

    def match(self, tracks, dets, use_app, thresh):
        return super().match(tracks, dets, use_app, thresh)
