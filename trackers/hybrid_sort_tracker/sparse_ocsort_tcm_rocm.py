# # -*- coding: utf-8 -*-
# """
# OC-SORT backbone with:
# - Soft-Depth Regularization
# - UA-HMIoU (Uncertainty-Aware Height Modulated IoU)
# - PC-HMIoU (Perspective-Consistent HMIoU)

# This version keeps the original 3-component pipeline as the backbone and
# lightly ports Hybrid-SORT cues in a conservative way:
# - TCM is added ONLY to round-2 (BYTE-like) and round-3 (re-association)
# - ROCM utilities / state are added, but round-1 is kept unchanged
# - Optional ROCM rerank is provided for round-3 only and is OFF by default

# Key design choice:
# - round-1 remains identical to the user's original tracker because that was
#   the strongest baseline.
# - acceptance thresholds still use raw geometric IoU / max(IoU(last), IoU(pred)).
# """

# from __future__ import print_function
# import numpy as np

# from .association import *  # iou_batch, giou_batch, ciou_batch, diou_batch, ct_dist, associate_kitti, linear_assignment


# # ---------- safe type conversions ----------
# def _to_numpy(x, dtype=float):
#     try:
#         import torch
#         if isinstance(x, torch.Tensor):
#             return x.detach().cpu().numpy().astype(dtype, copy=False)
#     except Exception:
#         pass
#     if isinstance(x, np.ndarray):
#         return x.astype(dtype, copy=False)
#     return np.array(x, dtype=dtype)


# def _to_scalar(x):
#     try:
#         import torch
#         if isinstance(x, torch.Tensor):
#             return float(x.detach().cpu().item())
#     except Exception:
#         pass
#     try:
#         return float(x)
#     except Exception:
#         return float(np.array(x).reshape(-1)[0])


# # ---------- helpers ----------
# def k_previous_obs(observations, cur_age, k):
#     if len(observations) == 0:
#         return [-1, -1, -1, -1, -1]
#     for i in range(k):
#         dt = k - i
#         if cur_age - dt in observations:
#             return observations[cur_age - dt]
#     max_age = max(observations.keys())
#     return observations[max_age]


# def convert_bbox_to_z(bbox):
#     # [x1,y1,x2,y2,(score)] -> z = [x,y,s,r]^T
#     w = bbox[2] - bbox[0]
#     h = bbox[3] - bbox[1]
#     x = bbox[0] + w / 2.0
#     y = bbox[1] + h / 2.0
#     s = w * h
#     r = w / float(h + 1e-6)
#     return np.array([x, y, s, r]).reshape((4, 1))


# def convert_x_to_bbox(x, score=None):
#     # [x,y,s,r,(vx,vy,vs)] -> [x1,y1,x2,y2,(score)]
#     w = np.sqrt(x[2] * x[3])
#     h = x[2] / (w + 1e-6)
#     if score is None:
#         return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
#     else:
#         return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


# def speed_direction(bbox1, bbox2):
#     cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
#     cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
#     speed = np.array([cy2 - cy1, cx2 - cx1])
#     norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
#     return speed / norm


# # ---------- Hybrid-SORT TCM score-difference helpers ----------
# def cal_score_dif_batch(bboxes1, bboxes2):
#     """Absolute difference between detection score and tracker score proxy.

#     bboxes1: (Nd, >=5), use column 4 as detection score
#     bboxes2: (Nt, >=5), use column 4 as tracker score proxy
#     returns: (Nd, Nt)
#     """
#     if len(bboxes1) == 0 or len(bboxes2) == 0:
#         return np.zeros((len(bboxes1), len(bboxes2)), dtype=float)
#     bboxes2 = np.expand_dims(np.asarray(bboxes2, dtype=float), 0)
#     bboxes1 = np.expand_dims(np.asarray(bboxes1, dtype=float), 1)
#     score2 = bboxes2[..., 4]
#     score1 = bboxes1[..., 4]
#     return np.abs(score2 - score1)


# def cal_score_dif_batch_two_score(det_boxes, trk_score_proxy):
#     """BYTE-step style score difference.

#     det_boxes: (Nd, >=5), column 4 is detection score
#     trk_score_proxy: (Nt,) or (Nt,1), tracker-side predicted/simple score proxy
#     returns: (Nd, Nt)
#     """
#     det_scores = np.asarray(det_boxes, dtype=float)[:, 4:5]  # (Nd,1)
#     trk_scores = np.asarray(trk_score_proxy, dtype=float).reshape(1, -1)  # (1,Nt)
#     return np.abs(det_scores - trk_scores)


# # ---------- Hybrid-SORT ROCM helpers (optional / conservative) ----------
# def speed_direction_batch_lt(dets, tracks):
#     tracks = tracks[..., np.newaxis]
#     CX1, CY1 = dets[:, 0], dets[:, 1]
#     CX2, CY2 = tracks[:, 0], tracks[:, 1]
#     dx = CX1 - CX2
#     dy = CY1 - CY2
#     norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
#     dx = dx / norm
#     dy = dy / norm
#     return dy, dx


# def speed_direction_batch_rt(dets, tracks):
#     tracks = tracks[..., np.newaxis]
#     CX1, CY1 = dets[:, 0], dets[:, 3]
#     CX2, CY2 = tracks[:, 0], tracks[:, 3]
#     dx = CX1 - CX2
#     dy = CY1 - CY2
#     norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
#     dx = dx / norm
#     dy = dy / norm
#     return dy, dx


# def speed_direction_batch_lb(dets, tracks):
#     tracks = tracks[..., np.newaxis]
#     CX1, CY1 = dets[:, 2], dets[:, 1]
#     CX2, CY2 = tracks[:, 2], tracks[:, 1]
#     dx = CX1 - CX2
#     dy = CY1 - CY2
#     norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
#     dx = dx / norm
#     dy = dy / norm
#     return dy, dx


# def speed_direction_batch_rb(dets, tracks):
#     tracks = tracks[..., np.newaxis]
#     CX1, CY1 = dets[:, 2], dets[:, 3]
#     CX2, CY2 = tracks[:, 2], tracks[:, 3]
#     dx = CX1 - CX2
#     dy = CY1 - CY2
#     norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
#     dx = dx / norm
#     dy = dy / norm
#     return dy, dx


# def cost_vel(Y, X, trackers, velocities, detections, previous_obs, vdc_weight):
#     inertia_Y, inertia_X = velocities[:, 0], velocities[:, 1]
#     inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
#     inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
#     diff_angle_cos = inertia_X * X + inertia_Y * Y
#     diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
#     diff_angle = np.arccos(diff_angle_cos)
#     diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

#     valid_mask = np.ones(previous_obs.shape[0])
#     valid_mask[np.where(previous_obs[:, 4] < 0)] = 0
#     scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
#     valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

#     angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
#     angle_diff_cost = angle_diff_cost.T
#     angle_diff_cost = angle_diff_cost * scores
#     return angle_diff_cost


# # ---------- Kalman tracker with UA-height stats + lightweight TCM/ROCM state ----------
# class KalmanBoxTracker(object):
#     """
#     OC-SORT Kalman tracker (7D state) +
#     depth_ema (EMA of bottom y2) for pseudo-depth prior +
#     UA-HMIoU height stats (EMA of h, h^2) with variance inflation +
#     lightweight score proxy for TCM and 4-point velocities for optional ROCM.
#     """
#     count = 0

#     def __init__(
#         self,
#         bbox,
#         delta_t=3,
#         orig=False,
#         ema_obs=0.5,
#         ema_pred=0.1,
#         # UA-HMIoU params for height stats
#         ua_h_ema=0.5,
#         ua_h2_ema=0.3,
#         ua_proc_var_frac=0.02,
#         ua_sigma_floor_frac=0.05,
#         # TCM score proxy
#         score_ema=0.7,
#     ):
#         if not orig:
#             from .kalmanfilter import KalmanFilterNew as KalmanFilter
#             self.kf = KalmanFilter(dim_x=7, dim_z=4)
#         else:
#             from filterpy.kalman import KalmanFilter
#             self.kf = KalmanFilter(dim_x=7, dim_z=4)

#         self.kf.F = np.array([
#             [1, 0, 0, 0, 1, 0, 0],
#             [0, 1, 0, 0, 0, 1, 0],
#             [0, 0, 1, 0, 0, 0, 1],
#             [0, 0, 0, 1, 0, 0, 0],
#             [0, 0, 0, 0, 1, 0, 0],
#             [0, 0, 0, 0, 0, 1, 0],
#             [0, 0, 0, 0, 0, 0, 1]
#         ])
#         self.kf.H = np.array([
#             [1, 0, 0, 0, 0, 0, 0],
#             [0, 1, 0, 0, 0, 0, 0],
#             [0, 0, 1, 0, 0, 0, 0],
#             [0, 0, 0, 1, 0, 0, 0]
#         ])

#         self.kf.R[2:, 2:] *= 10.0
#         self.kf.P[4:, 4:] *= 1000.0
#         self.kf.P *= 10.0
#         self.kf.Q[-1, -1] *= 0.01
#         self.kf.Q[4:, 4:] *= 0.01

#         self.kf.x[:4] = convert_bbox_to_z(bbox)

#         self.time_since_update = 0
#         self.id = KalmanBoxTracker.count
#         KalmanBoxTracker.count += 1
#         self.history = []
#         self.hits = 0
#         self.hit_streak = 0
#         self.age = 0

#         self.last_observation = np.array([-1, -1, -1, -1, -1])
#         self.observations = dict()
#         self.history_observations = []
#         self.velocity = None
#         self.delta_t = delta_t

#         # Optional 4-point velocities for future ROCM
#         self.velocity_lt = None
#         self.velocity_rt = None
#         self.velocity_lb = None
#         self.velocity_rb = None

#         # Depth EMA (pseudo-depth y2)
#         self.ema_obs = float(ema_obs)
#         self.ema_pred = float(ema_pred)
#         self.depth_ema = float(bbox[3])

#         # UA-HMIoU height stats
#         h0 = float(bbox[3] - bbox[1])
#         self.ua_h_ema = float(ua_h_ema)
#         self.ua_h2_ema = float(ua_h2_ema)
#         self.ua_proc_var_frac = float(ua_proc_var_frac)
#         self.ua_sigma_floor_frac = float(ua_sigma_floor_frac)

#         self.h_m1 = h0
#         self.h_m2 = h0 * h0
#         floor_var = (self.ua_sigma_floor_frac * h0) ** 2
#         self.h_var = max(self.h_m2 - self.h_m1 * self.h_m1, floor_var)

#         # TCM score proxy
#         init_score = float(bbox[4]) if len(bbox) >= 5 else 1.0
#         self.score_ema_decay = float(score_ema)
#         self.confidence = init_score
#         self.confidence_pre = None
#         self.simple_score = init_score

#     def get_height_mu_sigma(self):
#         mu = float(self.h_m1)
#         floor_var = (self.ua_sigma_floor_frac * max(mu, 1e-6)) ** 2
#         var = max(float(self.h_var), floor_var)
#         sigma = np.sqrt(var)
#         return mu, sigma

#     def _inflation_step(self):
#         mu = max(float(self.h_m1), 1e-6)
#         self.h_var = float(self.h_var) + (self.ua_proc_var_frac * mu) ** 2

#     def _update_corner_velocities(self, previous_box, bbox):
#         def _corner_dir(b1, b2, x_idx, y_idx):
#             p1 = np.array([b1[x_idx], b1[y_idx]], dtype=float)
#             p2 = np.array([b2[x_idx], b2[y_idx]], dtype=float)
#             v = np.array([p2[1] - p1[1], p2[0] - p1[0]], dtype=float)
#             n = np.sqrt(v[0] ** 2 + v[1] ** 2) + 1e-6
#             return v / n

#         self.velocity_lt = _corner_dir(previous_box, bbox, 0, 1)
#         self.velocity_rt = _corner_dir(previous_box, bbox, 0, 3)
#         self.velocity_lb = _corner_dir(previous_box, bbox, 2, 1)
#         self.velocity_rb = _corner_dir(previous_box, bbox, 2, 3)

#     def get_tcm_score_proxy(self, low=0.1, high=1.0):
#         if self.confidence_pre is None:
#             s = self.confidence
#         else:
#             s = self.confidence - (self.confidence_pre - self.confidence)
#         return float(np.clip(s, low, high))

#     def update(self, bbox):
#         if bbox is not None:
#             bbox = np.asarray(bbox, dtype=float)
#             if self.last_observation.sum() >= 0:
#                 previous_box = None
#                 for i in range(self.delta_t):
#                     dt = self.delta_t - i
#                     if self.age - dt in self.observations:
#                         previous_box = self.observations[self.age - dt]
#                         break
#                 if previous_box is None:
#                     previous_box = self.last_observation
#                 self.velocity = speed_direction(previous_box, bbox)
#                 self._update_corner_velocities(previous_box, bbox)

#             # Depth EMA with observation
#             y2 = float(bbox[3])
#             self.depth_ema = (1.0 - self.ema_obs) * self.depth_ema + self.ema_obs * y2

#             # Height stats update
#             h = float(bbox[3] - bbox[1])
#             self.h_m1 = (1.0 - self.ua_h_ema) * self.h_m1 + self.ua_h_ema * h
#             self.h_m2 = (1.0 - self.ua_h2_ema) * self.h_m2 + self.ua_h2_ema * (h * h)
#             raw_var = max(self.h_m2 - self.h_m1 * self.h_m1, 0.0)
#             floor_var = (self.ua_sigma_floor_frac * max(self.h_m1, 1e-6)) ** 2
#             self.h_var = 0.5 * self.h_var + 0.5 * max(raw_var, floor_var)

#             # TCM score proxy update
#             score = float(bbox[4]) if len(bbox) >= 5 else self.confidence
#             self.confidence_pre = self.confidence
#             self.confidence = (1.0 - self.score_ema_decay) * self.confidence + self.score_ema_decay * score
#             self.simple_score = self.get_tcm_score_proxy(low=0.1, high=1.0)

#             # bookkeeping
#             self.last_observation = bbox
#             self.observations[self.age] = bbox
#             self.history_observations.append(bbox)

#             self.time_since_update = 0
#             self.history = []
#             self.hits += 1
#             self.hit_streak += 1

#             self.kf.update(convert_bbox_to_z(bbox))
#         else:
#             self.kf.update(bbox)

#     def predict(self):
#         if (self.kf.x[6] + self.kf.x[2]) <= 0:
#             self.kf.x[6] *= 0.0
#         self.kf.predict()
#         self.age += 1
#         if self.time_since_update > 0:
#             self.hit_streak = 0
#         self.time_since_update += 1
#         pred = convert_x_to_bbox(self.kf.x)
#         self.history.append(pred)

#         # gently pull depth_ema towards predicted y2
#         pred_y2 = float(pred[0][3])
#         self.depth_ema = (1.0 - self.ema_pred) * self.depth_ema + self.ema_pred * pred_y2

#         # process noise inflation for UA-height; only once per frame in predict()
#         self._inflation_step()

#         # score proxy does not advance aggressively; keep simple extrapolation stable
#         self.simple_score = self.get_tcm_score_proxy(low=0.1, high=1.0)
#         return self.history[-1]

#     def get_state(self):
#         return convert_x_to_bbox(self.kf.x)


# ASSO_FUNCS = {
#     "iou": iou_batch,
#     "giou": giou_batch,
#     "ciou": ciou_batch,
#     "diou": diou_batch,
#     "ct_dist": ct_dist
# }


# # ---------- Perspective height model (PC-HMIoU core) ----------
# class PerspectiveHeightModel:
#     """
#     Online mapping h*(y2): bottom y2 -> expected pixel height
#     Implemented via normalized y2 binning + EMA; supports interpolation and fallback.
#     """
#     def __init__(self, num_bins=24, ema=0.08, min_count=2, floor=0.3, gamma=2.0):
#         self.num_bins = int(num_bins)
#         self.ema = float(ema)
#         self.min_count = int(min_count)
#         self.floor = float(floor)
#         self.gamma = float(gamma)
#         self.reset()

#     def reset(self):
#         self.bin_means = np.zeros(self.num_bins, dtype=float)
#         self.bin_counts = np.zeros(self.num_bins, dtype=float)
#         self.global_ema = 0.0
#         self.global_count = 0.0

#     def _bin_index(self, y2_norm):
#         y = np.clip(y2_norm, 0.0, 1.0 - 1e-8)
#         return np.floor(y * self.num_bins).astype(int)

#     def observe(self, y2_pixel, h_pixel, img_h):
#         if img_h <= 0:
#             return
#         y2_norm = float(y2_pixel) / float(img_h)
#         if not np.isfinite(y2_norm) or not np.isfinite(h_pixel) or h_pixel <= 0:
#             return
#         b = int(self._bin_index(y2_norm))
#         m = self.bin_means[b]
#         c = self.bin_counts[b]
#         m_new = h_pixel if c < 1e-6 else (1.0 - self.ema) * m + self.ema * h_pixel
#         self.bin_means[b] = m_new
#         self.bin_counts[b] = min(c + 1.0, 1e9)
#         self.global_ema = h_pixel if self.global_count < 1e-6 else 0.98 * self.global_ema + 0.02 * h_pixel
#         self.global_count = min(self.global_count + 1.0, 1e9)

#     def expected_vec(self, y2_pixels, img_h):
#         if img_h <= 0 or len(y2_pixels) == 0:
#             return np.zeros_like(y2_pixels, dtype=float)
#         y2n = np.clip(np.asarray(y2_pixels, dtype=float) / float(img_h), 0.0, 1.0 - 1e-8)
#         bins = np.floor(y2n * self.num_bins).astype(int)
#         pos = y2n * self.num_bins
#         frac = pos - np.floor(pos)
#         b0 = bins
#         b1 = np.clip(bins + 1, 0, self.num_bins - 1)
#         m0 = self.bin_means[b0]
#         m1 = self.bin_means[b1]
#         c0 = self.bin_counts[b0]
#         c1 = self.bin_counts[b1]
#         use0 = (c0 >= self.min_count)
#         use1 = (c1 >= self.min_count)
#         m0_eff = np.where(use0, m0, np.where(use1, m1, self.global_ema))
#         m1_eff = np.where(use1, m1, np.where(use0, m0, self.global_ema))
#         h_star = (1.0 - frac) * m0_eff + frac * m1_eff
#         h_star = np.maximum(h_star, 4.0)
#         return h_star

#     def affinity_matrix(self, dets_xyxy, trks_xyxy, img_h):
#         Nd = dets_xyxy.shape[0]
#         Nt = trks_xyxy.shape[0]
#         if Nd == 0 or Nt == 0:
#             return np.zeros((Nd, Nt), dtype=float)
#         h_det = (dets_xyxy[:, 3] - dets_xyxy[:, 1]).astype(float)[:, None]
#         y2_trk = trks_xyxy[:, 3].astype(float)
#         h_star = self.expected_vec(y2_trk, img_h)[None, :]
#         eps = 1e-6
#         r = np.abs(h_det - h_star) / (h_star + eps)
#         A_p = np.exp(-self.gamma * r)
#         if self.floor is not None and self.floor > 0.0:
#             A_p = np.maximum(A_p, self.floor)
#         return A_p

#     def ready(self):
#         return self.global_count >= 50


# # ---------- main tracker with original 3 components + conservative TCM/ROCM ports ----------
# class Soperm_Track_Hybird(object):
#     def __init__(
#         self,
#         det_thresh,
#         max_age=30,
#         min_hits=3,
#         iou_threshold=0.3,
#         delta_t=3,
#         asso_func="iou",
#         inertia=0.2,
#         use_byte=True,
#         # Soft-depth params
#         depth_alpha=0.5,
#         depth_beta=2.0,
#         depth_gate=0.25,
#         gate_floor=0.2,
#         depth_enable=True,
#         # UA-HMIoU params
#         ua_enable=True,
#         ua_mode="cauchy",
#         ua_alpha=1.0,
#         ua_border_margin_frac=0.02,
#         ua_border_boost=0.6,
#         ua_miss_boost=0.6,
#         ua_low_score=0.4,
#         ua_score_boost=0.3,
#         ua_sigma_floor_frac=0.05,
#         ua_proc_var_frac=0.02,
#         ua_h_ema=0.5,
#         ua_h2_ema=0.3,
#         # PC-HMIoU params
#         pc_enable=True,
#         pc_bins=24,
#         pc_ema=0.08,
#         pc_gamma=2.0,
#         pc_floor=0.3,
#         pc_min_count=2,
#         pc_sample_score=0.6,
#         # Depth EMA
#         ema_obs=0.5,
#         ema_pred=0.1,
#         # TCM first-step (very conservative)
#         tcm_round1_enable=False,
#         tcm_round1_weight=0.03,
#         tcm_round1_margin=0.05,
#         # TCM (ported conservatively: rounds 2 & 3 only)
#         tcm_round2_enable=True,
#         tcm_round2_weight=0.08,
#         tcm_round3_enable=True,
#         tcm_round3_weight=0.10,
#         # ROCM hooks (disabled by default for scheme-1)
#         rocm_round3_enable=False,
#         rocm_weight=0.08,
#         # score proxy for TCM
#         score_ema=0.7,
#     ):
#         self.max_age = max_age
#         self.min_hits = min_hits
#         self.iou_threshold = iou_threshold
#         self.trackers = []
#         self.frame_count = 0
#         self.det_thresh = det_thresh
#         self.delta_t = delta_t
#         self.asso_func = ASSO_FUNCS[asso_func]
#         self.inertia = inertia
#         self.use_byte = use_byte

#         # Soft-depth
#         self.depth_alpha = float(depth_alpha)
#         self.depth_beta = float(depth_beta)
#         self.depth_gate = float(depth_gate)
#         self.gate_floor = float(gate_floor)
#         self.depth_enable = bool(depth_enable)

#         # UA-HMIoU
#         self.ua_enable = bool(ua_enable)
#         self.ua_mode = str(ua_mode).lower()
#         assert self.ua_mode in ["cauchy", "gauss"]
#         self.ua_alpha = float(ua_alpha)
#         self.ua_border_margin_frac = float(ua_border_margin_frac)
#         self.ua_border_boost = float(ua_border_boost)
#         self.ua_miss_boost = float(ua_miss_boost)
#         self.ua_low_score = float(ua_low_score)
#         self.ua_score_boost = float(ua_score_boost)
#         self.ua_sigma_floor_frac = float(ua_sigma_floor_frac)
#         self.ua_proc_var_frac = float(ua_proc_var_frac)
#         self.ua_h_ema = float(ua_h_ema)
#         self.ua_h2_ema = float(ua_h2_ema)

#         # PC-HMIoU
#         self.pc_enable = bool(pc_enable)
#         self.pc_sample_score = float(pc_sample_score)
#         self.perspective = PerspectiveHeightModel(
#             num_bins=pc_bins, ema=pc_ema, min_count=pc_min_count, floor=pc_floor, gamma=pc_gamma
#         )

#         # Depth EMA
#         self.ema_obs = float(ema_obs)
#         self.ema_pred = float(ema_pred)

#         # TCM/ROCM
        
#         self.tcm_round1_enable = bool(tcm_round1_enable)
#         self.tcm_round1_weight = float(tcm_round1_weight)
#         self.tcm_round1_margin = float(tcm_round1_margin)


        
#         self.tcm_round2_enable = bool(tcm_round2_enable)
#         self.tcm_round2_weight = float(tcm_round2_weight)
#         self.tcm_round3_enable = bool(tcm_round3_enable)
#         self.tcm_round3_weight = float(tcm_round3_weight)
#         self.rocm_round3_enable = bool(rocm_round3_enable)
        
        
#         self.rocm_weight = float(rocm_weight)
#         self.score_ema = float(score_ema)

#         KalmanBoxTracker.count = 0

#         self._prev_frame_id = -1
#         self.last_img_size = None

#     def reset(self, reset_perspective=True, reset_id_counter=False):
#         self.trackers = []
#         self.frame_count = 0
#         self._prev_frame_id = -1
#         self.last_img_size = None
#         if reset_perspective and getattr(self, "pc_enable", False) and hasattr(self, "perspective"):
#             self.perspective.reset()
#         if reset_id_counter:
#             KalmanBoxTracker.count = 0

#     def _maybe_reset_on_boundary(self, frame_id=None, img_size=None,
#                                  reset_on_size_change=False,
#                                  verbose=False):
#         fid_back = (self._prev_frame_id != -1 and
#                     frame_id is not None and int(frame_id) <= int(self._prev_frame_id))
#         is_new_seq = False

#         if frame_id is not None:
#             fid = int(frame_id)
#             if fid == 1 or fid_back:
#                 is_new_seq = True

#         size_changed = False
#         if reset_on_size_change and img_size is not None and self.last_img_size is not None:
#             size_changed = tuple(img_size) != tuple(self.last_img_size)
#             if size_changed:
#                 is_new_seq = True

#         if is_new_seq:
#             if getattr(self, "pc_enable", False) and hasattr(self, "perspective"):
#                 self.perspective.reset()
#                 if verbose:
#                     print(f"[OCSort] perspective reset at boundary (frame_id={frame_id}, "
#                           f"fid_back={fid_back}, size_changed={size_changed})")

#         if frame_id is not None:
#             self._prev_frame_id = int(frame_id)
#         if img_size is not None:
#             self.last_img_size = tuple(img_size)

#     # ---------- Soft-depth helpers ----------
#     @staticmethod
#     def _y2_from_boxes(boxes):
#         return boxes[:, 3].astype(float) if boxes.size > 0 else np.array([])

#     def _depth_affinity(self, dets_xyxy, trks_xyxy, trk_depth_ema, img_h):
#         Nd = dets_xyxy.shape[0]
#         Nt = trks_xyxy.shape[0]
#         if Nd == 0 or Nt == 0:
#             return np.zeros((Nd, Nt), dtype=float)
#         y2d = dets_xyxy[:, 3].astype(float)[:, None]
#         y2t = trk_depth_ema.astype(float)[None, :]
#         norm = max(float(img_h), 1e-6)
#         d = np.abs(y2d - y2t) / norm
#         A = np.exp(-self.depth_beta * d)
#         if self.depth_gate > 0:
#             mask_far = d > self.depth_gate
#             if self.gate_floor >= 0.0:
#                 A[mask_far] = np.maximum(A[mask_far], self.gate_floor)
#         return A

#     # ---------- UA-HMIoU helpers ----------
#     def _ua_height_affinity(
#         self,
#         dets_xyxy, trks_xyxy,
#         trk_mu_h, trk_sigma_h,
#         det_scores, trk_miss,
#         img_w, img_h
#     ):
#         dets_xyxy = _to_numpy(dets_xyxy, dtype=float)
#         trks_xyxy = _to_numpy(trks_xyxy, dtype=float)
#         trk_mu_h = _to_numpy(trk_mu_h, dtype=float).reshape(-1)
#         trk_sigma_h = _to_numpy(trk_sigma_h, dtype=float).reshape(-1)
#         det_scores = _to_numpy(det_scores, dtype=float).reshape(-1)
#         trk_miss = _to_numpy(trk_miss, dtype=float).reshape(-1)
#         img_w = _to_scalar(img_w)
#         img_h = _to_scalar(img_h)

#         Nd = dets_xyxy.shape[0]
#         Nt = trks_xyxy.shape[0]
#         if Nd == 0 or Nt == 0:
#             return np.zeros((Nd, Nt), dtype=float)

#         eps = 1e-6
#         hd = (dets_xyxy[:, 3] - dets_xyxy[:, 1]).astype(float)[:, None]
#         mu = trk_mu_h.astype(float)[None, :]
#         sigma_base = trk_sigma_h.astype(float)[None, :]

#         miss = np.clip(trk_miss.astype(float), 0.0, 2.0)[None, :]
#         S_trk = 1.0 + self.ua_miss_boost * (miss / 2.0)

#         if Nd > 0:
#             m = max(1.0, min(float(img_w), float(img_h)) * self.ua_border_margin_frac)
#             x1 = dets_xyxy[:, 0].astype(float)
#             y1 = dets_xyxy[:, 1].astype(float)
#             x2 = dets_xyxy[:, 2].astype(float)
#             y2 = dets_xyxy[:, 3].astype(float)
#             min_dist = np.minimum(np.minimum(x1, y1), np.minimum(img_w - x2, img_h - y2))
#             closeness = np.clip(1.0 - (min_dist / m), 0.0, 1.0)
#             sc = det_scores.astype(float)
#             low_score_gap = np.clip((self.ua_low_score - sc) / max(self.ua_low_score, 1e-6), 0.0, 1.0)
#             S_det = 1.0 + self.ua_border_boost * closeness + self.ua_score_boost * low_score_gap
#             S_det = S_det[:, None]
#         else:
#             S_det = 1.0

#         sigma_eff = sigma_base * S_trk
#         sigma_eff = sigma_eff * S_det
#         sigma_eff = np.maximum(sigma_eff, eps)

#         z = np.abs(hd - mu) / (self.ua_alpha * sigma_eff + eps)
#         if self.ua_mode == "gauss":
#             A_u = np.exp(-0.5 * z * z)
#         else:
#             A_u = 1.0 / (1.0 + z * z)
#         return A_u

#     # ---------- PC-HMIoU helpers ----------
#     def _pc_affinity(self, dets_xyxy, trks_xyxy, img_h):
#         if not self.pc_enable or not self.perspective.ready():
#             Nd = dets_xyxy.shape[0]
#             Nt = trks_xyxy.shape[0]
#             if Nd == 0 or Nt == 0:
#                 return np.zeros((Nd, Nt), dtype=float)
#             return np.ones((Nd, Nt), dtype=float)
#         return self.perspective.affinity_matrix(dets_xyxy, trks_xyxy, img_h)

#     # ---------- Soft-depth blending ----------
#     def _blend_iou_with_depth(self, iou, affinity):
#         if (not self.depth_enable) or self.depth_alpha <= 0:
#             return iou
#         return iou * ((1.0 - self.depth_alpha) + self.depth_alpha * affinity)

#     # ---------- perspective model update from matched samples ----------
#     def _update_pc_model_with_det(self, det_row, img_h):
#         score = float(det_row[4])
#         if score < self.pc_sample_score:
#             return
#         y2 = float(det_row[3])
#         h = float(det_row[3] - det_row[1])
#         if h > 0 and np.isfinite(h) and np.isfinite(y2):
#             self.perspective.observe(y2, h, img_h)

#     # ---------- Optional ROCM rerank for round-3 only ----------
#     def _rocm_round3_cost(self, dets_xyxy, prev_obs, velocities_lt, velocities_rt, velocities_lb, velocities_rb):
#         if dets_xyxy.shape[0] == 0 or prev_obs.shape[0] == 0:
#             return np.zeros((dets_xyxy.shape[0], prev_obs.shape[0]), dtype=float)
#         Y1, X1 = speed_direction_batch_lt(dets_xyxy, prev_obs)
#         Y2, X2 = speed_direction_batch_rt(dets_xyxy, prev_obs)
#         Y3, X3 = speed_direction_batch_lb(dets_xyxy, prev_obs)
#         Y4, X4 = speed_direction_batch_rb(dets_xyxy, prev_obs)
#         cost_lt = cost_vel(Y1, X1, prev_obs[:, :4], velocities_lt, np.concatenate([dets_xyxy, np.ones((dets_xyxy.shape[0], 1))], axis=1), prev_obs, self.inertia)
#         cost_rt = cost_vel(Y2, X2, prev_obs[:, :4], velocities_rt, np.concatenate([dets_xyxy, np.ones((dets_xyxy.shape[0], 1))], axis=1), prev_obs, self.inertia)
#         cost_lb = cost_vel(Y3, X3, prev_obs[:, :4], velocities_lb, np.concatenate([dets_xyxy, np.ones((dets_xyxy.shape[0], 1))], axis=1), prev_obs, self.inertia)
#         cost_rb = cost_vel(Y4, X4, prev_obs[:, :4], velocities_rb, np.concatenate([dets_xyxy, np.ones((dets_xyxy.shape[0], 1))], axis=1), prev_obs, self.inertia)
#         return cost_lt + cost_rt + cost_lb + cost_rb

#     # ---------- First round associate: IoU * A_u * A_p (UNCHANGED backbone) ----------
#     # def _associate_round1(
#     #         self,
#     #         dets, trks,
#     #         height_mus, height_sigmas, miss_counts,
#     #         img_w, img_h,
#     #         iou_thr,
#     #         depth_emas_full=None
#     #     ):
    
#     def _associate_round1(
#         self,
#         dets, trks,
#         height_mus, height_sigmas, miss_counts,
#         img_w, img_h,
#         iou_thr,
#         depth_emas_full=None,
#         trk_score_proxy_full=None
#         ):
    
#             Nd = dets.shape[0]
#             Nt = trks.shape[0]
#             if Nd == 0 or Nt == 0:
#                 return np.empty((0, 2), dtype=int), np.arange(Nd), np.arange(Nt)

#             dets_xy = dets[:, :4]
#             trks_xy = trks[:, :4]
#             det_sc = dets[:, 4]

#             iou = self.asso_func(dets_xy, trks_xy)
#             iou = np.array(iou)

#             if self.ua_enable:
#                 A_u = self._ua_height_affinity(
#                     dets_xy, trks_xy,
#                     height_mus, height_sigmas,
#                     det_sc, miss_counts,
#                     img_w, img_h
#                 )
#             else:
#                 A_u = np.ones_like(iou)

#             A_p = self._pc_affinity(dets_xy, trks_xy, img_h)

#             # S = iou * A_u * A_p
#             # if (depth_emas_full is not None) and self.depth_enable and (self.depth_alpha > 0):
#             #     A_d_r1 = self._depth_affinity(dets_xy, trks_xy, np.asarray(depth_emas_full), img_h)
#             #     S = self._blend_iou_with_depth(S, A_d_r1)

#             # matched = []
#             # if S.size > 0 and iou.max() > iou_thr:
#             #     idxs = linear_assignment(-S)
#             #     for d_i, t_i in idxs:
#             #         if iou[d_i, t_i] >= iou_thr:
#             #             matched.append([d_i, t_i])
            
#             S = iou * A_u * A_p
#             if (depth_emas_full is not None) and self.depth_enable and (self.depth_alpha > 0):
#                 A_d_r1 = self._depth_affinity(dets_xy, trks_xy, np.asarray(depth_emas_full), img_h)
#                 S = self._blend_iou_with_depth(S, A_d_r1)

#             # very conservative TCM-on-round1:
#             # only re-rank ambiguous pairs, never change the raw IoU acceptance gate
#             score_r1 = S.copy()
#             if (
#                 self.tcm_round1_enable
#                 and score_r1.size > 0
#                 and trk_score_proxy_full is not None
#                 and len(trk_score_proxy_full) == trks.shape[0]
#             ):
#                 tcm_penalty = cal_score_dif_batch_two_score(dets, trk_score_proxy_full)  # (Nd, Nt)

#                 # 只在“歧义 pair”上生效：
#                 # 1) IoU 只比阈值高一点，说明排序可能有歧义
#                 # 2) track 有 miss，说明状态不够新鲜
#                 ambiguous = (iou < (iou_thr + self.tcm_round1_margin))
#                 ambiguous |= (np.asarray(miss_counts, dtype=float)[None, :] > 0)

#                 score_r1[ambiguous] = score_r1[ambiguous] - self.tcm_round1_weight * tcm_penalty[ambiguous]

#             matched = []
#             if score_r1.size > 0 and iou.max() > iou_thr:
#                 idxs = linear_assignment(-score_r1)
#                 for d_i, t_i in idxs:
#                     # keep original raw-IoU gate
#                     if iou[d_i, t_i] >= iou_thr:
#                         matched.append([d_i, t_i])
            
                        
                        
#             matched = np.asarray(matched, dtype=int) if len(matched) > 0 else np.empty((0, 2), dtype=int)

#             matched_d = matched[:, 0] if matched.size > 0 else np.array([], dtype=int)
#             matched_t = matched[:, 1] if matched.size > 0 else np.array([], dtype=int)
#             unmatched_dets = np.setdiff1d(np.arange(Nd), matched_d, assume_unique=False)
#             unmatched_trks = np.setdiff1d(np.arange(Nt), matched_t, assume_unique=False)

#             return matched, unmatched_dets, unmatched_trks

#     def update(self, output_results, img_info, img_size, frame_id=None):
#         self._maybe_reset_on_boundary(frame_id=frame_id,
#                                       img_size=(img_info[0], img_info[1]),
#                                       reset_on_size_change=True,
#                                       verbose=True)

#         if output_results is None:
#             return np.empty((0, 5))

#         self.frame_count += 1

#         try:
#             import torch
#             if isinstance(output_results, torch.Tensor):
#                 output_results = output_results.detach().cpu().numpy()
#             elif isinstance(output_results, (list, tuple)) and len(output_results) > 0:
#                 first = output_results[0]
#                 if isinstance(first, torch.Tensor):
#                     output_results = first.detach().cpu().numpy()
#                 else:
#                     output_results = np.array(first) if not isinstance(first, np.ndarray) else first
#         except Exception:
#             pass
#         if not isinstance(output_results, np.ndarray):
#             output_results = np.array(output_results)

#         img_h, img_w = _to_scalar(img_info[0]), _to_scalar(img_info[1])

#         if output_results.size == 0:
#             trks = np.zeros((len(self.trackers), 5))
#             to_del = []
#             ret = []
#             for t, trk in enumerate(trks):
#                 pos = self.trackers[t].predict()[0]
#                 trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
#                 if np.any(np.isnan(pos)):
#                     to_del.append(t)
#             trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#             for t in reversed(to_del):
#                 self.trackers.pop(t)

#             i = len(self.trackers)
#             for trk in reversed(self.trackers):
#                 if trk.last_observation.sum() < 0:
#                     d = trk.get_state()[0]
#                 else:
#                     d = trk.last_observation[:4]
#                 if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
#                     ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
#                 i -= 1
#                 if trk.time_since_update > self.max_age:
#                     self.trackers.pop(i)
#             if len(ret) > 0:
#                 return np.concatenate(ret)
#             return np.empty((0, 5))

#         ncol = output_results.shape[1]
#         if ncol >= 6:
#             scores = output_results[:, 4].astype(float) * output_results[:, 5].astype(float)
#             bboxes = output_results[:, :4].astype(float)
#         elif ncol == 5:
#             scores = output_results[:, 4].astype(float)
#             bboxes = output_results[:, :4].astype(float)
#         else:
#             raise ValueError(f"Unexpected detection shape {output_results.shape}, expected 5 or >=6 columns.")

#         scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
#         bboxes = bboxes / max(scale, 1e-6)
#         dets_all = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)

#         inds_low = scores > 0.1
#         inds_high = scores < self.det_thresh
#         inds_second = np.logical_and(inds_low, inds_high)
#         dets_second = dets_all[inds_second]
#         remain_inds = scores > self.det_thresh
#         dets = dets_all[remain_inds]

#         trks = np.zeros((len(self.trackers), 5))
#         to_del = []
#         ret = []
#         for t, trk in enumerate(trks):
#             pos = self.trackers[t].predict()[0]
#             trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
#             if np.any(np.isnan(pos)):
#                 to_del.append(t)
#         trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#         for t in reversed(to_del):
#             self.trackers.pop(t)

#         last_boxes = np.array([trk.last_observation for trk in self.trackers])
#         depth_emas = np.array([trk.depth_ema for trk in self.trackers], dtype=float)
#         height_mus_full = np.array([trk.get_height_mu_sigma()[0] for trk in self.trackers], dtype=float)
#         height_sigmas_full = np.array([trk.get_height_mu_sigma()[1] for trk in self.trackers], dtype=float)
#         miss_counts_full = np.array([trk.time_since_update for trk in self.trackers], dtype=float)
#         track_score_proxy_full = np.array([trk.get_tcm_score_proxy(low=0.1, high=1.0) for trk in self.trackers], dtype=float)

#         # Optional ROCM state for future round-3 rerank
#         velocities_lt_full = np.array([trk.velocity_lt if trk.velocity_lt is not None else np.array((0, 0)) for trk in self.trackers]) if len(self.trackers) > 0 else np.zeros((0, 2))
#         velocities_rt_full = np.array([trk.velocity_rt if trk.velocity_rt is not None else np.array((0, 0)) for trk in self.trackers]) if len(self.trackers) > 0 else np.zeros((0, 2))
#         velocities_lb_full = np.array([trk.velocity_lb if trk.velocity_lb is not None else np.array((0, 0)) for trk in self.trackers]) if len(self.trackers) > 0 else np.zeros((0, 2))
#         velocities_rb_full = np.array([trk.velocity_rb if trk.velocity_rb is not None else np.array((0, 0)) for trk in self.trackers]) if len(self.trackers) > 0 else np.zeros((0, 2))

#         # ---------- First round: KEEP ORIGINAL BACKBONE UNCHANGED ----------
#         if dets.shape[0] > 0 and trks.shape[0] > 0:
#             # matched, unmatched_dets, unmatched_trks = self._associate_round1(
#             #     dets, trks,
#             #     height_mus_full, height_sigmas_full, miss_counts_full,
#             #     img_w, img_h,
#             #     self.iou_threshold,
#             #     depth_emas_full=depth_emas
#             # )
            
#             matched, unmatched_dets, unmatched_trks = self._associate_round1(
#                 dets, trks,
#                 height_mus_full, height_sigmas_full, miss_counts_full,
#                 img_w, img_h,
#                 self.iou_threshold,
#                 depth_emas_full=depth_emas,
#                 trk_score_proxy_full=track_score_proxy_full
#             )
            

            
#             for d_i, t_i in matched:
#                 self.trackers[t_i].update(dets[d_i, :])
#                 self._update_pc_model_with_det(dets[d_i, :], img_h)
#         else:
#             matched = np.empty((0, 2), dtype=int)
#             unmatched_dets = np.arange(dets.shape[0])
#             unmatched_trks = np.arange(trks.shape[0])

#         # ---------- Second round: ORIGINAL BYTE + optional TCM ----------
#         if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
#             u_trks = trks[unmatched_trks]
#             iou_left = self.asso_func(dets_second[:, :4], u_trks[:, :4])
#             iou_left = np.array(iou_left)

#             if iou_left.size > 0:
#                 if self.ua_enable:
#                     u_mu = height_mus_full[unmatched_trks]
#                     u_sigma = height_sigmas_full[unmatched_trks]
#                     u_miss = miss_counts_full[unmatched_trks]
#                     det_sc = dets_second[:, 4]
#                     A_u = self._ua_height_affinity(
#                         dets_second[:, :4], u_trks[:, :4],
#                         u_mu, u_sigma,
#                         det_sc, u_miss,
#                         img_w, img_h
#                     )
#                 else:
#                     A_u = np.ones_like(iou_left)

#                 A_p = self._pc_affinity(dets_second[:, :4], u_trks[:, :4], img_h)
#                 iou_geom = iou_left * A_u * A_p
#             else:
#                 iou_geom = iou_left

#             if iou_geom.size > 0:
#                 u_trk_emas = depth_emas[unmatched_trks]
#                 A_d = self._depth_affinity(dets_second[:, :4], u_trks[:, :4], u_trk_emas, img_h) if self.depth_enable else np.ones_like(iou_geom)
#                 iou_adj = self._blend_iou_with_depth(iou_geom, A_d)
#             else:
#                 iou_adj = iou_geom

#             # Hybrid-style TCM port: only re-rank ambiguous low-score matching, keep raw IoU gate.
#             score_left = iou_adj.copy()
#             if self.tcm_round2_enable and score_left.size > 0:
#                 u_score_proxy = track_score_proxy_full[unmatched_trks]
#                 tcm_penalty = cal_score_dif_batch_two_score(dets_second, u_score_proxy)
#                 score_left = score_left - self.tcm_round2_weight * tcm_penalty

#             thr2 = max(0.1, self.iou_threshold - 0.07)
#             if score_left.size > 0 and score_left.max() > thr2:
#                 matched_indices = linear_assignment(-score_left)
#                 to_remove_trk_indices = []
#                 to_remove_det_indices = []
#                 for m2 in matched_indices:
#                     det_local, trk_local = m2[0], m2[1]
#                     # acceptance still uses raw geometric IoU, not TCM-adjusted score
#                     if iou_left[det_local, trk_local] < thr2:
#                         continue
#                     trk_ind = unmatched_trks[trk_local]
#                     self.trackers[trk_ind].update(dets_second[det_local, :])
#                     to_remove_trk_indices.append(trk_ind)
#                     to_remove_det_indices.append(det_local)
#                     self._update_pc_model_with_det(dets_second[det_local, :], img_h)
#                 if len(to_remove_trk_indices) > 0:
#                     unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))
#                 if len(to_remove_det_indices) > 0:
#                     keep_mask = np.ones(len(dets_second), dtype=bool)
#                     keep_mask[to_remove_det_indices] = False
#                     dets_second = dets_second[keep_mask]

#         # ---------- Third round: ORIGINAL RE-ASSOC + optional TCM (+ optional ROCM hook) ----------
#         if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
#             left_dets = dets[unmatched_dets]
#             left_trks_last = last_boxes[unmatched_trks]
#             left_trks_pred = trks[unmatched_trks]

#             iou_last = self.asso_func(left_dets[:, :4], left_trks_last[:, :4])
#             iou_pred = self.asso_func(left_dets[:, :4], left_trks_pred[:, :4])
#             iou_last = np.array(iou_last)
#             iou_pred = np.array(iou_pred)
#             iou_max = iou_last if iou_pred.size == 0 else np.maximum(iou_last, iou_pred)

#             if iou_max.size > 0:
#                 dets_xy = left_dets[:, :4]
#                 trks_xy = left_trks_pred[:, :4]

#                 if self.ua_enable:
#                     l_mu = height_mus_full[unmatched_trks]
#                     l_sigma = height_sigmas_full[unmatched_trks]
#                     l_miss = miss_counts_full[unmatched_trks]
#                     det_sc = left_dets[:, 4]
#                     A_u = self._ua_height_affinity(
#                         dets_xy, trks_xy,
#                         l_mu, l_sigma,
#                         det_sc, l_miss,
#                         img_w, img_h
#                     )
#                 else:
#                     A_u = np.ones_like(iou_max)

#                 A_p = self._pc_affinity(dets_xy, trks_xy, img_h)
#                 iou_geom = iou_max * A_u * A_p

#                 trk_emas = depth_emas[unmatched_trks]
#                 A_d = self._depth_affinity(dets_xy, trks_xy, trk_emas, img_h) if self.depth_enable else np.ones_like(iou_geom)
#                 iou_adj = self._blend_iou_with_depth(iou_geom, A_d)
#             else:
#                 iou_adj = iou_max

#             score_rematch = iou_adj.copy()
#             if self.tcm_round3_enable and score_rematch.size > 0:
#                 u_score_proxy = track_score_proxy_full[unmatched_trks]
#                 tcm_penalty = cal_score_dif_batch_two_score(left_dets, u_score_proxy)
#                 score_rematch = score_rematch - self.tcm_round3_weight * tcm_penalty

#             if self.rocm_round3_enable and score_rematch.size > 0:
#                 rocm_cost = self._rocm_round3_cost(
#                     left_dets[:, :4],
#                     left_trks_last,
#                     velocities_lt_full[unmatched_trks],
#                     velocities_rt_full[unmatched_trks],
#                     velocities_lb_full[unmatched_trks],
#                     velocities_rb_full[unmatched_trks],
#                 )
#                 score_rematch = score_rematch + self.rocm_weight * rocm_cost

#             thr3 = max(0.1, self.iou_threshold - 0.07)
#             if score_rematch.size > 0 and score_rematch.max() > thr3:
#                 rematched_indices = linear_assignment(-score_rematch)
#                 to_remove_det_indices = []
#                 to_remove_trk_indices = []
#                 for m3 in rematched_indices:
#                     det_local, trk_local = m3[0], m3[1]
#                     if iou_max[det_local, trk_local] < thr3:
#                         continue
#                     det_ind = unmatched_dets[det_local]
#                     trk_ind = unmatched_trks[trk_local]
#                     self.trackers[trk_ind].update(dets[det_ind, :])
#                     to_remove_det_indices.append(det_ind)
#                     to_remove_trk_indices.append(trk_ind)
#                     self._update_pc_model_with_det(dets[det_ind, :], img_h)
#                 if len(to_remove_det_indices) > 0:
#                     unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
#                 if len(to_remove_trk_indices) > 0:
#                     unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

#         for m in unmatched_trks:
#             self.trackers[m].update(None)

#         for i in unmatched_dets:
#             trk = KalmanBoxTracker(
#                 dets[i, :],
#                 delta_t=self.delta_t,
#                 ema_obs=self.ema_obs,
#                 ema_pred=self.ema_pred,
#                 ua_h_ema=self.ua_h_ema,
#                 ua_h2_ema=self.ua_h2_ema,
#                 ua_proc_var_frac=self.ua_proc_var_frac,
#                 ua_sigma_floor_frac=self.ua_sigma_floor_frac,
#                 score_ema=self.score_ema,
#             )
#             self.trackers.append(trk)
#             self._update_pc_model_with_det(dets[i, :], img_h)

#         i = len(self.trackers)
#         for trk in reversed(self.trackers):
#             if trk.last_observation.sum() < 0:
#                 d = trk.get_state()[0]
#             else:
#                 d = trk.last_observation[:4]
#             if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
#                 ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
#             i -= 1
#             if trk.time_since_update > self.max_age:
#                 self.trackers.pop(i)

#         if len(ret) > 0:
#             return np.concatenate(ret)
#         return np.empty((0, 5))

#     def update_public(self, dets, cates, scores):
#         # Keep baseline behavior for public detections.
#         self.frame_count += 1

#         det_scores = np.ones((dets.shape[0], 1))
#         dets = np.concatenate((dets, det_scores), axis=1)

#         remain_inds = scores > self.det_thresh
#         cates = cates[remain_inds]
#         dets = dets[remain_inds]

#         trks = np.zeros((len(self.trackers), 5))
#         to_del = []
#         ret = []
#         for t, trk in enumerate(trks):
#             pos = self.trackers[t].predict()[0]
#             cat = getattr(self.trackers[t], "cate", 1)
#             trk[:] = [pos[0], pos[1], pos[2], pos[3], cat]
#             if np.any(np.isnan(pos)):
#                 to_del.append(t)
#         trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#         for t in reversed(to_del):
#             self.trackers.pop(t)

#         velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers])
#         last_boxes = np.array([trk.last_observation for trk in self.trackers])
#         k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

#         matched, unmatched_dets, unmatched_trks = associate_kitti(
#             dets, trks, cates, self.iou_threshold, velocities, k_observations, self.inertia
#         )
#         for m in matched:
#             self.trackers[m[1]].update(dets[m[0], :])
#             self._update_pc_model_with_det(dets[m[0], :], img_h=1.0)

#         if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
#             left_dets = dets[unmatched_dets]
#             left_trks = last_boxes[unmatched_trks]
#             left_dets_c = left_dets.copy()
#             left_trks_c = left_trks.copy()

#             iou_left = iou_batch(left_dets_c, left_trks_c)
#             iou_left = np.array(iou_left)
#             det_cates_left = cates[unmatched_dets]
#             trk_cates_left = trks[unmatched_trks][:, 4]
#             num_dets = unmatched_dets.shape[0]
#             num_trks = unmatched_trks.shape[0]
#             cate_matrix = np.zeros((num_dets, num_trks))
#             for i in range(num_dets):
#                 for j in range(num_trks):
#                     if det_cates_left[i] != trk_cates_left[j]:
#                         cate_matrix[i][j] = -1e6
#             iou_left = iou_left + cate_matrix
#             if iou_left.max() > self.iou_threshold - 0.1:
#                 rematched_indices = linear_assignment(-iou_left)
#                 to_remove_det_indices = []
#                 to_remove_trk_indices = []
#                 for m in rematched_indices:
#                     det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
#                     if iou_left[m[0], m[1]] < self.iou_threshold - 0.1:
#                         continue
#                     self.trackers[trk_ind].update(dets[det_ind, :])
#                     to_remove_det_indices.append(det_ind)
#                     to_remove_trk_indices.append(trk_ind)
#                     self._update_pc_model_with_det(dets[det_ind, :], img_h=1.0)
#                 unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
#                 unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

#         for i in unmatched_dets:
#             trk = KalmanBoxTracker(
#                 dets[i, :],
#                 ema_obs=self.ema_obs,
#                 ema_pred=self.ema_pred,
#                 ua_h_ema=self.ua_h_ema,
#                 ua_h2_ema=self.ua_h2_ema,
#                 ua_proc_var_frac=self.ua_proc_var_frac,
#                 ua_sigma_floor_frac=self.ua_sigma_floor_frac,
#                 score_ema=self.score_ema,
#             )
#             trk.cate = cates[i]
#             self.trackers.append(trk)
#             self._update_pc_model_with_det(dets[i, :], img_h=1.0)

#         i = len(self.trackers)
#         for trk in reversed(self.trackers):
#             if trk.last_observation.sum() > 0:
#                 d = trk.last_observation[:4]
#             else:
#                 d = trk.get_state()[0]
#             if trk.time_since_update < 1:
#                 if (self.frame_count <= self.min_hits) or (trk.hit_streak >= self.min_hits):
#                     ret.append(np.concatenate((d, [trk.id + 1], [getattr(trk, "cate", 1)], [0])).reshape(1, -1))
#                 if trk.hit_streak == self.min_hits:
#                     for prev_i in range(self.min_hits - 1):
#                         prev_observation = trk.history_observations[-(prev_i + 2)]
#                         ret.append((np.concatenate((prev_observation[:4], [trk.id + 1], [getattr(trk, "cate", 1)],
#                                                     [-(prev_i + 1)]))).reshape(1, -1))
#             i -= 1
#             if trk.time_since_update > self.max_age:
#                 self.trackers.pop(i)

#         if len(ret) > 0:
#             return np.concatenate(ret)
#         return np.empty((0, 7))



# -*- coding: utf-8 -*-
"""
OC-SORT backbone with:
- Soft-Depth Regularization
- UA-HMIoU (Uncertainty-Aware Height Modulated IoU)
- PC-HMIoU (Perspective-Consistent HMIoU)

This version keeps the original 3-component pipeline as the backbone and
lightly ports Hybrid-SORT cues in a conservative way:
- TCM is added ONLY to round-2 (BYTE-like) and round-3 (re-association)
- ROCM utilities / state are added, but round-1 is kept unchanged
- Optional ROCM rerank is provided for round-3 only and is OFF by default

Key design choice:
- round-1 remains identical to the user's original tracker because that was
  the strongest baseline.
- acceptance thresholds still use raw geometric IoU / max(IoU(last), IoU(pred)).
"""

from __future__ import print_function
import numpy as np

from .association import *  # iou_batch, giou_batch, ciou_batch, diou_batch, ct_dist, associate_kitti, linear_assignment


# ---------- safe type conversions ----------
def _to_numpy(x, dtype=float):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(dtype, copy=False)
    except Exception:
        pass
    if isinstance(x, np.ndarray):
        return x.astype(dtype, copy=False)
    return np.array(x, dtype=dtype)


def _to_scalar(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return float(x.detach().cpu().item())
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return float(np.array(x).reshape(-1)[0])


# ---------- helpers ----------
def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_bbox_to_z(bbox):
    # [x1,y1,x2,y2,(score)] -> z = [x,y,s,r]^T
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    # [x,y,s,r,(vx,vy,vs)] -> [x1,y1,x2,y2,(score)]
    w = np.sqrt(x[2] * x[3])
    h = x[2] / (w + 1e-6)
    if score is None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


# ---------- Hybrid-SORT TCM score-difference helpers ----------
def cal_score_dif_batch(bboxes1, bboxes2):
    """Absolute difference between detection score and tracker score proxy.

    bboxes1: (Nd, >=5), use column 4 as detection score
    bboxes2: (Nt, >=5), use column 4 as tracker score proxy
    returns: (Nd, Nt)
    """
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return np.zeros((len(bboxes1), len(bboxes2)), dtype=float)
    bboxes2 = np.expand_dims(np.asarray(bboxes2, dtype=float), 0)
    bboxes1 = np.expand_dims(np.asarray(bboxes1, dtype=float), 1)
    score2 = bboxes2[..., 4]
    score1 = bboxes1[..., 4]
    return np.abs(score2 - score1)


def cal_score_dif_batch_two_score(det_boxes, trk_score_proxy):
    """BYTE-step style score difference.

    det_boxes: (Nd, >=5), column 4 is detection score
    trk_score_proxy: (Nt,) or (Nt,1), tracker-side predicted/simple score proxy
    returns: (Nd, Nt)
    """
    det_scores = np.asarray(det_boxes, dtype=float)[:, 4:5]  # (Nd,1)
    trk_scores = np.asarray(trk_score_proxy, dtype=float).reshape(1, -1)  # (1,Nt)
    return np.abs(det_scores - trk_scores)


# ---------- Hybrid-SORT ROCM helpers (optional / conservative) ----------
def speed_direction_batch_lt(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = dets[:, 0], dets[:, 1]
    CX2, CY2 = tracks[:, 0], tracks[:, 1]
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx


def speed_direction_batch_rt(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = dets[:, 0], dets[:, 3]
    CX2, CY2 = tracks[:, 0], tracks[:, 3]
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx


def speed_direction_batch_lb(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = dets[:, 2], dets[:, 1]
    CX2, CY2 = tracks[:, 2], tracks[:, 1]
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx


def speed_direction_batch_rb(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = dets[:, 2], dets[:, 3]
    CX2, CY2 = tracks[:, 2], tracks[:, 3]
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx


def cost_vel(Y, X, trackers, velocities, detections, previous_obs, vdc_weight):
    inertia_Y, inertia_X = velocities[:, 0], velocities[:, 1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0
    scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores
    return angle_diff_cost


# ---------- Kalman tracker with UA-height stats + lightweight TCM/ROCM state ----------
class KalmanBoxTracker(object):
    """
    OC-SORT Kalman tracker (7D state) +
    depth_ema (EMA of bottom y2) for pseudo-depth prior +
    UA-HMIoU height stats (EMA of h, h^2) with variance inflation +
    lightweight score proxy for TCM and 4-point velocities for optional ROCM.
    """
    count = 0

    def __init__(
        self,
        bbox,
        delta_t=3,
        orig=False,
        ema_obs=0.5,
        ema_pred=0.1,
        # UA-HMIoU params for height stats
        ua_h_ema=0.5,
        ua_h2_ema=0.3,
        ua_proc_var_frac=0.02,
        ua_sigma_floor_frac=0.05,
        # TCM score proxy
        score_ema=0.7,
    ):
        if not orig:
            from .kalmanfilter import KalmanFilterNew as KalmanFilter
            self.kf = KalmanFilter(dim_x=7, dim_z=4)
        else:
            from filterpy.kalman import KalmanFilter
            self.kf = KalmanFilter(dim_x=7, dim_z=4)

        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        self.last_observation = np.array([-1, -1, -1, -1, -1])
        self.observations = dict()
        self.history_observations = []
        self.velocity = None
        self.delta_t = delta_t

        # Optional 4-point velocities for future ROCM
        self.velocity_lt = None
        self.velocity_rt = None
        self.velocity_lb = None
        self.velocity_rb = None

        # Depth EMA (pseudo-depth y2)
        self.ema_obs = float(ema_obs)
        self.ema_pred = float(ema_pred)
        self.depth_ema = float(bbox[3])

        # UA-HMIoU height stats
        h0 = float(bbox[3] - bbox[1])
        self.ua_h_ema = float(ua_h_ema)
        self.ua_h2_ema = float(ua_h2_ema)
        self.ua_proc_var_frac = float(ua_proc_var_frac)
        self.ua_sigma_floor_frac = float(ua_sigma_floor_frac)

        self.h_m1 = h0
        self.h_m2 = h0 * h0
        floor_var = (self.ua_sigma_floor_frac * h0) ** 2
        self.h_var = max(self.h_m2 - self.h_m1 * self.h_m1, floor_var)

        # # TCM score proxy
        # init_score = float(bbox[4]) if len(bbox) >= 5 else 1.0
        # self.score_ema_decay = float(score_ema)
        # self.confidence = init_score
        # self.confidence_pre = None
        # self.simple_score = init_score
        
        # TCM: paper-style confidence modeling
        # round1 -> Kalman confidence
        # round2 -> Linear confidence from tracklet memory
        from filterpy.kalman import KalmanFilter as ScoreKalmanFilter

        init_score = float(bbox[4]) if len(bbox) >= 5 else 1.0
        init_score = float(np.clip(init_score, 0.1, 1.0))

        # confidence history for linear prediction (tracklet memory)
        self.conf_history = [init_score]

        # a tiny 1D constant-velocity Kalman filter on confidence: [c, c_dot]
        self.kf_score = ScoreKalmanFilter(dim_x=2, dim_z=1)
        self.kf_score.F = np.array([[1., 1.],
                                    [0., 1.]])
        self.kf_score.H = np.array([[1., 0.]])
        self.kf_score.R *= 1e-2
        self.kf_score.P *= 10.0
        self.kf_score.P[1, 1] *= 10.0
        self.kf_score.Q = np.array([[1e-4, 0.0],
                                    [0.0, 1e-3]])
        self.kf_score.x = np.array([[init_score],
                                    [0.0]])

        # cached scores used by association
        self.kalman_score = init_score   # for round1 high-conf association
        self.simple_score = init_score   # for round2 low-conf association
        
        

    def get_height_mu_sigma(self):
        mu = float(self.h_m1)
        floor_var = (self.ua_sigma_floor_frac * max(mu, 1e-6)) ** 2
        var = max(float(self.h_var), floor_var)
        sigma = np.sqrt(var)
        return mu, sigma

    def _inflation_step(self):
        mu = max(float(self.h_m1), 1e-6)
        self.h_var = float(self.h_var) + (self.ua_proc_var_frac * mu) ** 2

    def _update_corner_velocities(self, previous_box, bbox):
        def _corner_dir(b1, b2, x_idx, y_idx):
            p1 = np.array([b1[x_idx], b1[y_idx]], dtype=float)
            p2 = np.array([b2[x_idx], b2[y_idx]], dtype=float)
            v = np.array([p2[1] - p1[1], p2[0] - p1[0]], dtype=float)
            n = np.sqrt(v[0] ** 2 + v[1] ** 2) + 1e-6
            return v / n

        self.velocity_lt = _corner_dir(previous_box, bbox, 0, 1)
        self.velocity_rt = _corner_dir(previous_box, bbox, 0, 3)
        self.velocity_lb = _corner_dir(previous_box, bbox, 2, 1)
        self.velocity_rb = _corner_dir(previous_box, bbox, 2, 3)

    # def get_tcm_score_proxy(self, low=0.1, high=1.0):
    #     if self.confidence_pre is None:
    #         s = self.confidence
    #     else:
    #         s = self.confidence - (self.confidence_pre - self.confidence)
    #     return float(np.clip(s, low, high))
    
    
    def get_tcm_kalman_score(self, low=0.1, high=1.0):
        return float(np.clip(float(self.kf_score.x[0, 0]), low, high))

    def get_tcm_linear_score(self, low=0.1, high=1.0):
        # paper Eq.(3): c_hat = c_{t-1} if c_{t-2} is None else 2*c_{t-1} - c_{t-2}
        if len(self.conf_history) < 2:
            s = self.conf_history[-1]
        else:
            s = self.conf_history[-1] - (self.conf_history[-2] - self.conf_history[-1])
        return float(np.clip(float(s), low, high))

    # keep an alias only to avoid breaking old calls
    def get_tcm_score_proxy(self, low=0.1, high=1.0):
        return self.get_tcm_linear_score(low=low, high=high)
    

    def update(self, bbox):
        if bbox is not None:
            bbox = np.asarray(bbox, dtype=float)
            if self.last_observation.sum() >= 0:
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                self.velocity = speed_direction(previous_box, bbox)
                self._update_corner_velocities(previous_box, bbox)

            # Depth EMA with observation
            y2 = float(bbox[3])
            self.depth_ema = (1.0 - self.ema_obs) * self.depth_ema + self.ema_obs * y2

            # Height stats update
            h = float(bbox[3] - bbox[1])
            self.h_m1 = (1.0 - self.ua_h_ema) * self.h_m1 + self.ua_h_ema * h
            self.h_m2 = (1.0 - self.ua_h2_ema) * self.h_m2 + self.ua_h2_ema * (h * h)
            raw_var = max(self.h_m2 - self.h_m1 * self.h_m1, 0.0)
            floor_var = (self.ua_sigma_floor_frac * max(self.h_m1, 1e-6)) ** 2
            self.h_var = 0.5 * self.h_var + 0.5 * max(raw_var, floor_var)

            # # TCM score proxy update
            # score = float(bbox[4]) if len(bbox) >= 5 else self.confidence
            # self.confidence_pre = self.confidence
            # self.confidence = (1.0 - self.score_ema_decay) * self.confidence + self.score_ema_decay * score
            # self.simple_score = self.get_tcm_score_proxy(low=0.1, high=1.0)
            
            # TCM confidence update
            score = float(bbox[4]) if len(bbox) >= 5 else self.conf_history[-1]
            score = float(np.clip(score, 0.1, 1.0))

            # update confidence Kalman filter with measurement
            self.kf_score.update(np.array([[score]], dtype=float))

            # update tracklet memory for linear prediction
            self.conf_history.append(score)
            if len(self.conf_history) > 8:
                self.conf_history = self.conf_history[-8:]

            self.kalman_score = self.get_tcm_kalman_score(low=0.1, high=1.0)
            self.simple_score = self.get_tcm_linear_score(low=0.1, high=1.0)
            

            # bookkeeping
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1

            self.kf.update(convert_bbox_to_z(bbox))
        else:
            self.kf.update(bbox)

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        pred = convert_x_to_bbox(self.kf.x)
        self.history.append(pred)

        # gently pull depth_ema towards predicted y2
        pred_y2 = float(pred[0][3])
        self.depth_ema = (1.0 - self.ema_pred) * self.depth_ema + self.ema_pred * pred_y2

        # process noise inflation for UA-height; only once per frame in predict()
        self._inflation_step()

        # # score proxy does not advance aggressively; keep simple extrapolation stable
        # self.simple_score = self.get_tcm_score_proxy(low=0.1, high=1.0)
        # return self.history[-1]
        
        # TCM confidence prediction:
        # round1 uses Kalman-predicted confidence,
        # round2 uses linear prediction from memory
        self.kf_score.predict()
        self.kalman_score = self.get_tcm_kalman_score(low=0.1, high=1.0)
        self.simple_score = self.get_tcm_linear_score(low=0.1, high=1.0)
        
#####单条 track 的 conf_history / kalman_score / simple_score 打印
        # if (self.id < 3) and (self.age % 30 == 0):
        #     tail = self.conf_history[-3:] if len(self.conf_history) >= 3 else self.conf_history
        #     print(
        #         f"[TCM-TRACK] id={self.id} age={self.age} "
        #         f"conf_tail={tail} "
        #         f"kalman_score={self.kalman_score:.4f} "
        #         f"linear_score={self.simple_score:.4f}"
        #     )

        return self.history[-1]
    
        

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)


ASSO_FUNCS = {
    "iou": iou_batch,
    "giou": giou_batch,
    "ciou": ciou_batch,
    "diou": diou_batch,
    "ct_dist": ct_dist
}


# ---------- Perspective height model (PC-HMIoU core) ----------
class PerspectiveHeightModel:
    """
    Online mapping h*(y2): bottom y2 -> expected pixel height
    Implemented via normalized y2 binning + EMA; supports interpolation and fallback.
    """
    def __init__(self, num_bins=24, ema=0.08, min_count=2, floor=0.3, gamma=2.0):
        self.num_bins = int(num_bins)
        self.ema = float(ema)
        self.min_count = int(min_count)
        self.floor = float(floor)
        self.gamma = float(gamma)
        self.reset()

    def reset(self):
        self.bin_means = np.zeros(self.num_bins, dtype=float)
        self.bin_counts = np.zeros(self.num_bins, dtype=float)
        self.global_ema = 0.0
        self.global_count = 0.0

    def _bin_index(self, y2_norm):
        y = np.clip(y2_norm, 0.0, 1.0 - 1e-8)
        return np.floor(y * self.num_bins).astype(int)

    def observe(self, y2_pixel, h_pixel, img_h):
        if img_h <= 0:
            return
        y2_norm = float(y2_pixel) / float(img_h)
        if not np.isfinite(y2_norm) or not np.isfinite(h_pixel) or h_pixel <= 0:
            return
        b = int(self._bin_index(y2_norm))
        m = self.bin_means[b]
        c = self.bin_counts[b]
        m_new = h_pixel if c < 1e-6 else (1.0 - self.ema) * m + self.ema * h_pixel
        self.bin_means[b] = m_new
        self.bin_counts[b] = min(c + 1.0, 1e9)
        self.global_ema = h_pixel if self.global_count < 1e-6 else 0.98 * self.global_ema + 0.02 * h_pixel
        self.global_count = min(self.global_count + 1.0, 1e9)

    def expected_vec(self, y2_pixels, img_h):
        if img_h <= 0 or len(y2_pixels) == 0:
            return np.zeros_like(y2_pixels, dtype=float)
        y2n = np.clip(np.asarray(y2_pixels, dtype=float) / float(img_h), 0.0, 1.0 - 1e-8)
        bins = np.floor(y2n * self.num_bins).astype(int)
        pos = y2n * self.num_bins
        frac = pos - np.floor(pos)
        b0 = bins
        b1 = np.clip(bins + 1, 0, self.num_bins - 1)
        m0 = self.bin_means[b0]
        m1 = self.bin_means[b1]
        c0 = self.bin_counts[b0]
        c1 = self.bin_counts[b1]
        use0 = (c0 >= self.min_count)
        use1 = (c1 >= self.min_count)
        m0_eff = np.where(use0, m0, np.where(use1, m1, self.global_ema))
        m1_eff = np.where(use1, m1, np.where(use0, m0, self.global_ema))
        h_star = (1.0 - frac) * m0_eff + frac * m1_eff
        h_star = np.maximum(h_star, 4.0)
        return h_star

    def affinity_matrix(self, dets_xyxy, trks_xyxy, img_h):
        Nd = dets_xyxy.shape[0]
        Nt = trks_xyxy.shape[0]
        if Nd == 0 or Nt == 0:
            return np.zeros((Nd, Nt), dtype=float)
        h_det = (dets_xyxy[:, 3] - dets_xyxy[:, 1]).astype(float)[:, None]
        y2_trk = trks_xyxy[:, 3].astype(float)
        h_star = self.expected_vec(y2_trk, img_h)[None, :]
        eps = 1e-6
        r = np.abs(h_det - h_star) / (h_star + eps)
        A_p = np.exp(-self.gamma * r)
        if self.floor is not None and self.floor > 0.0:
            A_p = np.maximum(A_p, self.floor)
        return A_p

    def ready(self):
        return self.global_count >= 50


# ---------- main tracker with original 3 components + conservative TCM/ROCM ports ----------
class Soperm_Track_Hybird(object):
    def __init__(
        self,
        det_thresh,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        delta_t=3,
        asso_func="iou",
        inertia=0.2,
        use_byte=True,
        # Soft-depth params
        depth_alpha=0.5,
        depth_beta=2.0,
        depth_gate=0.25,
        gate_floor=0.2,
        depth_enable=True,
        # UA-HMIoU params
        ua_enable=True,
        ua_mode="cauchy",
        ua_alpha=1.0,
        ua_border_margin_frac=0.02,
        ua_border_boost=0.6,
        ua_miss_boost=0.6,
        ua_low_score=0.4,
        ua_score_boost=0.3,
        ua_sigma_floor_frac=0.05,
        ua_proc_var_frac=0.02,
        ua_h_ema=0.5,
        ua_h2_ema=0.3,
        # PC-HMIoU params
        pc_enable=True,
        pc_bins=24,
        pc_ema=0.08,
        pc_gamma=2.0,
        pc_floor=0.3,
        pc_min_count=2,
        pc_sample_score=0.6,
        # Depth EMA
        ema_obs=0.5,
        ema_pred=0.1,
        # TCM first-step (very conservative)
        tcm_round1_enable=False,
        tcm_round1_weight=0.03,
        tcm_round1_margin=0.05,
        # TCM (ported conservatively: rounds 2 & 3 only)
        tcm_round2_enable=True,
        tcm_round2_weight=0.08,
        tcm_round3_enable=True,
        tcm_round3_weight=0.10,
        # ROCM hooks (disabled by default for scheme-1)
        rocm_round3_enable=False,
        rocm_weight=0.08,
        # score proxy for TCM
        score_ema=0.7,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.use_byte = use_byte

        # Soft-depth
        self.depth_alpha = float(depth_alpha)
        self.depth_beta = float(depth_beta)
        self.depth_gate = float(depth_gate)
        self.gate_floor = float(gate_floor)
        self.depth_enable = bool(depth_enable)

        # UA-HMIoU
        self.ua_enable = bool(ua_enable)
        self.ua_mode = str(ua_mode).lower()
        assert self.ua_mode in ["cauchy", "gauss"]
        self.ua_alpha = float(ua_alpha)
        self.ua_border_margin_frac = float(ua_border_margin_frac)
        self.ua_border_boost = float(ua_border_boost)
        self.ua_miss_boost = float(ua_miss_boost)
        self.ua_low_score = float(ua_low_score)
        self.ua_score_boost = float(ua_score_boost)
        self.ua_sigma_floor_frac = float(ua_sigma_floor_frac)
        self.ua_proc_var_frac = float(ua_proc_var_frac)
        self.ua_h_ema = float(ua_h_ema)
        self.ua_h2_ema = float(ua_h2_ema)

        # PC-HMIoU
        self.pc_enable = bool(pc_enable)
        self.pc_sample_score = float(pc_sample_score)
        self.perspective = PerspectiveHeightModel(
            num_bins=pc_bins, ema=pc_ema, min_count=pc_min_count, floor=pc_floor, gamma=pc_gamma
        )

        # Depth EMA
        self.ema_obs = float(ema_obs)
        self.ema_pred = float(ema_pred)

        # TCM/ROCM
        
        self.tcm_round1_enable = bool(tcm_round1_enable)
        self.tcm_round1_weight = float(tcm_round1_weight)
        self.tcm_round1_margin = float(tcm_round1_margin)


        
        self.tcm_round2_enable = bool(tcm_round2_enable)
        self.tcm_round2_weight = float(tcm_round2_weight)
        self.tcm_round3_enable = bool(tcm_round3_enable)
        self.tcm_round3_weight = float(tcm_round3_weight)
        self.rocm_round3_enable = bool(rocm_round3_enable)
        
        
        self.rocm_weight = float(rocm_weight)
        self.score_ema = float(score_ema)

        KalmanBoxTracker.count = 0

        self._prev_frame_id = -1
        self.last_img_size = None

    def reset(self, reset_perspective=True, reset_id_counter=False):
        self.trackers = []
        self.frame_count = 0
        self._prev_frame_id = -1
        self.last_img_size = None
        if reset_perspective and getattr(self, "pc_enable", False) and hasattr(self, "perspective"):
            self.perspective.reset()
        if reset_id_counter:
            KalmanBoxTracker.count = 0

    def _maybe_reset_on_boundary(self, frame_id=None, img_size=None,
                                 reset_on_size_change=False,
                                 verbose=False):
        fid_back = (self._prev_frame_id != -1 and
                    frame_id is not None and int(frame_id) <= int(self._prev_frame_id))
        is_new_seq = False

        if frame_id is not None:
            fid = int(frame_id)
            if fid == 1 or fid_back:
                is_new_seq = True

        size_changed = False
        if reset_on_size_change and img_size is not None and self.last_img_size is not None:
            size_changed = tuple(img_size) != tuple(self.last_img_size)
            if size_changed:
                is_new_seq = True

        if is_new_seq:
            if getattr(self, "pc_enable", False) and hasattr(self, "perspective"):
                self.perspective.reset()
                if verbose:
                    print(f"[OCSort] perspective reset at boundary (frame_id={frame_id}, "
                          f"fid_back={fid_back}, size_changed={size_changed})")

        if frame_id is not None:
            self._prev_frame_id = int(frame_id)
        if img_size is not None:
            self.last_img_size = tuple(img_size)

    # ---------- Soft-depth helpers ----------
    @staticmethod
    def _y2_from_boxes(boxes):
        return boxes[:, 3].astype(float) if boxes.size > 0 else np.array([])

    def _depth_affinity(self, dets_xyxy, trks_xyxy, trk_depth_ema, img_h):
        Nd = dets_xyxy.shape[0]
        Nt = trks_xyxy.shape[0]
        if Nd == 0 or Nt == 0:
            return np.zeros((Nd, Nt), dtype=float)
        y2d = dets_xyxy[:, 3].astype(float)[:, None]
        y2t = trk_depth_ema.astype(float)[None, :]
        norm = max(float(img_h), 1e-6)
        d = np.abs(y2d - y2t) / norm
        A = np.exp(-self.depth_beta * d)
        if self.depth_gate > 0:
            mask_far = d > self.depth_gate
            if self.gate_floor >= 0.0:
                A[mask_far] = np.maximum(A[mask_far], self.gate_floor)
        return A

    # ---------- UA-HMIoU helpers ----------
    def _ua_height_affinity(
        self,
        dets_xyxy, trks_xyxy,
        trk_mu_h, trk_sigma_h,
        det_scores, trk_miss,
        img_w, img_h
    ):
        dets_xyxy = _to_numpy(dets_xyxy, dtype=float)
        trks_xyxy = _to_numpy(trks_xyxy, dtype=float)
        trk_mu_h = _to_numpy(trk_mu_h, dtype=float).reshape(-1)
        trk_sigma_h = _to_numpy(trk_sigma_h, dtype=float).reshape(-1)
        det_scores = _to_numpy(det_scores, dtype=float).reshape(-1)
        trk_miss = _to_numpy(trk_miss, dtype=float).reshape(-1)
        img_w = _to_scalar(img_w)
        img_h = _to_scalar(img_h)

        Nd = dets_xyxy.shape[0]
        Nt = trks_xyxy.shape[0]
        if Nd == 0 or Nt == 0:
            return np.zeros((Nd, Nt), dtype=float)

        eps = 1e-6
        hd = (dets_xyxy[:, 3] - dets_xyxy[:, 1]).astype(float)[:, None]
        mu = trk_mu_h.astype(float)[None, :]
        sigma_base = trk_sigma_h.astype(float)[None, :]

        miss = np.clip(trk_miss.astype(float), 0.0, 2.0)[None, :]
        S_trk = 1.0 + self.ua_miss_boost * (miss / 2.0)

        if Nd > 0:
            m = max(1.0, min(float(img_w), float(img_h)) * self.ua_border_margin_frac)
            x1 = dets_xyxy[:, 0].astype(float)
            y1 = dets_xyxy[:, 1].astype(float)
            x2 = dets_xyxy[:, 2].astype(float)
            y2 = dets_xyxy[:, 3].astype(float)
            min_dist = np.minimum(np.minimum(x1, y1), np.minimum(img_w - x2, img_h - y2))
            closeness = np.clip(1.0 - (min_dist / m), 0.0, 1.0)
            sc = det_scores.astype(float)
            low_score_gap = np.clip((self.ua_low_score - sc) / max(self.ua_low_score, 1e-6), 0.0, 1.0)
            S_det = 1.0 + self.ua_border_boost * closeness + self.ua_score_boost * low_score_gap
            S_det = S_det[:, None]
        else:
            S_det = 1.0

        sigma_eff = sigma_base * S_trk
        sigma_eff = sigma_eff * S_det
        sigma_eff = np.maximum(sigma_eff, eps)

        z = np.abs(hd - mu) / (self.ua_alpha * sigma_eff + eps)
        if self.ua_mode == "gauss":
            A_u = np.exp(-0.5 * z * z)
        else:
            A_u = 1.0 / (1.0 + z * z)
        return A_u

    # ---------- PC-HMIoU helpers ----------
    def _pc_affinity(self, dets_xyxy, trks_xyxy, img_h):
        if not self.pc_enable or not self.perspective.ready():
            Nd = dets_xyxy.shape[0]
            Nt = trks_xyxy.shape[0]
            if Nd == 0 or Nt == 0:
                return np.zeros((Nd, Nt), dtype=float)
            return np.ones((Nd, Nt), dtype=float)
        return self.perspective.affinity_matrix(dets_xyxy, trks_xyxy, img_h)

    # ---------- Soft-depth blending ----------
    def _blend_iou_with_depth(self, iou, affinity):
        if (not self.depth_enable) or self.depth_alpha <= 0:
            return iou
        return iou * ((1.0 - self.depth_alpha) + self.depth_alpha * affinity)

    # ---------- perspective model update from matched samples ----------
    def _update_pc_model_with_det(self, det_row, img_h):
        score = float(det_row[4])
        if score < self.pc_sample_score:
            return
        y2 = float(det_row[3])
        h = float(det_row[3] - det_row[1])
        if h > 0 and np.isfinite(h) and np.isfinite(y2):
            self.perspective.observe(y2, h, img_h)

    # ---------- Optional ROCM rerank for round-3 only ----------
    def _rocm_round3_cost(self, dets_xyxy, prev_obs, velocities_lt, velocities_rt, velocities_lb, velocities_rb):
        if dets_xyxy.shape[0] == 0 or prev_obs.shape[0] == 0:
            return np.zeros((dets_xyxy.shape[0], prev_obs.shape[0]), dtype=float)
        Y1, X1 = speed_direction_batch_lt(dets_xyxy, prev_obs)
        Y2, X2 = speed_direction_batch_rt(dets_xyxy, prev_obs)
        Y3, X3 = speed_direction_batch_lb(dets_xyxy, prev_obs)
        Y4, X4 = speed_direction_batch_rb(dets_xyxy, prev_obs)
        cost_lt = cost_vel(Y1, X1, prev_obs[:, :4], velocities_lt, np.concatenate([dets_xyxy, np.ones((dets_xyxy.shape[0], 1))], axis=1), prev_obs, self.inertia)
        cost_rt = cost_vel(Y2, X2, prev_obs[:, :4], velocities_rt, np.concatenate([dets_xyxy, np.ones((dets_xyxy.shape[0], 1))], axis=1), prev_obs, self.inertia)
        cost_lb = cost_vel(Y3, X3, prev_obs[:, :4], velocities_lb, np.concatenate([dets_xyxy, np.ones((dets_xyxy.shape[0], 1))], axis=1), prev_obs, self.inertia)
        cost_rb = cost_vel(Y4, X4, prev_obs[:, :4], velocities_rb, np.concatenate([dets_xyxy, np.ones((dets_xyxy.shape[0], 1))], axis=1), prev_obs, self.inertia)
        return cost_lt + cost_rt + cost_lb + cost_rb

    # ---------- First round associate: IoU * A_u * A_p (UNCHANGED backbone) ----------
    # def _associate_round1(
    #         self,
    #         dets, trks,
    #         height_mus, height_sigmas, miss_counts,
    #         img_w, img_h,
    #         iou_thr,
    #         depth_emas_full=None
    #     ):
    
    # def _associate_round1(
    #     self,
    #     dets, trks,
    #     height_mus, height_sigmas, miss_counts,
    #     img_w, img_h,
    #     iou_thr,
    #     depth_emas_full=None,
    #     trk_score_proxy_full=None
    #     ):
    
    def _associate_round1(
        self,
        dets, trks,
        height_mus, height_sigmas, miss_counts,
        img_w, img_h,
        iou_thr,
        depth_emas_full=None,
        trk_kalman_score_full=None
    ):
        
        
    
            Nd = dets.shape[0]
            Nt = trks.shape[0]
            if Nd == 0 or Nt == 0:
                return np.empty((0, 2), dtype=int), np.arange(Nd), np.arange(Nt)

            dets_xy = dets[:, :4]
            trks_xy = trks[:, :4]
            det_sc = dets[:, 4]

            iou = self.asso_func(dets_xy, trks_xy)
            iou = np.array(iou)

            if self.ua_enable:
                A_u = self._ua_height_affinity(
                    dets_xy, trks_xy,
                    height_mus, height_sigmas,
                    det_sc, miss_counts,
                    img_w, img_h
                )
            else:
                A_u = np.ones_like(iou)

            A_p = self._pc_affinity(dets_xy, trks_xy, img_h)

            # S = iou * A_u * A_p
            # if (depth_emas_full is not None) and self.depth_enable and (self.depth_alpha > 0):
            #     A_d_r1 = self._depth_affinity(dets_xy, trks_xy, np.asarray(depth_emas_full), img_h)
            #     S = self._blend_iou_with_depth(S, A_d_r1)

            # matched = []
            # if S.size > 0 and iou.max() > iou_thr:
            #     idxs = linear_assignment(-S)
            #     for d_i, t_i in idxs:
            #         if iou[d_i, t_i] >= iou_thr:
            #             matched.append([d_i, t_i])
            
            
            S = iou * A_u * A_p
            if (depth_emas_full is not None) and self.depth_enable and (self.depth_alpha > 0):
                A_d_r1 = self._depth_affinity(dets_xy, trks_xy, np.asarray(depth_emas_full), img_h)
                S = self._blend_iou_with_depth(S, A_d_r1)
                
            score_r1 = S.copy()

            # paper-style TCM for round1:
            # use Kalman-predicted confidence for all high-confidence pairs
            if self.tcm_round1_enable and score_r1.size > 0 and trk_kalman_score_full is not None:
                tcm_penalty = cal_score_dif_batch_two_score(dets, trk_kalman_score_full)
                score_r1 = score_r1 - self.tcm_round1_weight * tcm_penalty

            # very conservative TCM-on-round1:
            # only re-rank ambiguous pairs, never change the raw IoU acceptance gate
            score_r1 = S.copy()
            if (
                self.tcm_round1_enable
                and score_r1.size > 0
                and trk_kalman_score_full is not None
                and len(trk_kalman_score_full) == trks.shape[0]
            ):
                tcm_penalty = cal_score_dif_batch_two_score(dets, trk_kalman_score_full)  # (Nd, Nt)

                # 如果你现在想尽量贴近论文里的 round1 TCM，这里先不要做 ambiguous gating
                score_r1 = score_r1 - self.tcm_round1_weight * tcm_penalty

            matched = []
            if score_r1.size > 0 and iou.max() > iou_thr:
                idxs = linear_assignment(-score_r1)
                for d_i, t_i in idxs:
                    # keep original raw-IoU gate
                    if iou[d_i, t_i] >= iou_thr:
                        matched.append([d_i, t_i])
            
                        
                        
            matched = np.asarray(matched, dtype=int) if len(matched) > 0 else np.empty((0, 2), dtype=int)

            matched_d = matched[:, 0] if matched.size > 0 else np.array([], dtype=int)
            matched_t = matched[:, 1] if matched.size > 0 else np.array([], dtype=int)
            unmatched_dets = np.setdiff1d(np.arange(Nd), matched_d, assume_unique=False)
            unmatched_trks = np.setdiff1d(np.arange(Nt), matched_t, assume_unique=False)

            return matched, unmatched_dets, unmatched_trks

    def update(self, output_results, img_info, img_size, frame_id=None):
        self._maybe_reset_on_boundary(frame_id=frame_id,
                                      img_size=(img_info[0], img_info[1]),
                                      reset_on_size_change=True,
                                      verbose=True)

        if output_results is None:
            return np.empty((0, 5))

        self.frame_count += 1

        try:
            import torch
            if isinstance(output_results, torch.Tensor):
                output_results = output_results.detach().cpu().numpy()
            elif isinstance(output_results, (list, tuple)) and len(output_results) > 0:
                first = output_results[0]
                if isinstance(first, torch.Tensor):
                    output_results = first.detach().cpu().numpy()
                else:
                    output_results = np.array(first) if not isinstance(first, np.ndarray) else first
        except Exception:
            pass
        if not isinstance(output_results, np.ndarray):
            output_results = np.array(output_results)

        img_h, img_w = _to_scalar(img_info[0]), _to_scalar(img_info[1])

        if output_results.size == 0:
            trks = np.zeros((len(self.trackers), 5))
            to_del = []
            ret = []
            for t, trk in enumerate(trks):
                pos = self.trackers[t].predict()[0]
                trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
                if np.any(np.isnan(pos)):
                    to_del.append(t)
            trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
            for t in reversed(to_del):
                self.trackers.pop(t)

            i = len(self.trackers)
            for trk in reversed(self.trackers):
                if trk.last_observation.sum() < 0:
                    d = trk.get_state()[0]
                else:
                    d = trk.last_observation[:4]
                if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                    ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
                i -= 1
                if trk.time_since_update > self.max_age:
                    self.trackers.pop(i)
            if len(ret) > 0:
                return np.concatenate(ret)
            return np.empty((0, 5))

        ncol = output_results.shape[1]
        if ncol >= 6:
            scores = output_results[:, 4].astype(float) * output_results[:, 5].astype(float)
            bboxes = output_results[:, :4].astype(float)
        elif ncol == 5:
            scores = output_results[:, 4].astype(float)
            bboxes = output_results[:, :4].astype(float)
        else:
            raise ValueError(f"Unexpected detection shape {output_results.shape}, expected 5 or >=6 columns.")

        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes = bboxes / max(scale, 1e-6)
        dets_all = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)

        inds_low = scores > 0.1
        inds_high = scores < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = dets_all[inds_second]
        remain_inds = scores > self.det_thresh
        dets = dets_all[remain_inds]

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        depth_emas = np.array([trk.depth_ema for trk in self.trackers], dtype=float)
        height_mus_full = np.array([trk.get_height_mu_sigma()[0] for trk in self.trackers], dtype=float)
        height_sigmas_full = np.array([trk.get_height_mu_sigma()[1] for trk in self.trackers], dtype=float)
        miss_counts_full = np.array([trk.time_since_update for trk in self.trackers], dtype=float)
        # track_score_proxy_full = np.array([trk.get_tcm_score_proxy(low=0.1, high=1.0) for trk in self.trackers], dtype=float)
        
        track_kalman_score_full = np.array(
            [trk.get_tcm_kalman_score(low=self.det_thresh, high=1.0) for trk in self.trackers],
            dtype=float
        )

        track_linear_score_full = np.array(
            [trk.get_tcm_linear_score(low=0.1, high=self.det_thresh) for trk in self.trackers],
            dtype=float
        )
        
        
        ####打印是否 高置信阶段 KF 更合适，低置信/遮挡阶段 Linear 更合适。
        # if len(track_kalman_score_full) > 0 and (self.frame_count % 30 == 0):
        #     print(
        #         f"[TCM-R1] frame={self.frame_count} "
        #         f"kalman_score min/max="
        #         f"{track_kalman_score_full.min():.4f}/{track_kalman_score_full.max():.4f}"
        #     )
            
        # if len(track_linear_score_full) > 0 and (self.frame_count % 30 == 0):
        #     print(
        #         f"[TCM-R2] frame={self.frame_count} "
        #         f"linear_score min/max="
        #         f"{track_linear_score_full.min():.4f}/{track_linear_score_full.max():.4f}"
        #     )
        
        
        

        # Optional ROCM state for future round-3 rerank
        velocities_lt_full = np.array([trk.velocity_lt if trk.velocity_lt is not None else np.array((0, 0)) for trk in self.trackers]) if len(self.trackers) > 0 else np.zeros((0, 2))
        velocities_rt_full = np.array([trk.velocity_rt if trk.velocity_rt is not None else np.array((0, 0)) for trk in self.trackers]) if len(self.trackers) > 0 else np.zeros((0, 2))
        velocities_lb_full = np.array([trk.velocity_lb if trk.velocity_lb is not None else np.array((0, 0)) for trk in self.trackers]) if len(self.trackers) > 0 else np.zeros((0, 2))
        velocities_rb_full = np.array([trk.velocity_rb if trk.velocity_rb is not None else np.array((0, 0)) for trk in self.trackers]) if len(self.trackers) > 0 else np.zeros((0, 2))

        # ---------- First round: KEEP ORIGINAL BACKBONE UNCHANGED ----------
        if dets.shape[0] > 0 and trks.shape[0] > 0:
            # matched, unmatched_dets, unmatched_trks = self._associate_round1(
            #     dets, trks,
            #     height_mus_full, height_sigmas_full, miss_counts_full,
            #     img_w, img_h,
            #     self.iou_threshold,
            #     depth_emas_full=depth_emas
            # )
            
            # matched, unmatched_dets, unmatched_trks = self._associate_round1(
            #     dets, trks,
            #     height_mus_full, height_sigmas_full, miss_counts_full,
            #     img_w, img_h,
            #     self.iou_threshold,
            #     depth_emas_full=depth_emas,
            #     trk_score_proxy_full=track_score_proxy_full
            # )
            
            matched, unmatched_dets, unmatched_trks = self._associate_round1(
                dets, trks,
                height_mus_full, height_sigmas_full, miss_counts_full,
                img_w, img_h,
                self.iou_threshold,
                depth_emas_full=depth_emas,
                trk_kalman_score_full=track_kalman_score_full
            )
            
            
            

            
            for d_i, t_i in matched:
                self.trackers[t_i].update(dets[d_i, :])
                self._update_pc_model_with_det(dets[d_i, :], img_h)
        else:
            matched = np.empty((0, 2), dtype=int)
            unmatched_dets = np.arange(dets.shape[0])
            unmatched_trks = np.arange(trks.shape[0])

        # ---------- Second round: ORIGINAL BYTE + optional TCM ----------
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            iou_left = self.asso_func(dets_second[:, :4], u_trks[:, :4])
            iou_left = np.array(iou_left)

            if iou_left.size > 0:
                if self.ua_enable:
                    u_mu = height_mus_full[unmatched_trks]
                    u_sigma = height_sigmas_full[unmatched_trks]
                    u_miss = miss_counts_full[unmatched_trks]
                    det_sc = dets_second[:, 4]
                    A_u = self._ua_height_affinity(
                        dets_second[:, :4], u_trks[:, :4],
                        u_mu, u_sigma,
                        det_sc, u_miss,
                        img_w, img_h
                    )
                else:
                    A_u = np.ones_like(iou_left)

                A_p = self._pc_affinity(dets_second[:, :4], u_trks[:, :4], img_h)
                iou_geom = iou_left * A_u * A_p
            else:
                iou_geom = iou_left

            if iou_geom.size > 0:
                u_trk_emas = depth_emas[unmatched_trks]
                A_d = self._depth_affinity(dets_second[:, :4], u_trks[:, :4], u_trk_emas, img_h) if self.depth_enable else np.ones_like(iou_geom)
                iou_adj = self._blend_iou_with_depth(iou_geom, A_d)
            else:
                iou_adj = iou_geom

            # Hybrid-style TCM port: only re-rank ambiguous low-score matching, keep raw IoU gate.
            score_left = iou_adj.copy()
            # if self.tcm_round2_enable and score_left.size > 0:
            #     u_score_proxy = track_score_proxy_full[unmatched_trks]
            #     tcm_penalty = cal_score_dif_batch_two_score(dets_second, u_score_proxy)
            #     score_left = score_left - self.tcm_round2_weight * tcm_penalty
            
            if self.tcm_round2_enable and score_left.size > 0:
                u_linear_score = track_linear_score_full[unmatched_trks]
                tcm_penalty = cal_score_dif_batch_two_score(dets_second, u_linear_score)
                score_left = score_left - self.tcm_round2_weight * tcm_penalty            
            

            thr2 = max(0.1, self.iou_threshold - 0.07)
            if score_left.size > 0 and score_left.max() > thr2:
                matched_indices = linear_assignment(-score_left)
                to_remove_trk_indices = []
                to_remove_det_indices = []
                for m2 in matched_indices:
                    det_local, trk_local = m2[0], m2[1]
                    # acceptance still uses raw geometric IoU, not TCM-adjusted score
                    if iou_left[det_local, trk_local] < thr2:
                        continue
                    trk_ind = unmatched_trks[trk_local]
                    self.trackers[trk_ind].update(dets_second[det_local, :])
                    to_remove_trk_indices.append(trk_ind)
                    to_remove_det_indices.append(det_local)
                    self._update_pc_model_with_det(dets_second[det_local, :], img_h)
                if len(to_remove_trk_indices) > 0:
                    unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))
                if len(to_remove_det_indices) > 0:
                    keep_mask = np.ones(len(dets_second), dtype=bool)
                    keep_mask[to_remove_det_indices] = False
                    dets_second = dets_second[keep_mask]

        # ---------- Third round: ORIGINAL RE-ASSOC + optional TCM (+ optional ROCM hook) ----------
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks_last = last_boxes[unmatched_trks]
            left_trks_pred = trks[unmatched_trks]

            iou_last = self.asso_func(left_dets[:, :4], left_trks_last[:, :4])
            iou_pred = self.asso_func(left_dets[:, :4], left_trks_pred[:, :4])
            iou_last = np.array(iou_last)
            iou_pred = np.array(iou_pred)
            iou_max = iou_last if iou_pred.size == 0 else np.maximum(iou_last, iou_pred)

            if iou_max.size > 0:
                dets_xy = left_dets[:, :4]
                trks_xy = left_trks_pred[:, :4]

                if self.ua_enable:
                    l_mu = height_mus_full[unmatched_trks]
                    l_sigma = height_sigmas_full[unmatched_trks]
                    l_miss = miss_counts_full[unmatched_trks]
                    det_sc = left_dets[:, 4]
                    A_u = self._ua_height_affinity(
                        dets_xy, trks_xy,
                        l_mu, l_sigma,
                        det_sc, l_miss,
                        img_w, img_h
                    )
                else:
                    A_u = np.ones_like(iou_max)

                A_p = self._pc_affinity(dets_xy, trks_xy, img_h)
                iou_geom = iou_max * A_u * A_p

                trk_emas = depth_emas[unmatched_trks]
                A_d = self._depth_affinity(dets_xy, trks_xy, trk_emas, img_h) if self.depth_enable else np.ones_like(iou_geom)
                iou_adj = self._blend_iou_with_depth(iou_geom, A_d)
            else:
                iou_adj = iou_max

            score_rematch = iou_adj.copy()
            if self.tcm_round3_enable and score_rematch.size > 0:
                u_score_proxy = track_score_proxy_full[unmatched_trks]
                tcm_penalty = cal_score_dif_batch_two_score(left_dets, u_score_proxy)
                score_rematch = score_rematch - self.tcm_round3_weight * tcm_penalty

            if self.rocm_round3_enable and score_rematch.size > 0:
                rocm_cost = self._rocm_round3_cost(
                    left_dets[:, :4],
                    left_trks_last,
                    velocities_lt_full[unmatched_trks],
                    velocities_rt_full[unmatched_trks],
                    velocities_lb_full[unmatched_trks],
                    velocities_rb_full[unmatched_trks],
                )
                score_rematch = score_rematch + self.rocm_weight * rocm_cost

            thr3 = max(0.1, self.iou_threshold - 0.07)
            if score_rematch.size > 0 and score_rematch.max() > thr3:
                rematched_indices = linear_assignment(-score_rematch)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m3 in rematched_indices:
                    det_local, trk_local = m3[0], m3[1]
                    if iou_max[det_local, trk_local] < thr3:
                        continue
                    det_ind = unmatched_dets[det_local]
                    trk_ind = unmatched_trks[trk_local]
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                    self._update_pc_model_with_det(dets[det_ind, :], img_h)
                if len(to_remove_det_indices) > 0:
                    unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                if len(to_remove_trk_indices) > 0:
                    unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.trackers[m].update(None)

        for i in unmatched_dets:
            trk = KalmanBoxTracker(
                dets[i, :],
                delta_t=self.delta_t,
                ema_obs=self.ema_obs,
                ema_pred=self.ema_pred,
                ua_h_ema=self.ua_h_ema,
                ua_h2_ema=self.ua_h2_ema,
                ua_proc_var_frac=self.ua_proc_var_frac,
                ua_sigma_floor_frac=self.ua_sigma_floor_frac,
                score_ema=self.score_ema,
            )
            self.trackers.append(trk)
            self._update_pc_model_with_det(dets[i, :], img_h)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def update_public(self, dets, cates, scores):
        # Keep baseline behavior for public detections.
        self.frame_count += 1

        det_scores = np.ones((dets.shape[0], 1))
        dets = np.concatenate((dets, det_scores), axis=1)

        remain_inds = scores > self.det_thresh
        cates = cates[remain_inds]
        dets = dets[remain_inds]

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            cat = getattr(self.trackers[t], "cate", 1)
            trk[:] = [pos[0], pos[1], pos[2], pos[3], cat]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        matched, unmatched_dets, unmatched_trks = associate_kitti(
            dets, trks, cates, self.iou_threshold, velocities, k_observations, self.inertia
        )
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
            self._update_pc_model_with_det(dets[m[0], :], img_h=1.0)

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_dets_c = left_dets.copy()
            left_trks_c = left_trks.copy()

            iou_left = iou_batch(left_dets_c, left_trks_c)
            iou_left = np.array(iou_left)
            det_cates_left = cates[unmatched_dets]
            trk_cates_left = trks[unmatched_trks][:, 4]
            num_dets = unmatched_dets.shape[0]
            num_trks = unmatched_trks.shape[0]
            cate_matrix = np.zeros((num_dets, num_trks))
            for i in range(num_dets):
                for j in range(num_trks):
                    if det_cates_left[i] != trk_cates_left[j]:
                        cate_matrix[i][j] = -1e6
            iou_left = iou_left + cate_matrix
            if iou_left.max() > self.iou_threshold - 0.1:
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold - 0.1:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                    self._update_pc_model_with_det(dets[det_ind, :], img_h=1.0)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for i in unmatched_dets:
            trk = KalmanBoxTracker(
                dets[i, :],
                ema_obs=self.ema_obs,
                ema_pred=self.ema_pred,
                ua_h_ema=self.ua_h_ema,
                ua_h2_ema=self.ua_h2_ema,
                ua_proc_var_frac=self.ua_proc_var_frac,
                ua_sigma_floor_frac=self.ua_sigma_floor_frac,
                score_ema=self.score_ema,
            )
            trk.cate = cates[i]
            self.trackers.append(trk)
            self._update_pc_model_with_det(dets[i, :], img_h=1.0)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() > 0:
                d = trk.last_observation[:4]
            else:
                d = trk.get_state()[0]
            if trk.time_since_update < 1:
                if (self.frame_count <= self.min_hits) or (trk.hit_streak >= self.min_hits):
                    ret.append(np.concatenate((d, [trk.id + 1], [getattr(trk, "cate", 1)], [0])).reshape(1, -1))
                if trk.hit_streak == self.min_hits:
                    for prev_i in range(self.min_hits - 1):
                        prev_observation = trk.history_observations[-(prev_i + 2)]
                        ret.append((np.concatenate((prev_observation[:4], [trk.id + 1], [getattr(trk, "cate", 1)],
                                                    [-(prev_i + 1)]))).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 7))
