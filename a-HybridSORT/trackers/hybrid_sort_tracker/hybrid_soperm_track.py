# """
#     This script is adopted from the SORT script by Alex Bewley alex@bewley.ai

#     Hybrid-SORT + optional geometric priors:
#     - UA-HMIoU  : uncertainty-aware height affinity
#     - PC-HMIoU  : perspective-consistent height affinity
#     - Soft-Depth: soft depth continuity regularization by y2 EMA

#     Design goal:
#     Keep Hybrid-SORT's original strengths (4-point motion consistency,
#     confidence-state KF, TCM, BYTE, rematching), and only use the above three
#     components as additional pairwise geometric priors.

#     Important implementation choices:
#     1) First-round association remains Hybrid-SORT style.
#     2) All three components are independent toggles.
#     3) Acceptance gates still use the ORIGINAL geometric IoU-family score,
#        while the augmented score is only used for ranking / Hungarian matching.
#     4) No changes are required in association.py.
# """
# from __future__ import print_function

# import numpy as np
# from .association import *


# # ---------------------------------------------------------------------------
# # Safe conversions
# # ---------------------------------------------------------------------------

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


# # ---------------------------------------------------------------------------
# # Original helpers
# # ---------------------------------------------------------------------------

# def k_previous_obs(observations, cur_age, k):
#     if len(observations) == 0:
#         return [-1, -1, -1, -1, -1]
#     for i in range(k):
#         dt = k - i
#         if cur_age - dt in observations:
#             return observations[cur_age-dt]
#     max_age = max(observations.keys())
#     return observations[max_age]


# # NOTE: Hybrid-SORT uses [x, y, s, score, r] in the measurement/state front.
# def convert_bbox_to_z(bbox):
#     """
#     Takes a bounding box in the form [x1,y1,x2,y2,score] and returns z in the
#     form [x,y,s,score,r] where x,y is the centre of the box and s is the
#     scale/area and r is the aspect ratio.
#     """
#     w = bbox[2] - bbox[0]
#     h = bbox[3] - bbox[1]
#     x = bbox[0] + w / 2.0
#     y = bbox[1] + h / 2.0
#     s = w * h
#     r = w / float(h + 1e-6)
#     score = bbox[4]
#     return np.array([x, y, s, score, r]).reshape((5, 1))


# def convert_x_to_bbox(x, score=None):
#     """
#     Takes a bounding box in the centre form [x,y,s,score,r] and returns it in
#     the form [x1,y1,x2,y2] or [x1,y1,x2,y2,score].
#     """
#     w = np.sqrt(max(float(x[2] * x[4]), 0.0))
#     if w < 1e-12:
#         h = 0.0
#     else:
#         h = x[2] / w
#     state_score = x[3]
#     if score is None:
#         score = state_score
#     if score is None:
#         return np.array([x[0] - w / 2.0, x[1] - h / 2.0,
#                          x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
#     return np.array([x[0] - w / 2.0, x[1] - h / 2.0,
#                      x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


# def speed_direction(bbox1, bbox2):
#     cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
#     cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
#     speed = np.array([cy2 - cy1, cx2 - cx1])
#     norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
#     return speed / norm


# def speed_direction_lt(bbox1, bbox2):
#     cx1, cy1 = bbox1[0], bbox1[1]
#     cx2, cy2 = bbox2[0], bbox2[1]
#     speed = np.array([cy2 - cy1, cx2 - cx1])
#     norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
#     return speed / norm


# def speed_direction_rt(bbox1, bbox2):
#     cx1, cy1 = bbox1[0], bbox1[3]
#     cx2, cy2 = bbox2[0], bbox2[3]
#     speed = np.array([cy2 - cy1, cx2 - cx1])
#     norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
#     return speed / norm


# def speed_direction_lb(bbox1, bbox2):
#     cx1, cy1 = bbox1[2], bbox1[1]
#     cx2, cy2 = bbox2[2], bbox2[1]
#     speed = np.array([cy2 - cy1, cx2 - cx1])
#     norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
#     return speed / norm


# def speed_direction_rb(bbox1, bbox2):
#     cx1, cy1 = bbox1[2], bbox1[3]
#     cx2, cy2 = bbox2[2], bbox2[3]
#     speed = np.array([cy2 - cy1, cx2 - cx1])
#     norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
#     return speed / norm


# # ---------------------------------------------------------------------------
# # Perspective model (scene-level prior)
# # ---------------------------------------------------------------------------

# class PerspectiveHeightModel(object):
#     """
#     Online mapping h*(y2): bottom y2 -> expected pixel height.
#     Implemented via normalized y2 binning + EMA; supports interpolation and
#     fallback to global EMA during warmup.
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
#         if (not np.isfinite(y2_norm)) or (not np.isfinite(h_pixel)) or h_pixel <= 0:
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
#         y2_pixels = np.asarray(y2_pixels, dtype=float)
#         if img_h <= 0 or len(y2_pixels) == 0:
#             return np.zeros_like(y2_pixels, dtype=float)

#         y2n = np.clip(y2_pixels / float(img_h), 0.0, 1.0 - 1e-8)
#         bins = np.floor(y2n * self.num_bins).astype(int)
#         pos = y2n * self.num_bins
#         frac = pos - np.floor(pos)
#         b0 = bins
#         b1 = np.clip(bins + 1, 0, self.num_bins - 1)

#         m0, m1 = self.bin_means[b0], self.bin_means[b1]
#         c0, c1 = self.bin_counts[b0], self.bin_counts[b1]
#         use0 = c0 >= self.min_count
#         use1 = c1 >= self.min_count

#         m0_eff = np.where(use0, m0, np.where(use1, m1, self.global_ema))
#         m1_eff = np.where(use1, m1, np.where(use0, m0, self.global_ema))
#         h_star = (1.0 - frac) * m0_eff + frac * m1_eff
#         return np.maximum(h_star, 4.0)

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
#         if self.floor > 0.0:
#             A_p = np.maximum(A_p, self.floor)
#         return A_p

#     def ready(self):
#         return self.global_count >= 50


# # ---------------------------------------------------------------------------
# # Kalman tracker
# # ---------------------------------------------------------------------------

# class KalmanBoxTracker(object):
#     """
#     Hybrid-SORT Kalman tracker + optional side statistics for
#     depth regularization and uncertainty-aware height affinity.
#     """
#     count = 0

#     def __init__(self,
#                  bbox,
#                  delta_t=3,
#                  orig=False,
#                  args=None,
#                  depth_enable=True,
#                  ema_obs=0.5,
#                  ema_pred=0.1,
#                  ua_enable=True,
#                  ua_h_ema=0.5,
#                  ua_h2_ema=0.3,
#                  ua_proc_var_frac=0.02,
#                  ua_sigma_floor_frac=0.05):
#         if not orig:
#             from .kalmanfilter_score_new import KalmanFilterNew_score_new as KalmanFilter_score_new
#             self.kf = KalmanFilter_score_new(dim_x=9, dim_z=5)
#         else:
#             from filterpy.kalman import KalmanFilter
#             self.kf = KalmanFilter(dim_x=7, dim_z=4)

#         # u, v, s, c, r, ~u, ~v, ~s, ~c
#         self.kf.F = np.array([
#             [1, 0, 0, 0, 0, 1, 0, 0, 0],
#             [0, 1, 0, 0, 0, 0, 1, 0, 0],
#             [0, 0, 1, 0, 0, 0, 0, 1, 0],
#             [0, 0, 0, 1, 0, 0, 0, 0, 1],
#             [0, 0, 0, 0, 1, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 1, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 1, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 1, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 1],
#         ])
#         self.kf.H = np.array([
#             [1, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 1, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 1, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 1, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 1, 0, 0, 0, 0],
#         ])

#         self.kf.R[2:, 2:] *= 10.0
#         self.kf.P[5:, 5:] *= 1000.0
#         self.kf.P *= 10.0
#         self.kf.Q[-1, -1] *= 0.01
#         self.kf.Q[-2, -2] *= 0.01
#         self.kf.Q[5:, 5:] *= 0.01
#         self.kf.x[:5] = convert_bbox_to_z(bbox)

#         self.time_since_update = 0
#         self.id = KalmanBoxTracker.count
#         KalmanBoxTracker.count += 1
#         self.history = []
#         self.hits = 0
#         self.hit_streak = 0
#         self.age = 0
#         self.age_recover_for_cbiou = 0

#         self.last_observation = np.array([-1, -1, -1, -1, -1])
#         self.last_observation_save = np.array([-1, -1, -1, -1, -1])
#         self.observations = dict()
#         self.history_observations = []

#         self.velocity = None
#         self.velocity_lt = None
#         self.velocity_rt = None
#         self.velocity_lb = None
#         self.velocity_rb = None
#         self.delta_t = delta_t
#         self.confidence_pre = None
#         self.confidence = bbox[-1]
#         self.args = args
#         self.kf.args = args

#         # Optional side statistics
#         self.depth_enable = bool(depth_enable)
#         self.ema_obs = float(ema_obs)
#         self.ema_pred = float(ema_pred)
#         self.depth_ema = float(bbox[3])

#         self.ua_enable = bool(ua_enable)
#         self.ua_h_ema = float(ua_h_ema)
#         self.ua_h2_ema = float(ua_h2_ema)
#         self.ua_proc_var_frac = float(ua_proc_var_frac)
#         self.ua_sigma_floor_frac = float(ua_sigma_floor_frac)
#         h0 = float(max(bbox[3] - bbox[1], 1e-6))
#         self.h_m1 = h0
#         self.h_m2 = h0 * h0
#         floor_var = (self.ua_sigma_floor_frac * h0) ** 2
#         self.h_var = max(self.h_m2 - self.h_m1 * self.h_m1, floor_var)

#     def get_height_mu_sigma(self):
#         mu = float(self.h_m1)
#         floor_var = (self.ua_sigma_floor_frac * max(mu, 1e-6)) ** 2
#         var = max(float(self.h_var), floor_var)
#         sigma = np.sqrt(var)
#         return mu, sigma

#     def _inflate_height_uncertainty(self):
#         if not self.ua_enable:
#             return
#         mu = max(float(self.h_m1), 1e-6)
#         self.h_var = float(self.h_var) + (self.ua_proc_var_frac * mu) ** 2

#     def _update_depth_and_height_stats(self, bbox):
#         if bbox is None:
#             return
#         if self.depth_enable:
#             y2 = float(bbox[3])
#             self.depth_ema = (1.0 - self.ema_obs) * self.depth_ema + self.ema_obs * y2
#         if self.ua_enable:
#             h = float(max(bbox[3] - bbox[1], 1e-6))
#             self.h_m1 = (1.0 - self.ua_h_ema) * self.h_m1 + self.ua_h_ema * h
#             self.h_m2 = (1.0 - self.ua_h2_ema) * self.h_m2 + self.ua_h2_ema * (h * h)
#             raw_var = max(self.h_m2 - self.h_m1 * self.h_m1, 0.0)
#             floor_var = (self.ua_sigma_floor_frac * max(self.h_m1, 1e-6)) ** 2
#             self.h_var = 0.5 * self.h_var + 0.5 * max(raw_var, floor_var)

#     def update(self, bbox):
#         velocity = None
#         velocity_lt = None
#         velocity_rt = None
#         velocity_lb = None
#         velocity_rb = None

#         if bbox is not None:
#             if self.last_observation.sum() >= 0:
#                 previous_box = None
#                 for i in range(self.delta_t):
#                     if self.age - i - 1 in self.observations:
#                         previous_box = self.observations[self.age - i - 1]
#                         if velocity_lt is not None:
#                             velocity += speed_direction(previous_box, bbox)
#                             velocity_lt += speed_direction_lt(previous_box, bbox)
#                             velocity_rt += speed_direction_rt(previous_box, bbox)
#                             velocity_lb += speed_direction_lb(previous_box, bbox)
#                             velocity_rb += speed_direction_rb(previous_box, bbox)
#                         else:
#                             velocity = speed_direction(previous_box, bbox)
#                             velocity_lt = speed_direction_lt(previous_box, bbox)
#                             velocity_rt = speed_direction_rt(previous_box, bbox)
#                             velocity_lb = speed_direction_lb(previous_box, bbox)
#                             velocity_rb = speed_direction_rb(previous_box, bbox)
#                 if previous_box is None:
#                     previous_box = self.last_observation
#                     self.velocity = speed_direction(previous_box, bbox)
#                     self.velocity_lt = speed_direction_lt(previous_box, bbox)
#                     self.velocity_rt = speed_direction_rt(previous_box, bbox)
#                     self.velocity_lb = speed_direction_lb(previous_box, bbox)
#                     self.velocity_rb = speed_direction_rb(previous_box, bbox)
#                 else:
#                     self.velocity = velocity
#                     self.velocity_lt = velocity_lt
#                     self.velocity_rt = velocity_rt
#                     self.velocity_lb = velocity_lb
#                     self.velocity_rb = velocity_rb

#             self.last_observation = bbox
#             self.last_observation_save = bbox
#             self.observations[self.age] = bbox
#             self.history_observations.append(bbox)

#             self.time_since_update = 0
#             self.history = []
#             self.hits += 1
#             self.hit_streak += 1
#             self.kf.update(convert_bbox_to_z(bbox))
#             self.confidence_pre = self.confidence
#             self.confidence = bbox[-1]
#             self.age_recover_for_cbiou = self.age

#             self._update_depth_and_height_stats(bbox)
#         else:
#             self.kf.update(bbox)
#             self.confidence_pre = None

#     def predict(self):
#         if (self.kf.x[7] + self.kf.x[2]) <= 0:
#             self.kf.x[7] *= 0.0

#         self.kf.predict()
#         self.age += 1
#         if self.time_since_update > 0:
#             self.hit_streak = 0
#         self.time_since_update += 1

#         pred = convert_x_to_bbox(self.kf.x)
#         self.history.append(pred)

#         if self.depth_enable:
#             pred_y2 = float(pred[0][3])
#             self.depth_ema = (1.0 - self.ema_pred) * self.depth_ema + self.ema_pred * pred_y2

#         # Inflate UA variance only once per frame in predict().
#         self._inflate_height_uncertainty()

#         # if not self.confidence_pre:
#         #     return self.history[-1], np.clip(self.kf.x[3], self.args.track_thresh, 1.0), np.clip(self.confidence, 0.1, self.args.track_thresh)
#         # return self.history[-1], np.clip(self.kf.x[3], self.args.track_thresh, 1.0), np.clip(self.confidence - (self.confidence_pre - self.confidence), 0.1, self.args.track_thresh)
        
#         kalman_score = float(np.clip(_to_scalar(self.kf.x[3]), self.args.track_thresh, 1.0))

#         if self.confidence_pre is None:
#             simple_raw = self.confidence
#         else:
#             simple_raw = self.confidence - (self.confidence_pre - self.confidence)

#         simple_score = float(np.clip(_to_scalar(simple_raw), 0.1, self.args.track_thresh))

#         return self.history[-1], kalman_score, simple_score

#     def get_state(self):
#         return convert_x_to_bbox(self.kf.x)


# """
#     We support multiple ways for association cost calculation, by default
#     we use IoU. GIoU may have better performance in some situations. We note
#     that we hardly normalize the cost by all methods to (0,1) which may not be
#     the best practice.
# """
# ASSO_FUNCS = {
#     "iou": iou_batch,
#     "giou": giou_batch,
#     "ciou": ciou_batch,
#     "diou": diou_batch,
#     "ct_dist": ct_dist,
#     "Height_Modulated_IoU": hmiou,
# }


# class Hybrid_Soperm_Track(object):
#     def __init__(self,
#                  args,
#                  det_thresh,
#                  max_age=30,
#                  min_hits=3,
#                  iou_threshold=0.3,
#                  delta_t=3,
#                  asso_func="iou",
#                  inertia=0.2,
#                  use_byte=False,
#                  # component switches
#                  ua_enable=True,
#                  pc_enable=True,
#                  depth_enable=True,
#                  # soft-depth params
#                  depth_alpha=0.5,
#                  depth_beta=2.0,
#                  depth_gate=0.25,
#                  gate_floor=0.2,
#                  # UA-HMIoU params
#                  ua_mode="cauchy",
#                  ua_alpha=1.0,
#                  ua_border_margin_frac=0.02,
#                  ua_border_boost=0.6,
#                  ua_miss_boost=0.6,
#                  ua_low_score=0.4,
#                  ua_score_boost=0.3,
#                  ua_sigma_floor_frac=0.05,
#                  ua_proc_var_frac=0.02,
#                  ua_h_ema=0.5,
#                  ua_h2_ema=0.3,
#                  # PC-HMIoU params
#                  pc_bins=24,
#                  pc_ema=0.08,
#                  pc_gamma=2.0,
#                  pc_floor=0.3,
#                  pc_min_count=2,
#                  pc_sample_score=0.6,
#                  pc_update_matched_only=True,
#                  # depth EMA
#                  ema_obs=0.5,
#                  ema_pred=0.1,
                 
#                  prior_w_u=0.60,
#                  prior_w_p=0.50,
#                  prior_w_d=0.40,
#                  prior_floor=0.55,
#                  r1_amb_delta=0.05,
#                  r1_lambda_u=0.08,
#                  r1_lambda_d=0.06,
#                  r2_lambda_u=0.18,
#                  r2_lambda_d=0.14,
#                  r3_lambda_u=0.20,
#                  r3_lambda_d=0.16,
#                  r2_amb_delta=0.03,
#                  r3_amb_delta=0.05,
#                  r2_miss_trigger=2.0,
#                  r3_miss_trigger=2.0,
#                  r1_repair_enable=True,
#                  r1_protect_iou=0.55,
#                  r1_protect_det_margin=0.10,
#                  r1_protect_trk_margin=0.08,
#                  r1_protect_score=0.50,
#                  r1_protect_miss=1.0,
#                  prior_height_cv_scale=0.18,
#                  prior_miss_decay=2.0,
#                  prior_min_lambda_scale=0.20,
                 
#                  ):
#         """
#         Sets key parameters for Hybrid-SORT.

#         Three added components are independent:
#         - ua_enable
#         - pc_enable
#         - depth_enable
#         """
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
#         self.args = args

#         # switches
#         self.ua_enable = bool(ua_enable)
#         self.pc_enable = bool(pc_enable)
#         self.depth_enable = bool(depth_enable)

#         # soft-depth
#         self.depth_alpha = float(depth_alpha)
#         self.depth_beta = float(depth_beta)
#         self.depth_gate = float(depth_gate)
#         self.gate_floor = float(gate_floor)

#         # UA-HMIoU
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
#         self.pc_sample_score = float(pc_sample_score)
#         self.pc_update_matched_only = bool(pc_update_matched_only)
#         self.perspective = PerspectiveHeightModel(
#             num_bins=pc_bins,
#             ema=pc_ema,
#             min_count=pc_min_count,
#             floor=pc_floor,
#             gamma=pc_gamma,
#         )

#         # depth EMA
#         self.ema_obs = float(ema_obs)
#         self.ema_pred = float(ema_pred)

#         KalmanBoxTracker.count = 0
        
#         ######
#         self.prior_w_u = float(prior_w_u)
#         self.prior_w_p = float(prior_w_p)
#         self.prior_w_d = float(prior_w_d)
#         self.prior_floor = float(prior_floor)
#         self.r1_amb_delta = float(r1_amb_delta)
#         self.r1_lambda_u = float(r1_lambda_u)
#         self.r1_lambda_d = float(r1_lambda_d)
#         self.r2_lambda_u = float(r2_lambda_u)
#         self.r2_lambda_d = float(r2_lambda_d)
#         self.r3_lambda_u = float(r3_lambda_u)
#         self.r3_lambda_d = float(r3_lambda_d)
#         self.r2_amb_delta = float(r2_amb_delta)
#         self.r3_amb_delta = float(r3_amb_delta)
#         self.r2_miss_trigger = float(r2_miss_trigger)
#         self.r3_miss_trigger = float(r3_miss_trigger)
#         self.r1_repair_enable = bool(r1_repair_enable)
#         self.r1_protect_iou = float(r1_protect_iou)
#         self.r1_protect_det_margin = float(r1_protect_det_margin)
#         self.r1_protect_trk_margin = float(r1_protect_trk_margin)
#         self.r1_protect_score = float(r1_protect_score)
#         self.r1_protect_miss = float(r1_protect_miss)
#         self.prior_height_cv_scale = float(prior_height_cv_scale)
#         self.prior_miss_decay = float(prior_miss_decay)
#         self.prior_min_lambda_scale = float(prior_min_lambda_scale)

#     # -----------------------------------------------------------------------
#     # Optional reset helper
#     # -----------------------------------------------------------------------
#     def reset(self, reset_perspective=True, reset_id_counter=False):
#         self.trackers = []
#         self.frame_count = 0
#         if reset_perspective and self.pc_enable:
#             self.perspective.reset()
#         if reset_id_counter:
#             KalmanBoxTracker.count = 0

#     # -----------------------------------------------------------------------
#     # Depth helpers
#     # -----------------------------------------------------------------------
#     def _depth_affinity(self, dets_xyxy, trks_xyxy, trk_depth_ema, img_h):
#         Nd = dets_xyxy.shape[0]
#         Nt = trks_xyxy.shape[0]
#         if Nd == 0 or Nt == 0:
#             return np.zeros((Nd, Nt), dtype=float)
#         y2d = dets_xyxy[:, 3].astype(float)[:, None]
#         y2t = np.asarray(trk_depth_ema, dtype=float)[None, :]
#         norm = max(float(img_h), 1e-6)
#         d = np.abs(y2d - y2t) / norm
#         A = np.exp(-self.depth_beta * d)
#         if self.depth_gate > 0:
#             mask_far = d > self.depth_gate
#             if self.gate_floor >= 0.0:
#                 A[mask_far] = np.maximum(A[mask_far], self.gate_floor)
#         return A

#     def _blend_with_depth(self, base_score, dets_xyxy, trks_xyxy, trk_depth_ema, img_h):
#         if (not self.depth_enable) or self.depth_alpha <= 0:
#             return base_score
#         A_d = self._depth_affinity(dets_xyxy, trks_xyxy, trk_depth_ema, img_h)
#         return base_score * ((1.0 - self.depth_alpha) + self.depth_alpha * A_d)

#     # -----------------------------------------------------------------------
#     # UA helpers
#     # -----------------------------------------------------------------------
#     def _ua_height_affinity(self,
#                             dets_xyxy,
#                             trks_xyxy,
#                             trk_mu_h,
#                             trk_sigma_h,
#                             det_scores,
#                             trk_miss,
#                             img_w,
#                             img_h):
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

#         m = max(1.0, min(float(img_w), float(img_h)) * self.ua_border_margin_frac)
#         x1 = dets_xyxy[:, 0].astype(float)
#         y1 = dets_xyxy[:, 1].astype(float)
#         x2 = dets_xyxy[:, 2].astype(float)
#         y2 = dets_xyxy[:, 3].astype(float)
#         min_dist = np.minimum(np.minimum(x1, y1), np.minimum(img_w - x2, img_h - y2))
#         closeness = np.clip(1.0 - (min_dist / m), 0.0, 1.0)
#         sc = det_scores.astype(float)
#         low_score_gap = np.clip((self.ua_low_score - sc) / max(self.ua_low_score, 1e-6), 0.0, 1.0)
#         S_det = 1.0 + self.ua_border_boost * closeness + self.ua_score_boost * low_score_gap
#         S_det = S_det[:, None]

#         sigma_eff = sigma_base * S_trk
#         sigma_eff = sigma_eff * S_det
#         sigma_eff = np.maximum(sigma_eff, eps)

#         z = np.abs(hd - mu) / (self.ua_alpha * sigma_eff + eps)
#         if self.ua_mode == "gauss":
#             return np.exp(-0.5 * z * z)
#         return 1.0 / (1.0 + z * z)
    
    
#     def _build_selective_ua_affinity(
#         self,
#         dets_xyxy,
#         trks_xyxy,
#         det_scores,
#         trk_height_mu,
#         trk_height_sigma,
#         trk_miss,
#         img_w,
#         img_h,
#         raw_iou,
#         iou_gate,
#         round_id,
#     ):
#         """
#         方案 B：
#         只在“有歧义”的 det-track pair 上启用 UA。
#         清晰匹配不干预，避免破坏 Hybrid-SORT 原本的 ROCM + TCM 主匹配。

#         ambiguous 条件（满足任一即可）：
#         1) raw_iou < iou_gate + 0.10
#         2) track 有 miss（time_since_update > 0）
#         3) detection 分数不高（score < det_thresh + 0.05）

#         返回：
#             pair_aff in [ (1-beta), 1 ]
#         其中不歧义的 pair_aff = 1.0
#         """
#         Nd = dets_xyxy.shape[0]
#         Nt = trks_xyxy.shape[0]

#         if Nd == 0 or Nt == 0:
#             return np.ones((Nd, Nt), dtype=float)

#         if not self.ua_enable:
#             return np.ones((Nd, Nt), dtype=float)

#         # 先算完整 UA affinity
#         A_u = self._ua_height_affinity(
#             dets_xyxy,
#             trks_xyxy,
#             trk_height_mu,
#             trk_height_sigma,
#             det_scores,
#             trk_miss,
#             img_w,
#             img_h,
#         )

#         # 不同轮次给不同强度：第一轮最轻，后两轮稍强
#         if round_id == 1:
#             beta = 0.15
#         elif round_id == 2:
#             beta = 0.30
#         else:
#             beta = 0.30

#         # pair 级歧义掩码
#         ambiguous = np.zeros_like(raw_iou, dtype=bool)

#         # 1) IoU 落在“模糊区”
#         ambiguous |= (raw_iou < (iou_gate + 0.10))

#         # 2) track 有 miss -> 这一整列都算模糊
#         ambiguous |= (np.asarray(trk_miss, dtype=float)[None, :] > 0)

#         # 3) det 分数不够高 -> 这一整行都算模糊
#         ambiguous |= (np.asarray(det_scores, dtype=float)[:, None] < (self.det_thresh + 0.05))

#         # 不模糊：pair_aff = 1
#         # 模糊：pair_aff = (1-beta) + beta * A_u
#         pair_aff = np.ones_like(raw_iou, dtype=float)
#         pair_aff[ambiguous] = (1.0 - beta) + beta * A_u[ambiguous]

#         return pair_aff
    

#     # -----------------------------------------------------------------------
#     # PC helpers
#     # -----------------------------------------------------------------------
#     def _pc_affinity(self, dets_xyxy, trks_xyxy, img_h):
#         Nd = dets_xyxy.shape[0]
#         Nt = trks_xyxy.shape[0]
#         if Nd == 0 or Nt == 0:
#             return np.zeros((Nd, Nt), dtype=float)
#         if (not self.pc_enable) or (not self.perspective.ready()):
#             return np.ones((Nd, Nt), dtype=float)
#         return self.perspective.affinity_matrix(dets_xyxy, trks_xyxy, img_h)

#     def _update_pc_model_with_det(self, det_row, img_h):
#         if not self.pc_enable:
#             return
#         score = float(det_row[4])
#         if score < self.pc_sample_score:
#             return
#         y2 = float(det_row[3])
#         h = float(det_row[3] - det_row[1])
#         if h > 0 and np.isfinite(h) and np.isfinite(y2):
#             self.perspective.observe(y2, h, img_h)

#     # -----------------------------------------------------------------------
#     # Pair affinity builder shared by all rounds
#     # -----------------------------------------------------------------------
#     # def _build_pair_affinity(self,
#     #                          dets_xyxy,
#     #                          trks_xyxy,
#     #                          det_scores,
#     #                          trk_height_mu,
#     #                          trk_height_sigma,
#     #                          trk_miss,
#     #                          trk_depth_ema,
#     #                          img_w,
#     #                          img_h):
#     #     Nd = dets_xyxy.shape[0]
#     #     Nt = trks_xyxy.shape[0]
#     #     if Nd == 0 or Nt == 0:
#     #         return np.zeros((Nd, Nt), dtype=float)

#     #     pair_aff = np.ones((Nd, Nt), dtype=float)

#     #     if self.ua_enable:
#     #         pair_aff *= self._ua_height_affinity(
#     #             dets_xyxy, trks_xyxy,
#     #             trk_height_mu, trk_height_sigma,
#     #             det_scores, trk_miss,
#     #             img_w, img_h,
#     #         )

#     #     if self.pc_enable:
#     #         pair_aff *= self._pc_affinity(dets_xyxy, trks_xyxy, img_h)

#     #     if self.depth_enable:
#     #         pair_aff = self._blend_with_depth(
#     #             pair_aff,
#     #             dets_xyxy,
#     #             trks_xyxy,
#     #             trk_depth_ema,
#     #             img_h,
#     #         )

#     #     return pair_aff
    
    
#     def _build_prior_components(
#         self,
#         dets_xyxy,
#         trks_xyxy,
#         det_scores,
#         trk_height_mu,
#         trk_height_sigma,
#         trk_miss,
#         trk_depth_ema,
#         img_w,
#         img_h,
#         include_pc=False,
#     ):
#         Nd = dets_xyxy.shape[0]
#         Nt = trks_xyxy.shape[0]
#         if Nd == 0 or Nt == 0:
#             z = np.zeros((Nd, Nt), dtype=float)
#             return z, z, z

#         A_u = np.ones((Nd, Nt), dtype=float)
#         A_p = np.ones((Nd, Nt), dtype=float)
#         A_d = np.ones((Nd, Nt), dtype=float)

#         if self.ua_enable:
#             A_u = self._ua_height_affinity(
#                 dets_xyxy, trks_xyxy,
#                 trk_height_mu, trk_height_sigma,
#                 det_scores, trk_miss,
#                 img_w, img_h
#             )

#         # GPM/PC is intentionally disabled for scoring on this main line.
#         # The perspective model can still warm up via _update_pc_model_with_det().
#         if include_pc and self.pc_enable:
#             A_p = self._pc_affinity(dets_xyxy, trks_xyxy, img_h)

#         if self.depth_enable:
#             A_d = self._depth_affinity(dets_xyxy, trks_xyxy, trk_depth_ema, img_h)

#         return A_u, A_p, A_d
    
    
#     def _build_residual_prior(self, A_u, A_p, A_d):
#         prior = (
#             1.0
#             + self.prior_w_u * (A_u - 1.0)
#             + self.prior_w_p * (A_p - 1.0)
#             + self.prior_w_d * (A_d - 1.0)
#         )
#         prior = np.clip(prior, self.prior_floor, 1.0)
#         return prior

#     def _track_prior_reliability(self, trk_height_mu, trk_height_sigma, trk_miss):
#         trk_height_mu = _to_numpy(trk_height_mu, dtype=float).reshape(-1)
#         trk_height_sigma = _to_numpy(trk_height_sigma, dtype=float).reshape(-1)
#         trk_miss = _to_numpy(trk_miss, dtype=float).reshape(-1)
#         if trk_height_mu.size == 0:
#             return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)

#         eps = 1e-6
#         cv = trk_height_sigma / np.maximum(trk_height_mu, eps)
#         cv_scale = max(float(self.prior_height_cv_scale), eps)
#         height_rel = 1.0 / (1.0 + (cv / cv_scale) ** 2)

#         # In this tracker, predict() increments time_since_update before
#         # association, so a normal live track has miss ~= 1 here.
#         miss_excess = np.maximum(trk_miss - 1.0, 0.0)
#         miss_decay = max(float(self.prior_miss_decay), eps)
#         miss_rel = np.exp(-miss_excess / miss_decay)

#         floor = np.clip(float(self.prior_min_lambda_scale), 0.0, 1.0)
#         shs_rel = np.clip(height_rel * miss_rel, floor, 1.0)
#         depth_rel = np.clip(miss_rel, floor, 1.0)
#         return shs_rel, depth_rel

#     def _lambda_matrices(self,
#                          score_shape,
#                          lambda_u,
#                          lambda_d,
#                          trk_height_mu,
#                          trk_height_sigma,
#                          trk_miss):
#         Nd, Nt = score_shape
#         if Nd == 0 or Nt == 0:
#             z = np.zeros(score_shape, dtype=float)
#             return z, z

#         shs_rel, depth_rel = self._track_prior_reliability(
#             trk_height_mu,
#             trk_height_sigma,
#             trk_miss,
#         )
#         if shs_rel.size != Nt:
#             shs_rel = np.ones((Nt,), dtype=float)
#         if depth_rel.size != Nt:
#             depth_rel = np.ones((Nt,), dtype=float)

#         lambda_u_mat = float(lambda_u) * shs_rel[None, :]
#         lambda_d_mat = float(lambda_d) * depth_rel[None, :]
#         return (
#             np.broadcast_to(lambda_u_mat, score_shape).astype(float, copy=False),
#             np.broadcast_to(lambda_d_mat, score_shape).astype(float, copy=False),
#         )

#     def _add_shs_depth_log_penalty(self, base_score, A_u, A_d, lambda_u, lambda_d, row_mask=None, pair_mask=None):
#         score = np.array(base_score, dtype=float, copy=True)
#         if score.size == 0:
#             return score

#         eps = 1e-6
#         penalty = np.zeros_like(score, dtype=float)
#         lambda_u = np.asarray(lambda_u, dtype=float)
#         lambda_d = np.asarray(lambda_d, dtype=float)
#         if self.ua_enable and np.any(lambda_u > 0):
#             penalty += lambda_u * np.log(np.clip(A_u, eps, 1.0))
#         if self.depth_enable and np.any(lambda_d > 0):
#             penalty += lambda_d * np.log(np.clip(A_d, eps, 1.0))

#         if row_mask is None:
#             if pair_mask is None:
#                 return score + penalty
#             pair_mask = np.asarray(pair_mask, dtype=bool)
#             if pair_mask.shape != score.shape:
#                 return score
#             score[pair_mask] += penalty[pair_mask]
#             return score

#         row_mask = np.asarray(row_mask, dtype=bool).reshape(-1)
#         if row_mask.size != score.shape[0]:
#             return score
#         score[row_mask, :] += penalty[row_mask, :]
#         return score

#     def _score_margins(self, score_matrix):
#         score_matrix = np.asarray(score_matrix, dtype=float)
#         Nd, Nt = score_matrix.shape
#         if Nd == 0 or Nt == 0:
#             return np.zeros((Nd,), dtype=float), np.zeros((Nt,), dtype=float)

#         if Nt < 2:
#             det_margin = np.full((Nd,), np.inf, dtype=float)
#         else:
#             top2_det = np.partition(score_matrix, -2, axis=1)[:, -2:]
#             det_margin = np.max(top2_det, axis=1) - np.min(top2_det, axis=1)

#         if Nd < 2:
#             trk_margin = np.full((Nt,), np.inf, dtype=float)
#         else:
#             top2_trk = np.partition(score_matrix, -2, axis=0)[-2:, :]
#             trk_margin = np.max(top2_trk, axis=0) - np.min(top2_trk, axis=0)

#         return det_margin, trk_margin

#     def _ambiguous_detection_mask(self, score_matrix, amb_delta):
#         det_margin, _ = self._score_margins(score_matrix)
#         return det_margin < float(amb_delta)

#     def _selective_prior_mask(self, score_matrix, amb_delta, trk_miss=None, miss_trigger=2.0):
#         score_matrix = np.asarray(score_matrix, dtype=float)
#         Nd, Nt = score_matrix.shape
#         if Nd == 0 or Nt == 0:
#             return np.zeros((Nd, Nt), dtype=bool)

#         det_mask = self._ambiguous_detection_mask(score_matrix, amb_delta)[:, None]
#         pair_mask = np.broadcast_to(det_mask, score_matrix.shape).copy()

#         if trk_miss is not None:
#             trk_miss = _to_numpy(trk_miss, dtype=float).reshape(-1)
#             if trk_miss.size == Nt:
#                 pair_mask |= (trk_miss[None, :] >= float(miss_trigger))

#         return pair_mask

#     def _split_protected_matches(self,
#                                  matches,
#                                  iou_matrix,
#                                  hybrid_score,
#                                  detections,
#                                  trk_miss):
#         if len(matches) == 0:
#             return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=int)

#         det_margin, trk_margin = self._score_margins(hybrid_score)
#         trk_miss = _to_numpy(trk_miss, dtype=float).reshape(-1)
#         protected = []
#         repair = []
#         protect_iou = max(float(self.r1_protect_iou), float(self.iou_threshold))

#         for m in matches:
#             d, t = int(m[0]), int(m[1])
#             det_score = float(detections[d, 4]) if detections.shape[1] > 4 else 1.0
#             miss = trk_miss[t] if t < trk_miss.size else 1.0
#             is_protected = (
#                 iou_matrix[d, t] >= protect_iou
#                 and det_margin[d] >= self.r1_protect_det_margin
#                 and trk_margin[t] >= self.r1_protect_trk_margin
#                 and det_score >= self.r1_protect_score
#                 and miss <= self.r1_protect_miss
#             )
#             if is_protected:
#                 protected.append([d, t])
#             else:
#                 repair.append([d, t])

#         if len(protected) == 0:
#             protected = np.empty((0, 2), dtype=int)
#         else:
#             protected = np.asarray(protected, dtype=int)
#         if len(repair) == 0:
#             repair = np.empty((0, 2), dtype=int)
#         else:
#             repair = np.asarray(repair, dtype=int)
#         return protected, repair

#     # -----------------------------------------------------------------------
#     # First-round Hybrid association kept identical to Hybrid-SORT
#     # -----------------------------------------------------------------------
#     def _associate_4_points_augmented(self,
#                                       detections,
#                                       trackers,
#                                       iou_threshold,
#                                       lt,
#                                       rt,
#                                       lb,
#                                       rb,
#                                       previous_obs,
#                                       vdc_weight,
#                                       iou_type=None,
#                                       args=None,
#                                       A_u=None,
#                                       A_d=None,
#                                       amb_delta=0.05,
#                                       lambda_u=0.0,
#                                       lambda_d=0.0,
#                                       trk_height_mu=None,
#                                       trk_height_sigma=None,
#                                       trk_miss=None,
#                                       use_tcm_score=False):
#         if len(trackers) == 0:
#             return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

#         Y1, X1 = speed_direction_batch_lt(detections, previous_obs)
#         Y2, X2 = speed_direction_batch_rt(detections, previous_obs)
#         Y3, X3 = speed_direction_batch_lb(detections, previous_obs)
#         Y4, X4 = speed_direction_batch_rb(detections, previous_obs)

#         cost_lt = cost_vel(Y1, X1, trackers, lt, detections, previous_obs, vdc_weight)
#         cost_rt = cost_vel(Y2, X2, trackers, rt, detections, previous_obs, vdc_weight)
#         cost_lb = cost_vel(Y3, X3, trackers, lb, detections, previous_obs, vdc_weight)
#         cost_rb = cost_vel(Y4, X4, trackers, rb, detections, previous_obs, vdc_weight)

#         iou_matrix = iou_type(detections, trackers)
#         iou_matrix = np.array(iou_matrix)

#         angle_diff_cost = cost_lt + cost_rt + cost_lb + cost_rb
#         hybrid_score = iou_matrix + angle_diff_cost
#         if use_tcm_score:
#             score_dif = cal_score_dif_batch(detections, trackers)
#             hybrid_score -= score_dif * float(getattr(args, "TCM_first_step_weight", 1.0))

#         # Step 1: exact Hybrid-SORT first-round association.
#         if min(iou_matrix.shape) > 0:
#             a = (iou_matrix > iou_threshold).astype(np.int32)
#             if a.sum(1).max() == 1 and a.sum(0).max() == 1:
#                 matched_indices = np.stack(np.where(a), axis=1)
#             else:
#                 matched_indices = linear_assignment(-hybrid_score)
#         else:
#             matched_indices = np.empty(shape=(0, 2))

#         unmatched_detections = []
#         for d in range(len(detections)):
#             if d not in matched_indices[:, 0]:
#                 unmatched_detections.append(d)
#         unmatched_trackers = []
#         for t in range(len(trackers)):
#             if t not in matched_indices[:, 1]:
#                 unmatched_trackers.append(t)

#         matches = []
#         for m in matched_indices:
#             if iou_matrix[m[0], m[1]] < iou_threshold:
#                 unmatched_detections.append(m[0])
#                 unmatched_trackers.append(m[1])
#             else:
#                 matches.append(m.reshape(1, 2))

#         if len(matches) == 0:
#             matches = np.empty((0, 2), dtype=int)
#         else:
#             matches = np.concatenate(matches, axis=0)

#         unmatched_detections = np.array(unmatched_detections, dtype=int)
#         unmatched_trackers = np.array(unmatched_trackers, dtype=int)

#         # Optional PBR: protect easy baseline matches, and only repair the
#         # uncertain local subgraph with SHS + Depth residual prior.
#         if not self.r1_repair_enable:
#             return matches, unmatched_detections, unmatched_trackers
#         if A_u is None or A_d is None or trk_miss is None:
#             return matches, unmatched_detections, unmatched_trackers

#         A_u = np.asarray(A_u, dtype=float)
#         A_d = np.asarray(A_d, dtype=float)
#         if A_u.shape != iou_matrix.shape or A_d.shape != iou_matrix.shape:
#             return matches, unmatched_detections, unmatched_trackers

#         protected, repair = self._split_protected_matches(
#             matches,
#             iou_matrix,
#             hybrid_score,
#             detections,
#             trk_miss,
#         )

#         if len(repair) == 0 and len(unmatched_detections) == 0 and len(unmatched_trackers) == 0:
#             return matches, unmatched_detections, unmatched_trackers

#         if len(repair) > 0:
#             repair_dets = repair[:, 0]
#             repair_trks = repair[:, 1]
#         else:
#             repair_dets = np.empty((0,), dtype=int)
#             repair_trks = np.empty((0,), dtype=int)

#         local_dets = np.unique(np.concatenate((repair_dets, unmatched_detections))).astype(int)
#         local_trks = np.unique(np.concatenate((repair_trks, unmatched_trackers))).astype(int)

#         local_matches = []
#         if len(local_dets) > 0 and len(local_trks) > 0:
#             local_iou = iou_matrix[np.ix_(local_dets, local_trks)]
#             local_score = hybrid_score[np.ix_(local_dets, local_trks)]
#             local_A_u = A_u[np.ix_(local_dets, local_trks)]
#             local_A_d = A_d[np.ix_(local_dets, local_trks)]
#             local_A_p = np.ones_like(local_A_u, dtype=float)
#             local_prior = self._build_residual_prior(local_A_u, local_A_p, local_A_d)
#             local_score = local_score * local_prior

#             if local_iou.max() > iou_threshold:
#                 local_indices = linear_assignment(-local_score)
#                 for m in local_indices:
#                     d_local, t_local = int(m[0]), int(m[1])
#                     if local_iou[d_local, t_local] < iou_threshold:
#                         continue
#                     local_matches.append([local_dets[d_local], local_trks[t_local]])

#         if len(local_matches) == 0:
#             final_matches = protected
#         else:
#             local_matches = np.asarray(local_matches, dtype=int)
#             final_matches = local_matches if len(protected) == 0 else np.concatenate((protected, local_matches), axis=0)

#         if len(final_matches) == 0:
#             unmatched_detections = np.arange(len(detections), dtype=int)
#             unmatched_trackers = np.arange(len(trackers), dtype=int)
#         else:
#             matched_dets = set(final_matches[:, 0].tolist())
#             matched_trks = set(final_matches[:, 1].tolist())
#             unmatched_detections = np.array([d for d in range(len(detections)) if d not in matched_dets], dtype=int)
#             unmatched_trackers = np.array([t for t in range(len(trackers)) if t not in matched_trks], dtype=int)

#         return final_matches, unmatched_detections, unmatched_trackers

#     def update(self, output_results, img_info, img_size, frame_id=None):
#         """
#         Params:
#           dets - a numpy array of detections in the format
#                  [[x1,y1,x2,y2,score], ...]
#         Requires: this method must be called once for each frame even with empty
#         detections (use np.empty((0, 5)) for frames without detections).
#         Returns a similar array, where the last column is the object ID.
#         """
#         if output_results is None:
#             return np.empty((0, 5))

#         self.frame_count += 1

#         # unify detections to numpy
#         try:
#             import torch
#             if isinstance(output_results, torch.Tensor):
#                 output_results = output_results.detach().cpu().numpy()
#         except Exception:
#             pass
#         if not isinstance(output_results, np.ndarray):
#             output_results = np.array(output_results)

#         # post-process detections
#         if output_results.shape[1] == 5:
#             scores = output_results[:, 4].astype(float)
#             bboxes = output_results[:, :4].astype(float)
#         else:
#             scores = (output_results[:, 4] * output_results[:, 5]).astype(float)
#             bboxes = output_results[:, :4].astype(float)

#         img_h, img_w = _to_scalar(img_info[0]), _to_scalar(img_info[1])
#         scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
#         bboxes /= max(scale, 1e-6)
#         dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)

#         inds_low = scores > 0.1
#         inds_high = scores < self.det_thresh
#         inds_second = np.logical_and(inds_low, inds_high)
#         dets_second = dets[inds_second]
#         remain_inds = scores > self.det_thresh
#         dets = dets[remain_inds]

#         # predict trackers
#         trks = np.zeros((len(self.trackers), 6))
#         to_del = []
#         ret = []
#         # for t, trk in enumerate(trks):
#         #     pos, kalman_score, simple_score = self.trackers[t].predict()
#         #     try:
#         #         trk[:] = [pos[0][0], pos[0][1], pos[0][2], pos[0][3], kalman_score, simple_score[0]]
#         #     except Exception:
#         #         trk[:] = [pos[0][0], pos[0][1], pos[0][2], pos[0][3], kalman_score, simple_score]
#         #     if np.any(np.isnan(pos)):
#         #         to_del.append(t)
        
#         for t, trk in enumerate(trks):
#             pos, kalman_score, simple_score = self.trackers[t].predict()
#             try:
#                 trk[:] = [pos[0][0], pos[0][1], pos[0][2], pos[0][3], kalman_score, simple_score[0]]
#             except Exception:
#                 trk[:] = [pos[0][0], pos[0][1], pos[0][2], pos[0][3], kalman_score, simple_score]
#             if np.any(np.isnan(pos)):
#                 to_del.append(t)
        
        
        
#         trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#         for t in reversed(to_del):
#             self.trackers.pop(t)

#         velocities_lt = np.array([trk.velocity_lt if trk.velocity_lt is not None else np.array((0, 0)) for trk in self.trackers])
#         velocities_rt = np.array([trk.velocity_rt if trk.velocity_rt is not None else np.array((0, 0)) for trk in self.trackers])
#         velocities_lb = np.array([trk.velocity_lb if trk.velocity_lb is not None else np.array((0, 0)) for trk in self.trackers])
#         velocities_rb = np.array([trk.velocity_rb if trk.velocity_rb is not None else np.array((0, 0)) for trk in self.trackers])
#         last_boxes = np.array([trk.last_observation for trk in self.trackers])
#         k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

#         # extra stats aligned with self.trackers
#         depth_emas = np.array([trk.depth_ema for trk in self.trackers], dtype=float) if len(self.trackers) > 0 else np.zeros((0,), dtype=float)
#         height_mus_full = np.array([trk.get_height_mu_sigma()[0] for trk in self.trackers], dtype=float) if len(self.trackers) > 0 else np.zeros((0,), dtype=float)
#         height_sigmas_full = np.array([trk.get_height_mu_sigma()[1] for trk in self.trackers], dtype=float) if len(self.trackers) > 0 else np.zeros((0,), dtype=float)
#         miss_counts_full = np.array([trk.time_since_update for trk in self.trackers], dtype=float) if len(self.trackers) > 0 else np.zeros((0,), dtype=float)

#         # -------------------------------------------------------------------
#         # First round: run Hybrid-SORT first, then optionally repair only the
#         # uncertain local subgraph with SHS + Depth. GPM/PC is not used here.
#         # -------------------------------------------------------------------
#         if dets.shape[0] > 0 and trks.shape[0] > 0:
#             A_u_r1, A_p_r1, A_d_r1 = self._build_prior_components(
#                 dets[:, :4],
#                 trks[:, :4],
#                 dets[:, 4],
#                 height_mus_full,
#                 height_sigmas_full,
#                 miss_counts_full,
#                 depth_emas,
#                 img_w,
#                 img_h,
#             )
#             matched, unmatched_dets, unmatched_trks = self._associate_4_points_augmented(
#                 dets,
#                 trks,
#                 self.iou_threshold,
#                 velocities_lt,
#                 velocities_rt,
#                 velocities_lb,
#                 velocities_rb,
#                 k_observations,
#                 self.inertia,
#                 self.asso_func,
#                 self.args,
#                 A_u=A_u_r1,
#                 A_d=A_d_r1,
#                 trk_miss=miss_counts_full,
#                 use_tcm_score=bool(getattr(self.args, "TCM_first_step", False)),
#             )
#         else:
#             matched = np.empty((0, 2), dtype=int)
#             unmatched_dets = np.arange(dets.shape[0])
#             unmatched_trks = np.arange(trks.shape[0])

#         for m in matched:
#             self.trackers[m[1]].update(dets[m[0], :])
#             self._update_pc_model_with_det(dets[m[0], :], img_h)

#         # -------------------------------------------------------------------
#         # Second round: BYTE with SHS + Depth, TCM-byte preserved, no GPM score.
#         # -------------------------------------------------------------------
#         if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
#             u_trks = trks[unmatched_trks]
#             geom_left = np.array(self.asso_func(dets_second, u_trks))

#             if geom_left.size > 0:
#                 A_u_r2, A_p_r2, A_d_r2 = self._build_prior_components(
#                     dets_second[:, :4],
#                     u_trks[:, :4],
#                     dets_second[:, 4],
#                     height_mus_full[unmatched_trks],
#                     height_sigmas_full[unmatched_trks],
#                     miss_counts_full[unmatched_trks],
#                     depth_emas[unmatched_trks],
#                     img_w,
#                     img_h,
#                 )

#                 byte_score = np.array(geom_left, dtype=float, copy=True)
#                 if getattr(self.args, "TCM_byte_step", False):
#                     byte_score -= np.array(
#                         cal_score_dif_batch_two_score(dets_second, u_trks)
#                         * float(getattr(self.args, "TCM_byte_step_weight", 1.0))
#                     )

#                 # BYTE uses a mild residual prior instead of log penalty.
#                 # Acceptance is still checked by geom_left below.
#                 prior_r2 = self._build_residual_prior(A_u_r2, A_p_r2, A_d_r2)
#                 score_left = byte_score * prior_r2

#                 if geom_left.max() > self.iou_threshold:
#                     matched_indices = linear_assignment(-score_left)
#                     to_remove_trk_indices = []
#                     to_remove_det_indices = []
#                     for m in matched_indices:
#                         det_local, trk_local = m[0], m[1]
#                         trk_ind = unmatched_trks[trk_local]
#                         if geom_left[det_local, trk_local] < self.iou_threshold:
#                             continue
#                         self.trackers[trk_ind].update(dets_second[det_local, :])
#                         to_remove_trk_indices.append(trk_ind)
#                         to_remove_det_indices.append(det_local)
#                         self._update_pc_model_with_det(dets_second[det_local, :], img_h)
#                     if len(to_remove_trk_indices) > 0:
#                         unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))
#                     if len(to_remove_det_indices) > 0:
#                         keep_mask = np.ones(len(dets_second), dtype=bool)
#                         keep_mask[to_remove_det_indices] = False
#                         dets_second = dets_second[keep_mask]

#         # -------------------------------------------------------------------
#         # Third round: rematch with last observations + SHS/Depth, no GPM score.
#         # -------------------------------------------------------------------
#         if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
#             left_dets = dets[unmatched_dets]
#             left_trks = last_boxes[unmatched_trks]
#             geom_left = np.array(self.asso_func(left_dets, left_trks))

#             if geom_left.size > 0:
#                 A_u_r3, A_p_r3, A_d_r3 = self._build_prior_components(
#                     left_dets[:, :4],
#                     left_trks[:, :4],
#                     left_dets[:, 4],
#                     height_mus_full[unmatched_trks],
#                     height_sigmas_full[unmatched_trks],
#                     miss_counts_full[unmatched_trks],
#                     depth_emas[unmatched_trks],
#                     img_w,
#                     img_h,
#                 )

#                 # Rematch also uses residual prior. It can change ranking,
#                 # but final acceptance still uses the original geom_left score.
#                 prior_r3 = self._build_residual_prior(A_u_r3, A_p_r3, A_d_r3)
#                 score_left = geom_left * prior_r3

#                 if geom_left.max() > self.iou_threshold:
#                     rematched_indices = linear_assignment(-score_left)
#                     to_remove_det_indices = []
#                     to_remove_trk_indices = []
#                     for m in rematched_indices:
#                         det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
#                         if geom_left[m[0], m[1]] < self.iou_threshold:
#                             continue
#                         self.trackers[trk_ind].update(dets[det_ind, :])
#                         to_remove_det_indices.append(det_ind)
#                         to_remove_trk_indices.append(trk_ind)
#                         self._update_pc_model_with_det(dets[det_ind, :], img_h)
#                     unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices)) if len(to_remove_det_indices) > 0 else unmatched_dets
#                     unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices)) if len(to_remove_trk_indices) > 0 else unmatched_trks

#         # Unmatched tracks: no measurement update
#         for m in unmatched_trks:
#             self.trackers[m].update(None)

#         # Create new trackers
#         for i in unmatched_dets:
#             trk = KalmanBoxTracker(
#                 dets[i, :],
#                 delta_t=self.delta_t,
#                 args=self.args,
#                 depth_enable=self.depth_enable,
#                 ema_obs=self.ema_obs,
#                 ema_pred=self.ema_pred,
#                 ua_enable=self.ua_enable,
#                 ua_h_ema=self.ua_h_ema,
#                 ua_h2_ema=self.ua_h2_ema,
#                 ua_proc_var_frac=self.ua_proc_var_frac,
#                 ua_sigma_floor_frac=self.ua_sigma_floor_frac,
#             )
#             self.trackers.append(trk)
#             if self.pc_enable and (not self.pc_update_matched_only):
#                 self._update_pc_model_with_det(dets[i, :], img_h)

#         i = len(self.trackers)
#         for trk in reversed(self.trackers):
#             if trk.last_observation.sum() < 0:
#                 d = trk.get_state()[0][:4]
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
#         """
#         Keep public-det pipeline behavior close to original Hybrid-SORT.
#         Added tracker side statistics do not alter its association policy.
#         """
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

#         if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
#             left_dets = dets[unmatched_dets]
#             left_trks = last_boxes[unmatched_trks]
#             left_dets_c = left_dets.copy()
#             left_trks_c = left_trks.copy()

#             iou_left = np.array(self.asso_func(left_dets_c, left_trks_c))
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
#                 unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices)) if len(to_remove_det_indices) > 0 else unmatched_dets
#                 unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices)) if len(to_remove_trk_indices) > 0 else unmatched_trks

#         for i in unmatched_dets:
#             trk = KalmanBoxTracker(
#                 dets[i, :],
#                 delta_t=self.delta_t,
#                 args=self.args,
#                 depth_enable=self.depth_enable,
#                 ema_obs=self.ema_obs,
#                 ema_pred=self.ema_pred,
#                 ua_enable=self.ua_enable,
#                 ua_h_ema=self.ua_h_ema,
#                 ua_h2_ema=self.ua_h2_ema,
#                 ua_proc_var_frac=self.ua_proc_var_frac,
#                 ua_sigma_floor_frac=self.ua_sigma_floor_frac,
#             )
#             trk.cate = cates[i]
#             self.trackers.append(trk)

#         i = len(self.trackers)
#         for trk in reversed(self.trackers):
#             if trk.last_observation.sum() > 0:
#                 d = trk.last_observation[:4]
#             else:
#                 d = trk.get_state()[0][:4]
#             if trk.time_since_update < 1:
#                 if (self.frame_count <= self.min_hits) or (trk.hit_streak >= self.min_hits):
#                     ret.append(np.concatenate((d, [trk.id + 1], [getattr(trk, "cate", 1)], [0])).reshape(1, -1))
#                 if trk.hit_streak == self.min_hits:
#                     for prev_i in range(self.min_hits - 1):
#                         prev_observation = trk.history_observations[-(prev_i + 2)]
#                         ret.append((np.concatenate((prev_observation[:4], [trk.id + 1], [getattr(trk, "cate", 1)], [-(prev_i + 1)]))).reshape(1, -1))
#             i -= 1
#             if trk.time_since_update > self.max_age:
#                 self.trackers.pop(i)

#         if len(ret) > 0:
#             return np.concatenate(ret)
#         return np.empty((0, 7))



"""
    目前再dan-val指标最高，但在dan-test不如上方codex修复过的
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai

    Hybrid-SORT + optional geometric priors:
    - UA-HMIoU  : uncertainty-aware height affinity
    - PC-HMIoU  : perspective-consistent height affinity
    - Soft-Depth: soft depth continuity regularization by y2 EMA

    Design goal:
    Keep Hybrid-SORT's original strengths (4-point motion consistency,
    confidence-state KF, TCM, BYTE, rematching), and only use the above three
    components as additional pairwise geometric priors.

    Important implementation choices:
    1) First-round association remains Hybrid-SORT style.
    2) All three components are independent toggles.
    3) Acceptance gates still use the ORIGINAL geometric IoU-family score,
       while the augmented score is only used for ranking / Hungarian matching.
    4) No changes are required in association.py.
"""
from __future__ import print_function

import numpy as np
from .association import *


# ---------------------------------------------------------------------------
# Safe conversions
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Original helpers
# ---------------------------------------------------------------------------

def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age-dt]
    max_age = max(observations.keys())
    return observations[max_age]


# NOTE: Hybrid-SORT uses [x, y, s, score, r] in the measurement/state front.
def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2,score] and returns z in the
    form [x,y,s,score,r] where x,y is the centre of the box and s is the
    scale/area and r is the aspect ratio.
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h + 1e-6)
    score = bbox[4]
    return np.array([x, y, s, score, r]).reshape((5, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,score,r] and returns it in
    the form [x1,y1,x2,y2] or [x1,y1,x2,y2,score].
    """
    w = np.sqrt(max(float(x[2] * x[4]), 0.0))
    if w < 1e-12:
        h = 0.0
    else:
        h = x[2] / w
    state_score = x[3]
    if score is None:
        score = state_score
    if score is None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0,
                         x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
    return np.array([x[0] - w / 2.0, x[1] - h / 2.0,
                     x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def speed_direction_lt(bbox1, bbox2):
    cx1, cy1 = bbox1[0], bbox1[1]
    cx2, cy2 = bbox2[0], bbox2[1]
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def speed_direction_rt(bbox1, bbox2):
    cx1, cy1 = bbox1[0], bbox1[3]
    cx2, cy2 = bbox2[0], bbox2[3]
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def speed_direction_lb(bbox1, bbox2):
    cx1, cy1 = bbox1[2], bbox1[1]
    cx2, cy2 = bbox2[2], bbox2[1]
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def speed_direction_rb(bbox1, bbox2):
    cx1, cy1 = bbox1[2], bbox1[3]
    cx2, cy2 = bbox2[2], bbox2[3]
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


# ---------------------------------------------------------------------------
# Perspective model (scene-level prior)
# ---------------------------------------------------------------------------

class PerspectiveHeightModel(object):
    """
    Online mapping h*(y2): bottom y2 -> expected pixel height.
    Implemented via normalized y2 binning + EMA; supports interpolation and
    fallback to global EMA during warmup.
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
        if (not np.isfinite(y2_norm)) or (not np.isfinite(h_pixel)) or h_pixel <= 0:
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
        y2_pixels = np.asarray(y2_pixels, dtype=float)
        if img_h <= 0 or len(y2_pixels) == 0:
            return np.zeros_like(y2_pixels, dtype=float)

        y2n = np.clip(y2_pixels / float(img_h), 0.0, 1.0 - 1e-8)
        bins = np.floor(y2n * self.num_bins).astype(int)
        pos = y2n * self.num_bins
        frac = pos - np.floor(pos)
        b0 = bins
        b1 = np.clip(bins + 1, 0, self.num_bins - 1)

        m0, m1 = self.bin_means[b0], self.bin_means[b1]
        c0, c1 = self.bin_counts[b0], self.bin_counts[b1]
        use0 = c0 >= self.min_count
        use1 = c1 >= self.min_count

        m0_eff = np.where(use0, m0, np.where(use1, m1, self.global_ema))
        m1_eff = np.where(use1, m1, np.where(use0, m0, self.global_ema))
        h_star = (1.0 - frac) * m0_eff + frac * m1_eff
        return np.maximum(h_star, 4.0)

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
        if self.floor > 0.0:
            A_p = np.maximum(A_p, self.floor)
        return A_p

    def ready(self):
        return self.global_count >= 50


# ---------------------------------------------------------------------------
# Kalman tracker
# ---------------------------------------------------------------------------

class KalmanBoxTracker(object):
    """
    Hybrid-SORT Kalman tracker + optional side statistics for
    depth regularization and uncertainty-aware height affinity.
    """
    count = 0

    def __init__(self,
                 bbox,
                 delta_t=3,
                 orig=False,
                 args=None,
                 depth_enable=True,
                 ema_obs=0.5,
                 ema_pred=0.1,
                 ua_enable=True,
                 ua_h_ema=0.5,
                 ua_h2_ema=0.3,
                 ua_proc_var_frac=0.02,
                 ua_sigma_floor_frac=0.05):
        if not orig:
            from .kalmanfilter_score_new import KalmanFilterNew_score_new as KalmanFilter_score_new
            self.kf = KalmanFilter_score_new(dim_x=9, dim_z=5)
        else:
            from filterpy.kalman import KalmanFilter
            self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # u, v, s, c, r, ~u, ~v, ~s, ~c
        self.kf.F = np.array([
            [1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
        ])

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[5:, 5:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[-2, -2] *= 0.01
        self.kf.Q[5:, 5:] *= 0.01
        self.kf.x[:5] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.age_recover_for_cbiou = 0

        self.last_observation = np.array([-1, -1, -1, -1, -1])
        self.last_observation_save = np.array([-1, -1, -1, -1, -1])
        self.observations = dict()
        self.history_observations = []

        self.velocity = None
        self.velocity_lt = None
        self.velocity_rt = None
        self.velocity_lb = None
        self.velocity_rb = None
        self.delta_t = delta_t
        self.confidence_pre = None
        self.confidence = bbox[-1]
        self.args = args
        self.kf.args = args

        # Optional side statistics
        self.depth_enable = bool(depth_enable)
        self.ema_obs = float(ema_obs)
        self.ema_pred = float(ema_pred)
        self.depth_ema = float(bbox[3])

        self.ua_enable = bool(ua_enable)
        self.ua_h_ema = float(ua_h_ema)
        self.ua_h2_ema = float(ua_h2_ema)
        self.ua_proc_var_frac = float(ua_proc_var_frac)
        self.ua_sigma_floor_frac = float(ua_sigma_floor_frac)
        h0 = float(max(bbox[3] - bbox[1], 1e-6))
        self.h_m1 = h0
        self.h_m2 = h0 * h0
        floor_var = (self.ua_sigma_floor_frac * h0) ** 2
        self.h_var = max(self.h_m2 - self.h_m1 * self.h_m1, floor_var)

    def get_height_mu_sigma(self):
        mu = float(self.h_m1)
        floor_var = (self.ua_sigma_floor_frac * max(mu, 1e-6)) ** 2
        var = max(float(self.h_var), floor_var)
        sigma = np.sqrt(var)
        return mu, sigma

    def _inflate_height_uncertainty(self):
        if not self.ua_enable:
            return
        mu = max(float(self.h_m1), 1e-6)
        self.h_var = float(self.h_var) + (self.ua_proc_var_frac * mu) ** 2

    def _update_depth_and_height_stats(self, bbox):
        if bbox is None:
            return
        if self.depth_enable:
            y2 = float(bbox[3])
            self.depth_ema = (1.0 - self.ema_obs) * self.depth_ema + self.ema_obs * y2
        if self.ua_enable:
            h = float(max(bbox[3] - bbox[1], 1e-6))
            self.h_m1 = (1.0 - self.ua_h_ema) * self.h_m1 + self.ua_h_ema * h
            self.h_m2 = (1.0 - self.ua_h2_ema) * self.h_m2 + self.ua_h2_ema * (h * h)
            raw_var = max(self.h_m2 - self.h_m1 * self.h_m1, 0.0)
            floor_var = (self.ua_sigma_floor_frac * max(self.h_m1, 1e-6)) ** 2
            self.h_var = 0.5 * self.h_var + 0.5 * max(raw_var, floor_var)

    def update(self, bbox):
        velocity = None
        velocity_lt = None
        velocity_rt = None
        velocity_lb = None
        velocity_rb = None

        if bbox is not None:
            if self.last_observation.sum() >= 0:
                previous_box = None
                for i in range(self.delta_t):
                    if self.age - i - 1 in self.observations:
                        previous_box = self.observations[self.age - i - 1]
                        if velocity_lt is not None:
                            velocity += speed_direction(previous_box, bbox)
                            velocity_lt += speed_direction_lt(previous_box, bbox)
                            velocity_rt += speed_direction_rt(previous_box, bbox)
                            velocity_lb += speed_direction_lb(previous_box, bbox)
                            velocity_rb += speed_direction_rb(previous_box, bbox)
                        else:
                            velocity = speed_direction(previous_box, bbox)
                            velocity_lt = speed_direction_lt(previous_box, bbox)
                            velocity_rt = speed_direction_rt(previous_box, bbox)
                            velocity_lb = speed_direction_lb(previous_box, bbox)
                            velocity_rb = speed_direction_rb(previous_box, bbox)
                if previous_box is None:
                    previous_box = self.last_observation
                    self.velocity = speed_direction(previous_box, bbox)
                    self.velocity_lt = speed_direction_lt(previous_box, bbox)
                    self.velocity_rt = speed_direction_rt(previous_box, bbox)
                    self.velocity_lb = speed_direction_lb(previous_box, bbox)
                    self.velocity_rb = speed_direction_rb(previous_box, bbox)
                else:
                    self.velocity = velocity
                    self.velocity_lt = velocity_lt
                    self.velocity_rt = velocity_rt
                    self.velocity_lb = velocity_lb
                    self.velocity_rb = velocity_rb

            self.last_observation = bbox
            self.last_observation_save = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(convert_bbox_to_z(bbox))
            self.confidence_pre = self.confidence
            self.confidence = bbox[-1]
            self.age_recover_for_cbiou = self.age

            self._update_depth_and_height_stats(bbox)
        else:
            self.kf.update(bbox)
            self.confidence_pre = None

    def predict(self):
        if (self.kf.x[7] + self.kf.x[2]) <= 0:
            self.kf.x[7] *= 0.0

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        pred = convert_x_to_bbox(self.kf.x)
        self.history.append(pred)

        if self.depth_enable:
            pred_y2 = float(pred[0][3])
            self.depth_ema = (1.0 - self.ema_pred) * self.depth_ema + self.ema_pred * pred_y2

        # Inflate UA variance only once per frame in predict().
        self._inflate_height_uncertainty()

        # if not self.confidence_pre:
        #     return self.history[-1], np.clip(self.kf.x[3], self.args.track_thresh, 1.0), np.clip(self.confidence, 0.1, self.args.track_thresh)
        # return self.history[-1], np.clip(self.kf.x[3], self.args.track_thresh, 1.0), np.clip(self.confidence - (self.confidence_pre - self.confidence), 0.1, self.args.track_thresh)
        
        kalman_score = float(np.clip(_to_scalar(self.kf.x[3]), self.args.track_thresh, 1.0))

        if self.confidence_pre is None:
            simple_raw = self.confidence
        else:
            simple_raw = self.confidence - (self.confidence_pre - self.confidence)

        simple_score = float(np.clip(_to_scalar(simple_raw), 0.1, self.args.track_thresh))

        return self.history[-1], kalman_score, simple_score

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)


"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note
    that we hardly normalize the cost by all methods to (0,1) which may not be
    the best practice.
"""
ASSO_FUNCS = {
    "iou": iou_batch,
    "giou": giou_batch,
    "ciou": ciou_batch,
    "diou": diou_batch,
    "ct_dist": ct_dist,
    "Height_Modulated_IoU": hmiou,
}


class Hybrid_Soperm_Track(object):
    def __init__(self,
                 args,
                 det_thresh,
                 max_age=30,
                 min_hits=3,
                 iou_threshold=0.3,
                 delta_t=3,
                 asso_func="iou",
                 inertia=0.2,
                 use_byte=False,
                 # component switches
                 ua_enable=True,
                 pc_enable=True,
                 depth_enable=True,
                 # soft-depth params
                 depth_alpha=0.5,
                 depth_beta=2.0,
                 depth_gate=0.25,
                 gate_floor=0.2,
                 # UA-HMIoU params
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
                 pc_bins=24,
                 pc_ema=0.08,
                 pc_gamma=2.0,
                 pc_floor=0.3,
                 pc_min_count=2,
                 pc_sample_score=0.6,
                 pc_update_matched_only=True,
                 # depth EMA
                 ema_obs=0.5,
                 ema_pred=0.1,
                 
                 prior_w_u=0.60,
                 prior_w_p=0.50,
                 prior_w_d=0.40,
                 prior_floor=0.55,
                 
                 ):
        """
        Sets key parameters for Hybrid-SORT.

        Three added components are independent:
        - ua_enable
        - pc_enable
        - depth_enable
        """
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
        self.args = args

        # switches
        self.ua_enable = bool(ua_enable)
        self.pc_enable = bool(pc_enable)
        self.depth_enable = bool(depth_enable)

        # soft-depth
        self.depth_alpha = float(depth_alpha)
        self.depth_beta = float(depth_beta)
        self.depth_gate = float(depth_gate)
        self.gate_floor = float(gate_floor)

        # UA-HMIoU
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
        self.pc_sample_score = float(pc_sample_score)
        self.pc_update_matched_only = bool(pc_update_matched_only)
        self.perspective = PerspectiveHeightModel(
            num_bins=pc_bins,
            ema=pc_ema,
            min_count=pc_min_count,
            floor=pc_floor,
            gamma=pc_gamma,
        )

        # depth EMA
        self.ema_obs = float(ema_obs)
        self.ema_pred = float(ema_pred)

        KalmanBoxTracker.count = 0
        
        ######
        self.prior_w_u = float(prior_w_u)
        self.prior_w_p = float(prior_w_p)
        self.prior_w_d = float(prior_w_d)
        self.prior_floor = float(prior_floor)

    # -----------------------------------------------------------------------
    # Optional reset helper
    # -----------------------------------------------------------------------
    def reset(self, reset_perspective=True, reset_id_counter=False):
        self.trackers = []
        self.frame_count = 0
        if reset_perspective and self.pc_enable:
            self.perspective.reset()
        if reset_id_counter:
            KalmanBoxTracker.count = 0

    # -----------------------------------------------------------------------
    # Depth helpers
    # -----------------------------------------------------------------------
    def _depth_affinity(self, dets_xyxy, trks_xyxy, trk_depth_ema, img_h):
        Nd = dets_xyxy.shape[0]
        Nt = trks_xyxy.shape[0]
        if Nd == 0 or Nt == 0:
            return np.zeros((Nd, Nt), dtype=float)
        y2d = dets_xyxy[:, 3].astype(float)[:, None]
        y2t = np.asarray(trk_depth_ema, dtype=float)[None, :]
        norm = max(float(img_h), 1e-6)
        d = np.abs(y2d - y2t) / norm
        A = np.exp(-self.depth_beta * d)
        if self.depth_gate > 0:
            mask_far = d > self.depth_gate
            if self.gate_floor >= 0.0:
                A[mask_far] = np.maximum(A[mask_far], self.gate_floor)
        return A

    def _blend_with_depth(self, base_score, dets_xyxy, trks_xyxy, trk_depth_ema, img_h):
        if (not self.depth_enable) or self.depth_alpha <= 0:
            return base_score
        A_d = self._depth_affinity(dets_xyxy, trks_xyxy, trk_depth_ema, img_h)
        return base_score * ((1.0 - self.depth_alpha) + self.depth_alpha * A_d)

    # -----------------------------------------------------------------------
    # UA helpers
    # -----------------------------------------------------------------------
    def _ua_height_affinity(self,
                            dets_xyxy,
                            trks_xyxy,
                            trk_mu_h,
                            trk_sigma_h,
                            det_scores,
                            trk_miss,
                            img_w,
                            img_h):
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

        sigma_eff = sigma_base * S_trk
        sigma_eff = sigma_eff * S_det
        sigma_eff = np.maximum(sigma_eff, eps)

        z = np.abs(hd - mu) / (self.ua_alpha * sigma_eff + eps)
        if self.ua_mode == "gauss":
            return np.exp(-0.5 * z * z)
        return 1.0 / (1.0 + z * z)
    
    
    def _build_selective_ua_affinity(
        self,
        dets_xyxy,
        trks_xyxy,
        det_scores,
        trk_height_mu,
        trk_height_sigma,
        trk_miss,
        img_w,
        img_h,
        raw_iou,
        iou_gate,
        round_id,
    ):
        """
        方案 B：
        只在“有歧义”的 det-track pair 上启用 UA。
        清晰匹配不干预，避免破坏 Hybrid-SORT 原本的 ROCM + TCM 主匹配。

        ambiguous 条件（满足任一即可）：
        1) raw_iou < iou_gate + 0.10
        2) track 有 miss（time_since_update > 0）
        3) detection 分数不高（score < det_thresh + 0.05）

        返回：
            pair_aff in [ (1-beta), 1 ]
        其中不歧义的 pair_aff = 1.0
        """
        Nd = dets_xyxy.shape[0]
        Nt = trks_xyxy.shape[0]

        if Nd == 0 or Nt == 0:
            return np.ones((Nd, Nt), dtype=float)

        if not self.ua_enable:
            return np.ones((Nd, Nt), dtype=float)

        # 先算完整 UA affinity
        A_u = self._ua_height_affinity(
            dets_xyxy,
            trks_xyxy,
            trk_height_mu,
            trk_height_sigma,
            det_scores,
            trk_miss,
            img_w,
            img_h,
        )

        # 不同轮次给不同强度：第一轮最轻，后两轮稍强
        if round_id == 1:
            beta = 0.15
        elif round_id == 2:
            beta = 0.30
        else:
            beta = 0.30

        # pair 级歧义掩码
        ambiguous = np.zeros_like(raw_iou, dtype=bool)

        # 1) IoU 落在“模糊区”
        ambiguous |= (raw_iou < (iou_gate + 0.10))

        # 2) track 有 miss -> 这一整列都算模糊
        ambiguous |= (np.asarray(trk_miss, dtype=float)[None, :] > 0)

        # 3) det 分数不够高 -> 这一整行都算模糊
        ambiguous |= (np.asarray(det_scores, dtype=float)[:, None] < (self.det_thresh + 0.05))

        # 不模糊：pair_aff = 1
        # 模糊：pair_aff = (1-beta) + beta * A_u
        pair_aff = np.ones_like(raw_iou, dtype=float)
        pair_aff[ambiguous] = (1.0 - beta) + beta * A_u[ambiguous]

        return pair_aff
    

    # -----------------------------------------------------------------------
    # PC helpers
    # -----------------------------------------------------------------------
    def _pc_affinity(self, dets_xyxy, trks_xyxy, img_h):
        Nd = dets_xyxy.shape[0]
        Nt = trks_xyxy.shape[0]
        if Nd == 0 or Nt == 0:
            return np.zeros((Nd, Nt), dtype=float)
        if (not self.pc_enable) or (not self.perspective.ready()):
            return np.ones((Nd, Nt), dtype=float)
        return self.perspective.affinity_matrix(dets_xyxy, trks_xyxy, img_h)

    def _update_pc_model_with_det(self, det_row, img_h):
        if not self.pc_enable:
            return
        score = float(det_row[4])
        if score < self.pc_sample_score:
            return
        y2 = float(det_row[3])
        h = float(det_row[3] - det_row[1])
        if h > 0 and np.isfinite(h) and np.isfinite(y2):
            self.perspective.observe(y2, h, img_h)

    # -----------------------------------------------------------------------
    # Pair affinity builder shared by all rounds
    # -----------------------------------------------------------------------
    # def _build_pair_affinity(self,
    #                          dets_xyxy,
    #                          trks_xyxy,
    #                          det_scores,
    #                          trk_height_mu,
    #                          trk_height_sigma,
    #                          trk_miss,
    #                          trk_depth_ema,
    #                          img_w,
    #                          img_h):
    #     Nd = dets_xyxy.shape[0]
    #     Nt = trks_xyxy.shape[0]
    #     if Nd == 0 or Nt == 0:
    #         return np.zeros((Nd, Nt), dtype=float)

    #     pair_aff = np.ones((Nd, Nt), dtype=float)

    #     if self.ua_enable:
    #         pair_aff *= self._ua_height_affinity(
    #             dets_xyxy, trks_xyxy,
    #             trk_height_mu, trk_height_sigma,
    #             det_scores, trk_miss,
    #             img_w, img_h,
    #         )

    #     if self.pc_enable:
    #         pair_aff *= self._pc_affinity(dets_xyxy, trks_xyxy, img_h)

    #     if self.depth_enable:
    #         pair_aff = self._blend_with_depth(
    #             pair_aff,
    #             dets_xyxy,
    #             trks_xyxy,
    #             trk_depth_ema,
    #             img_h,
    #         )

    #     return pair_aff
    
    
    def _build_prior_components(
        self,
        dets_xyxy,
        trks_xyxy,
        det_scores,
        trk_height_mu,
        trk_height_sigma,
        trk_miss,
        trk_depth_ema,
        img_w,
        img_h,
    ):
        Nd = dets_xyxy.shape[0]
        Nt = trks_xyxy.shape[0]
        if Nd == 0 or Nt == 0:
            z = np.zeros((Nd, Nt), dtype=float)
            return z, z, z

        A_u = np.ones((Nd, Nt), dtype=float)
        A_p = np.ones((Nd, Nt), dtype=float)
        A_d = np.ones((Nd, Nt), dtype=float)

        if self.ua_enable:
            A_u = self._ua_height_affinity(
                dets_xyxy, trks_xyxy,
                trk_height_mu, trk_height_sigma,
                det_scores, trk_miss,
                img_w, img_h
            )

        # 保留 warm up：ready() 前返回全 1
        if self.pc_enable:
            A_p = self._pc_affinity(dets_xyxy, trks_xyxy, img_h)

        if self.depth_enable:
            A_d = self._depth_affinity(dets_xyxy, trks_xyxy, trk_depth_ema, img_h)

        return A_u, A_p, A_d
    
    
    def _build_residual_prior(self, A_u, A_p, A_d):
        prior = (
            1.0
            + self.prior_w_u * (A_u - 1.0)
            + self.prior_w_p * (A_p - 1.0)
            + self.prior_w_d * (A_d - 1.0)
        )
        prior = np.clip(prior, self.prior_floor, 1.0)
        return prior

    # -----------------------------------------------------------------------
    # First-round Hybrid association with augmented geometric priors
    # -----------------------------------------------------------------------
    def _associate_4_points_augmented(self,
                                      detections,
                                      trackers,
                                      iou_threshold,
                                      lt,
                                      rt,
                                      lb,
                                      rb,
                                      previous_obs,
                                      vdc_weight,
                                      iou_type=None,
                                      args=None,
                                      pair_prior=None,#   pair_affinity=None,
                                      use_tcm_score=False):
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

        Y1, X1 = speed_direction_batch_lt(detections, previous_obs)
        Y2, X2 = speed_direction_batch_rt(detections, previous_obs)
        Y3, X3 = speed_direction_batch_lb(detections, previous_obs)
        Y4, X4 = speed_direction_batch_rb(detections, previous_obs)

        cost_lt = cost_vel(Y1, X1, trackers, lt, detections, previous_obs, vdc_weight)
        cost_rt = cost_vel(Y2, X2, trackers, rt, detections, previous_obs, vdc_weight)
        cost_lb = cost_vel(Y3, X3, trackers, lb, detections, previous_obs, vdc_weight)
        cost_rb = cost_vel(Y4, X4, trackers, rb, detections, previous_obs, vdc_weight)

        # iou_matrix = iou_type(detections, trackers)
        # iou_matrix = np.array(iou_matrix)
        # geom_score = iou_matrix if pair_affinity is None else iou_matrix * pair_affinity
        
        iou_matrix = iou_type(detections, trackers)
        iou_matrix = np.array(iou_matrix)

        # 关键：prior 只修正几何项，不改 motion / TCM
        geom_score = iou_matrix if pair_prior is None else iou_matrix * pair_prior
        

        angle_diff_cost = cost_lt + cost_rt + cost_lb + cost_rb
        if use_tcm_score:
            score_dif = cal_score_dif_batch(detections, trackers)
            angle_diff_cost -= score_dif * args.TCM_first_step_weight

        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_assignment(-(geom_score + angle_diff_cost))
        else:
            matched_indices = np.empty(shape=(0, 2))

        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def update(self, output_results, img_info, img_size, frame_id=None):
        """
        Params:
          dets - a numpy array of detections in the format
                 [[x1,y1,x2,y2,score], ...]
        Requires: this method must be called once for each frame even with empty
        detections (use np.empty((0, 5)) for frames without detections).
        Returns a similar array, where the last column is the object ID.
        """
        if output_results is None:
            return np.empty((0, 5))

        self.frame_count += 1

        # unify detections to numpy
        try:
            import torch
            if isinstance(output_results, torch.Tensor):
                output_results = output_results.detach().cpu().numpy()
        except Exception:
            pass
        if not isinstance(output_results, np.ndarray):
            output_results = np.array(output_results)

        # post-process detections
        if output_results.shape[1] == 5:
            scores = output_results[:, 4].astype(float)
            bboxes = output_results[:, :4].astype(float)
        else:
            scores = (output_results[:, 4] * output_results[:, 5]).astype(float)
            bboxes = output_results[:, :4].astype(float)

        img_h, img_w = _to_scalar(img_info[0]), _to_scalar(img_info[1])
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= max(scale, 1e-6)
        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)

        inds_low = scores > 0.1
        inds_high = scores < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = dets[inds_second]
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]

        # predict trackers
        trks = np.zeros((len(self.trackers), 6))
        to_del = []
        ret = []
        # for t, trk in enumerate(trks):
        #     pos, kalman_score, simple_score = self.trackers[t].predict()
        #     try:
        #         trk[:] = [pos[0][0], pos[0][1], pos[0][2], pos[0][3], kalman_score, simple_score[0]]
        #     except Exception:
        #         trk[:] = [pos[0][0], pos[0][1], pos[0][2], pos[0][3], kalman_score, simple_score]
        #     if np.any(np.isnan(pos)):
        #         to_del.append(t)
        
        for t, trk in enumerate(trks):
            pos, kalman_score, simple_score = self.trackers[t].predict()
            try:
                trk[:] = [pos[0][0], pos[0][1], pos[0][2], pos[0][3], kalman_score, simple_score[0]]
            except Exception:
                trk[:] = [pos[0][0], pos[0][1], pos[0][2], pos[0][3], kalman_score, simple_score]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities_lt = np.array([trk.velocity_lt if trk.velocity_lt is not None else np.array((0, 0)) for trk in self.trackers])
        velocities_rt = np.array([trk.velocity_rt if trk.velocity_rt is not None else np.array((0, 0)) for trk in self.trackers])
        velocities_lb = np.array([trk.velocity_lb if trk.velocity_lb is not None else np.array((0, 0)) for trk in self.trackers])
        velocities_rb = np.array([trk.velocity_rb if trk.velocity_rb is not None else np.array((0, 0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        # extra stats aligned with self.trackers
        depth_emas = np.array([trk.depth_ema for trk in self.trackers], dtype=float) if len(self.trackers) > 0 else np.zeros((0,), dtype=float)
        height_mus_full = np.array([trk.get_height_mu_sigma()[0] for trk in self.trackers], dtype=float) if len(self.trackers) > 0 else np.zeros((0,), dtype=float)
        height_sigmas_full = np.array([trk.get_height_mu_sigma()[1] for trk in self.trackers], dtype=float) if len(self.trackers) > 0 else np.zeros((0,), dtype=float)
        miss_counts_full = np.array([trk.time_since_update for trk in self.trackers], dtype=float) if len(self.trackers) > 0 else np.zeros((0,), dtype=float)

        # -------------------------------------------------------------------
        # First round: keep Hybrid-SORT 4-point + TCM, but augment geometry
        # -------------------------------------------------------------------
        if dets.shape[0] > 0 and trks.shape[0] > 0:
            # pair_affinity_r1 = self._build_pair_affinity(
            #     dets[:, :4],
            #     trks[:, :4],
            #     dets[:, 4],
            #     height_mus_full,
            #     height_sigmas_full,
            #     miss_counts_full,
            #     depth_emas,
            #     img_w,
            #     img_h,
            # )
            
            # raw_iou_r1 = np.array(self.asso_func(dets, trks))
            # pair_affinity_r1 = self._build_selective_ua_affinity(
            #     dets[:, :4],
            #     trks[:, :4],
            #     dets[:, 4],
            #     height_mus_full,
            #     height_sigmas_full,
            #     miss_counts_full,
            #     img_w,
            #     img_h,
            #     raw_iou=raw_iou_r1,
            #     iou_gate=self.iou_threshold,
            #     round_id=1,
            # )
            
            # matched, unmatched_dets, unmatched_trks = self._associate_4_points_augmented(
            #     dets,
            #     trks,
            #     self.iou_threshold,
            #     velocities_lt,
            #     velocities_rt,
            #     velocities_lb,
            #     velocities_rb,
            #     k_observations,
            #     self.inertia,
            #     self.asso_func,
            #     self.args,
            #     pair_affinity=pair_affinity_r1,
            #     use_tcm_score=bool(getattr(self.args, "TCM_first_step", False)),
            # )
            
            A_u_r1, A_p_r1, A_d_r1 = self._build_prior_components(
                dets[:, :4],
                trks[:, :4],
                dets[:, 4],
                height_mus_full,
                height_sigmas_full,
                miss_counts_full,
                depth_emas,
                img_w,
                img_h,
            )

            pair_prior_r1 = self._build_residual_prior(A_u_r1, A_p_r1, A_d_r1)

            matched, unmatched_dets, unmatched_trks = self._associate_4_points_augmented(
                dets,
                trks,
                self.iou_threshold,
                velocities_lt,
                velocities_rt,
                velocities_lb,
                velocities_rb,
                k_observations,
                self.inertia,
                self.asso_func,
                self.args,
                pair_prior=pair_prior_r1,
                use_tcm_score=bool(getattr(self.args, "TCM_first_step", False)),
            )
            
            
        else:
            matched = np.empty((0, 2), dtype=int)
            unmatched_dets = np.arange(dets.shape[0])
            unmatched_trks = np.arange(trks.shape[0])

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
            self._update_pc_model_with_det(dets[m[0], :], img_h)

        # -------------------------------------------------------------------
        # Second round: BYTE with UA / PC / Depth, TCM-byte preserved
        # -------------------------------------------------------------------
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            geom_left = np.array(self.asso_func(dets_second, u_trks))

            if geom_left.size > 0:
                A_u_r2, A_p_r2, A_d_r2 = self._build_prior_components(
                    dets_second[:, :4],
                    u_trks[:, :4],
                    dets_second[:, 4],
                    height_mus_full[unmatched_trks],
                    height_sigmas_full[unmatched_trks],
                    miss_counts_full[unmatched_trks],
                    depth_emas[unmatched_trks],
                    img_w,
                    img_h,
                )

            pair_prior_r2 = self._build_residual_prior(A_u_r2, A_p_r2, A_d_r2)

            score_left = geom_left * pair_prior_r2

            if getattr(self.args, "TCM_byte_step", False):
                score_left -= np.array(
                    cal_score_dif_batch_two_score(dets_second, u_trks)
                    * self.args.TCM_byte_step_weight
                )

            # if geom_left.size > 0:
                # pair_affinity_r2 = self._build_pair_affinity(
                #     dets_second[:, :4],
                #     u_trks[:, :4],
                #     dets_second[:, 4],
                #     height_mus_full[unmatched_trks],
                #     height_sigmas_full[unmatched_trks],
                #     miss_counts_full[unmatched_trks],
                #     depth_emas[unmatched_trks],
                #     img_w,
                #     img_h,
                # )
            #     score_left = geom_left * pair_affinity_r2
            
            # geom_left = np.array(self.asso_func(dets_second, u_trks))
            # if geom_left.size > 0:
            #     pair_affinity_r2 = self._build_selective_ua_affinity(
            #         dets_second[:, :4],
            #         u_trks[:, :4],
            #         dets_second[:, 4],
            #         height_mus_full[unmatched_trks],
            #         height_sigmas_full[unmatched_trks],
            #         miss_counts_full[unmatched_trks],
            #         img_w,
            #         img_h,
            #         raw_iou=geom_left,
            #         iou_gate=self.iou_threshold,
            #         round_id=2,
            #     )
            #     score_left = geom_left * pair_affinity_r2
            
                
            #     # if self.args.TCM_byte_step:
            #     # if self.args.getattr(self.args, "TCM_byte_step", False):
            #     if getattr(self.args, "TCM_byte_step", False):
                    
            #         score_left -= np.array(cal_score_dif_batch_two_score(dets_second, u_trks) * self.args.TCM_byte_step_weight)
            

            if score_left.max() > self.iou_threshold:
                matched_indices = linear_assignment(-score_left)
                to_remove_trk_indices = []
                to_remove_det_indices = []
                for m in matched_indices:
                    det_local, trk_local = m[0], m[1]
                    trk_ind = unmatched_trks[trk_local]
                    if geom_left[det_local, trk_local] < self.iou_threshold:
                        continue
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

        # -------------------------------------------------------------------
        # Third round: rematch with last observations + priors
        # -------------------------------------------------------------------
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            geom_left = np.array(self.asso_func(left_dets, left_trks))

            # if geom_left.size > 0:
            #     pair_affinity_r3 = self._build_pair_affinity(
            #         left_dets[:, :4],
            #         left_trks[:, :4],
            #         left_dets[:, 4],
            #         height_mus_full[unmatched_trks],
            #         height_sigmas_full[unmatched_trks],
            #         miss_counts_full[unmatched_trks],
            #         depth_emas[unmatched_trks],
            #         img_w,
            #         img_h,
            #     )
            #     score_left = geom_left * pair_affinity_r3
                
            if geom_left.size > 0:
                # pair_affinity_r3 = self._build_selective_ua_affinity(
                #     left_dets[:, :4],
                #     left_trks[:, :4],
                #     left_dets[:, 4],
                #     height_mus_full[unmatched_trks],
                #     height_sigmas_full[unmatched_trks],
                #     miss_counts_full[unmatched_trks],
                #     img_w,
                #     img_h,
                #     raw_iou=geom_left,
                #     iou_gate=self.iou_threshold,
                #     round_id=3,
                # )
                # score_left = geom_left * pair_affinity_r3
                
                A_u_r3, A_p_r3, A_d_r3 = self._build_prior_components(
                    left_dets[:, :4],
                    left_trks[:, :4],
                    left_dets[:, 4],
                    height_mus_full[unmatched_trks],
                    height_sigmas_full[unmatched_trks],
                    miss_counts_full[unmatched_trks],
                    depth_emas[unmatched_trks],
                    img_w,
                    img_h,
                )

                pair_prior_r3 = self._build_residual_prior(A_u_r3, A_p_r3, A_d_r3)

                score_left = geom_left * pair_prior_r3
                
                

                if score_left.max() > self.iou_threshold:
                    rematched_indices = linear_assignment(-score_left)
                    to_remove_det_indices = []
                    to_remove_trk_indices = []
                    for m in rematched_indices:
                        det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                        if geom_left[m[0], m[1]] < self.iou_threshold:
                            continue
                        self.trackers[trk_ind].update(dets[det_ind, :])
                        to_remove_det_indices.append(det_ind)
                        to_remove_trk_indices.append(trk_ind)
                        self._update_pc_model_with_det(dets[det_ind, :], img_h)
                    unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices)) if len(to_remove_det_indices) > 0 else unmatched_dets
                    unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices)) if len(to_remove_trk_indices) > 0 else unmatched_trks

        # Unmatched tracks: no measurement update
        for m in unmatched_trks:
            self.trackers[m].update(None)

        # Create new trackers
        for i in unmatched_dets:
            trk = KalmanBoxTracker(
                dets[i, :],
                delta_t=self.delta_t,
                args=self.args,
                depth_enable=self.depth_enable,
                ema_obs=self.ema_obs,
                ema_pred=self.ema_pred,
                ua_enable=self.ua_enable,
                ua_h_ema=self.ua_h_ema,
                ua_h2_ema=self.ua_h2_ema,
                ua_proc_var_frac=self.ua_proc_var_frac,
                ua_sigma_floor_frac=self.ua_sigma_floor_frac,
            )
            self.trackers.append(trk)
            if self.pc_enable and (not self.pc_update_matched_only):
                self._update_pc_model_with_det(dets[i, :], img_h)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0][:4]
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
        """
        Keep public-det pipeline behavior close to original Hybrid-SORT.
        Added tracker side statistics do not alter its association policy.
        """
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

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_dets_c = left_dets.copy()
            left_trks_c = left_trks.copy()

            iou_left = np.array(self.asso_func(left_dets_c, left_trks_c))
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
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices)) if len(to_remove_det_indices) > 0 else unmatched_dets
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices)) if len(to_remove_trk_indices) > 0 else unmatched_trks

        for i in unmatched_dets:
            trk = KalmanBoxTracker(
                dets[i, :],
                delta_t=self.delta_t,
                args=self.args,
                depth_enable=self.depth_enable,
                ema_obs=self.ema_obs,
                ema_pred=self.ema_pred,
                ua_enable=self.ua_enable,
                ua_h_ema=self.ua_h_ema,
                ua_h2_ema=self.ua_h2_ema,
                ua_proc_var_frac=self.ua_proc_var_frac,
                ua_sigma_floor_frac=self.ua_sigma_floor_frac,
            )
            trk.cate = cates[i]
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() > 0:
                d = trk.last_observation[:4]
            else:
                d = trk.get_state()[0][:4]
            if trk.time_since_update < 1:
                if (self.frame_count <= self.min_hits) or (trk.hit_streak >= self.min_hits):
                    ret.append(np.concatenate((d, [trk.id + 1], [getattr(trk, "cate", 1)], [0])).reshape(1, -1))
                if trk.hit_streak == self.min_hits:
                    for prev_i in range(self.min_hits - 1):
                        prev_observation = trk.history_observations[-(prev_i + 2)]
                        ret.append((np.concatenate((prev_observation[:4], [trk.id + 1], [getattr(trk, "cate", 1)], [-(prev_i + 1)]))).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 7))




"##### -*- coding: utf-8 -*-####"
# Hybrid-SORT + conservative SOPERM-style geometric priors.

# Recommended integration strategy:
# - Round 1: motion-first, weak SHS(UA) only, and only for ambiguous/stale cases.
# - Round 2: SHS + GCD(Depth) as the main rescue cues, optional weak GPM(PC).
# - Round 3: SHS + GCD as the main rematch cues, optional weak GPM(PC).
# - GPM updates only from stable matched samples by default.
# - Priors are applied as small residual penalties rather than strong multiplicative takeover.
# - Acceptance gates still use the original geometric score, not the prior-adjusted score.

# Notes:
# - SHS == UA in this file
# - GCD == Depth in this file
# - GPM == PC in this file
# - Keep Hybrid-SORT's 4-point motion, TCM, BYTE, and rematch skeleton.
# """

# from __future__ import print_function
# import numpy as np

# from .association import *


# # ---------------------------------------------------------------------------
# # Safe conversions
# # ---------------------------------------------------------------------------

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


# # ---------------------------------------------------------------------------
# # Original helpers
# # ---------------------------------------------------------------------------

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
#     """
#     [x1,y1,x2,y2,score] -> [x,y,s,score,r]^T
#     """
#     w = bbox[2] - bbox[0]
#     h = bbox[3] - bbox[1]
#     x = bbox[0] + w / 2.0
#     y = bbox[1] + h / 2.0
#     s = w * h
#     r = w / float(h + 1e-6)
#     score = bbox[4]
#     return np.array([x, y, s, score, r]).reshape((5, 1))


# def convert_x_to_bbox(x, score=None):
#     """
#     [x,y,s,score,r] -> [x1,y1,x2,y2] or [x1,y1,x2,y2,score]
#     """
#     xx = np.asarray(x, dtype=float).reshape(-1)
#     s = max(xx[2], 0.0)
#     r = max(xx[4], 1e-12)
#     w = np.sqrt(max(s * r, 0.0))
#     h = 0.0 if w < 1e-12 else s / w
#     if score is None:
#         return np.array([xx[0] - w / 2.0, xx[1] - h / 2.0,
#                          xx[0] + w / 2.0, xx[1] + h / 2.0]).reshape((1, 4))
#     return np.array([xx[0] - w / 2.0, xx[1] - h / 2.0,
#                      xx[0] + w / 2.0, xx[1] + h / 2.0, score]).reshape((1, 5))


# def speed_direction(bbox1, bbox2):
#     cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
#     cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
#     speed = np.array([cy2 - cy1, cx2 - cx1])
#     norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
#     return speed / norm


# def speed_direction_lt(bbox1, bbox2):
#     cx1, cy1 = bbox1[0], bbox1[1]
#     cx2, cy2 = bbox2[0], bbox2[1]
#     speed = np.array([cy2 - cy1, cx2 - cx1])
#     norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
#     return speed / norm


# def speed_direction_rt(bbox1, bbox2):
#     cx1, cy1 = bbox1[0], bbox1[3]
#     cx2, cy2 = bbox2[0], bbox2[3]
#     speed = np.array([cy2 - cy1, cx2 - cx1])
#     norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
#     return speed / norm


# def speed_direction_lb(bbox1, bbox2):
#     cx1, cy1 = bbox1[2], bbox1[1]
#     cx2, cy2 = bbox2[2], bbox2[1]
#     speed = np.array([cy2 - cy1, cx2 - cx1])
#     norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
#     return speed / norm


# def speed_direction_rb(bbox1, bbox2):
#     cx1, cy1 = bbox1[2], bbox1[3]
#     cx2, cy2 = bbox2[2], bbox2[3]
#     speed = np.array([cy2 - cy1, cx2 - cx1])
#     norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
#     return speed / norm


# # ---------------------------------------------------------------------------
# # Perspective model (GPM / PC)
# # ---------------------------------------------------------------------------

# class PerspectiveHeightModel(object):
#     def __init__(self, num_bins=24, ema=0.08, min_count=2, floor=0.3, gamma=2.0, ready_count=50):
#         self.num_bins = int(num_bins)
#         self.ema = float(ema)
#         self.min_count = int(min_count)
#         self.floor = float(floor)
#         self.gamma = float(gamma)
#         self.ready_count = int(ready_count)
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
#         if (not np.isfinite(y2_norm)) or (not np.isfinite(h_pixel)) or h_pixel <= 0:
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
#         y2_pixels = np.asarray(y2_pixels, dtype=float)
#         if img_h <= 0 or len(y2_pixels) == 0:
#             return np.zeros_like(y2_pixels, dtype=float)

#         y2n = np.clip(y2_pixels / float(img_h), 0.0, 1.0 - 1e-8)
#         bins = np.floor(y2n * self.num_bins).astype(int)
#         pos = y2n * self.num_bins
#         frac = pos - np.floor(pos)
#         b0 = bins
#         b1 = np.clip(bins + 1, 0, self.num_bins - 1)

#         m0, m1 = self.bin_means[b0], self.bin_means[b1]
#         c0, c1 = self.bin_counts[b0], self.bin_counts[b1]
#         use0 = c0 >= self.min_count
#         use1 = c1 >= self.min_count
#         m0_eff = np.where(use0, m0, np.where(use1, m1, self.global_ema))
#         m1_eff = np.where(use1, m1, np.where(use0, m0, self.global_ema))
#         h_star = (1.0 - frac) * m0_eff + frac * m1_eff
#         return np.maximum(h_star, 4.0)

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
#         if self.floor > 0.0:
#             A_p = np.maximum(A_p, self.floor)
#         return A_p

#     def ready(self):
#         return self.global_count >= self.ready_count


# # ---------------------------------------------------------------------------
# # Kalman tracker with Hybrid-SORT backbone + SOPERM side stats
# # ---------------------------------------------------------------------------

# class KalmanBoxTracker(object):
#     count = 0

#     def __init__(self,
#                  bbox,
#                  delta_t=3,
#                  orig=False,
#                  args=None,
#                  depth_enable=True,
#                  ema_obs=0.5,
#                  ema_pred=0.1,
#                  ua_enable=True,
#                  ua_h_ema=0.5,
#                  ua_h2_ema=0.3,
#                  ua_proc_var_frac=0.02,
#                  ua_sigma_floor_frac=0.05):
#         if not orig:
#             from .kalmanfilter_score_new import KalmanFilterNew_score_new as KalmanFilter_score_new
#             self.kf = KalmanFilter_score_new(dim_x=9, dim_z=5)
#         else:
#             from filterpy.kalman import KalmanFilter
#             self.kf = KalmanFilter(dim_x=7, dim_z=4)

#         # u, v, s, c, r, ~u, ~v, ~s, ~c
#         self.kf.F = np.array([
#             [1, 0, 0, 0, 0, 1, 0, 0, 0],
#             [0, 1, 0, 0, 0, 0, 1, 0, 0],
#             [0, 0, 1, 0, 0, 0, 0, 1, 0],
#             [0, 0, 0, 1, 0, 0, 0, 0, 1],
#             [0, 0, 0, 0, 1, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 1, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 1, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 1, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 1],
#         ])
#         self.kf.H = np.array([
#             [1, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 1, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 1, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 1, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 1, 0, 0, 0, 0],
#         ])

#         self.kf.R[2:, 2:] *= 10.0
#         self.kf.P[5:, 5:] *= 1000.0
#         self.kf.P *= 10.0
#         self.kf.Q[-1, -1] *= 0.01
#         self.kf.Q[-2, -2] *= 0.01
#         self.kf.Q[5:, 5:] *= 0.01
#         self.kf.x[:5] = convert_bbox_to_z(bbox)

#         self.time_since_update = 0
#         self.id = KalmanBoxTracker.count
#         KalmanBoxTracker.count += 1
#         self.history = []
#         self.hits = 0
#         self.hit_streak = 0
#         self.age = 0
#         self.age_recover_for_cbiou = 0

#         self.last_observation = np.array([-1, -1, -1, -1, -1])
#         self.last_observation_save = np.array([-1, -1, -1, -1, -1])
#         self.observations = dict()
#         self.history_observations = []

#         self.velocity = None
#         self.velocity_lt = None
#         self.velocity_rt = None
#         self.velocity_lb = None
#         self.velocity_rb = None
#         self.delta_t = delta_t
#         self.confidence_pre = None
#         self.confidence = float(bbox[-1])
#         self.args = args
#         self.kf.args = args

#         # SOPERM side stats
#         self.depth_enable = bool(depth_enable)
#         self.ema_obs = float(ema_obs)
#         self.ema_pred = float(ema_pred)
#         self.depth_ema = float(bbox[3])

#         self.ua_enable = bool(ua_enable)
#         self.ua_h_ema = float(ua_h_ema)
#         self.ua_h2_ema = float(ua_h2_ema)
#         self.ua_proc_var_frac = float(ua_proc_var_frac)
#         self.ua_sigma_floor_frac = float(ua_sigma_floor_frac)
#         h0 = float(max(bbox[3] - bbox[1], 1e-6))
#         self.h_m1 = h0
#         self.h_m2 = h0 * h0
#         floor_var = (self.ua_sigma_floor_frac * h0) ** 2
#         self.h_var = max(self.h_m2 - self.h_m1 * self.h_m1, floor_var)

#     def get_height_mu_sigma(self):
#         mu = float(self.h_m1)
#         floor_var = (self.ua_sigma_floor_frac * max(mu, 1e-6)) ** 2
#         var = max(float(self.h_var), floor_var)
#         sigma = np.sqrt(var)
#         return mu, sigma

#     def _inflate_height_uncertainty(self):
#         if not self.ua_enable:
#             return
#         mu = max(float(self.h_m1), 1e-6)
#         self.h_var = float(self.h_var) + (self.ua_proc_var_frac * mu) ** 2

#     def _update_depth_and_height_stats(self, bbox):
#         if bbox is None:
#             return
#         if self.depth_enable:
#             y2 = float(bbox[3])
#             self.depth_ema = (1.0 - self.ema_obs) * self.depth_ema + self.ema_obs * y2
#         if self.ua_enable:
#             h = float(max(bbox[3] - bbox[1], 1e-6))
#             self.h_m1 = (1.0 - self.ua_h_ema) * self.h_m1 + self.ua_h_ema * h
#             self.h_m2 = (1.0 - self.ua_h2_ema) * self.h_m2 + self.ua_h2_ema * (h * h)
#             raw_var = max(self.h_m2 - self.h_m1 * self.h_m1, 0.0)
#             floor_var = (self.ua_sigma_floor_frac * max(self.h_m1, 1e-6)) ** 2
#             self.h_var = 0.5 * self.h_var + 0.5 * max(raw_var, floor_var)

#     def update(self, bbox):
#         velocity = None
#         velocity_lt = None
#         velocity_rt = None
#         velocity_lb = None
#         velocity_rb = None

#         if bbox is not None:
#             bbox = np.asarray(bbox, dtype=float)
#             if self.last_observation.sum() >= 0:
#                 previous_box = None
#                 for i in range(self.delta_t):
#                     if self.age - i - 1 in self.observations:
#                         previous_box = self.observations[self.age - i - 1]
#                         if velocity_lt is not None:
#                             velocity += speed_direction(previous_box, bbox)
#                             velocity_lt += speed_direction_lt(previous_box, bbox)
#                             velocity_rt += speed_direction_rt(previous_box, bbox)
#                             velocity_lb += speed_direction_lb(previous_box, bbox)
#                             velocity_rb += speed_direction_rb(previous_box, bbox)
#                         else:
#                             velocity = speed_direction(previous_box, bbox)
#                             velocity_lt = speed_direction_lt(previous_box, bbox)
#                             velocity_rt = speed_direction_rt(previous_box, bbox)
#                             velocity_lb = speed_direction_lb(previous_box, bbox)
#                             velocity_rb = speed_direction_rb(previous_box, bbox)
#                 if previous_box is None:
#                     previous_box = self.last_observation
#                     self.velocity = speed_direction(previous_box, bbox)
#                     self.velocity_lt = speed_direction_lt(previous_box, bbox)
#                     self.velocity_rt = speed_direction_rt(previous_box, bbox)
#                     self.velocity_lb = speed_direction_lb(previous_box, bbox)
#                     self.velocity_rb = speed_direction_rb(previous_box, bbox)
#                 else:
#                     self.velocity = velocity
#                     self.velocity_lt = velocity_lt
#                     self.velocity_rt = velocity_rt
#                     self.velocity_lb = velocity_lb
#                     self.velocity_rb = velocity_rb

#             self.last_observation = bbox
#             self.last_observation_save = bbox
#             self.observations[self.age] = bbox
#             self.history_observations.append(bbox)

#             self.time_since_update = 0
#             self.history = []
#             self.hits += 1
#             self.hit_streak += 1
#             self.kf.update(convert_bbox_to_z(bbox))
#             self.confidence_pre = self.confidence
#             self.confidence = float(bbox[-1])
#             self.age_recover_for_cbiou = self.age
#             self._update_depth_and_height_stats(bbox)
#         else:
#             self.kf.update(bbox)
#             self.confidence_pre = None

#     def predict(self):
#         if (_to_scalar(self.kf.x[7]) + _to_scalar(self.kf.x[2])) <= 0:
#             self.kf.x[7] *= 0.0

#         self.kf.predict()
#         self.age += 1
#         if self.time_since_update > 0:
#             self.hit_streak = 0
#         self.time_since_update += 1

#         pred = convert_x_to_bbox(self.kf.x)
#         self.history.append(pred)

#         if self.depth_enable:
#             pred_y2 = float(pred[0][3])
#             self.depth_ema = (1.0 - self.ema_pred) * self.depth_ema + self.ema_pred * pred_y2

#         # Inflate once per frame here, not again in update(None).
#         self._inflate_height_uncertainty()

#         track_thresh = float(getattr(self.args, 'track_thresh', 0.3)) if self.args is not None else 0.3
#         kalman_score = float(np.clip(_to_scalar(self.kf.x[3]), track_thresh, 1.0))
#         if self.confidence_pre is None:
#             simple_raw = self.confidence
#         else:
#             simple_raw = self.confidence - (self.confidence_pre - self.confidence)
#         simple_score = float(np.clip(_to_scalar(simple_raw), 0.1, track_thresh))
#         return self.history[-1], kalman_score, simple_score

#     def get_state(self):
#         return convert_x_to_bbox(self.kf.x)


# ASSO_FUNCS = {
#     'iou': iou_batch,
#     'giou': giou_batch,
#     'ciou': ciou_batch,
#     'diou': diou_batch,
#     'ct_dist': ct_dist,
#     'Height_Modulated_IoU': hmiou,
# }


# class Hybrid_Soperm_Track(object):
#     def __init__(self,
#                  args,
#                  det_thresh,
#                  max_age=30,
#                  min_hits=3,
#                  iou_threshold=0.3,
#                  delta_t=3,
#                  asso_func='iou',
#                  inertia=0.2,
#                  use_byte=False,
#                  # component switches
#                  ua_enable=True,
#                  pc_enable=True,
#                  depth_enable=True,
#                  # original SOPERM hyperparams (kept)
#                  depth_alpha=0.5,
#                  depth_beta=2.0,
#                  depth_gate=0.25,
#                  gate_floor=0.2,
#                  ua_mode='cauchy',
#                  ua_alpha=1.0,
#                  ua_border_margin_frac=0.02,
#                  ua_border_boost=0.6,
#                  ua_miss_boost=0.6,
#                  ua_low_score=0.4,
#                  ua_score_boost=0.3,
#                  ua_sigma_floor_frac=0.05,
#                  ua_proc_var_frac=0.02,
#                  ua_h_ema=0.5,
#                  ua_h2_ema=0.3,
#                  pc_bins=24,
#                  pc_ema=0.08,
#                  pc_gamma=2.0,
#                  pc_floor=0.3,
#                  pc_min_count=2,
#                  pc_sample_score=0.6,
#                  ema_obs=0.5,
#                  ema_pred=0.1,
#                  # recommended conservative fusion weights
                 
#                 #  r1_ua_lambda=0.12,
#                 #  r1_depth_lambda=0.00,
#                 #  r1_pc_lambda=0.00,
#                 #  r1_geom_margin=0.12,
#                 #  r2_ua_lambda=0.28,
#                 #  r2_depth_lambda=0.22,
#                 #  r2_pc_lambda=0.08,
#                 #  r3_ua_lambda=0.30,
#                 #  r3_depth_lambda=0.24,
#                 #  r3_pc_lambda=0.08,
                
#                 # stronger fusion weights
#                 r1_ua_lambda=0.22,
#                 r1_depth_lambda=0.18,
#                 r1_pc_lambda=0.22,
#                 r1_geom_margin=0.12,   # 这个先留着，虽然 round1 不再用 trigger 了

#                 r2_ua_lambda=0.28,
#                 r2_depth_lambda=0.22,
#                 r2_pc_lambda=0.18,

#                 r3_ua_lambda=0.30,
#                 r3_depth_lambda=0.24,
#                 r3_pc_lambda=0.18,
                 
                 
                 
#                  min_pair_factor=0.45,
#                  max_pair_factor=1.02,
#                  pc_update_matched_only=True,
#                  pc_track_min_hits=3,
#                  pc_border_skip=True,
#                  pc_warmup_count=50):
#         self.args = args
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

#         # Robust args cache
#         self.tcm_first_step = bool(getattr(self.args, 'TCM_first_step', False))
#         self.tcm_first_step_weight = float(getattr(self.args, 'TCM_first_step_weight', 1.0))
#         self.tcm_byte_step = bool(getattr(self.args, 'TCM_byte_step', False))
#         self.tcm_byte_step_weight = float(getattr(self.args, 'TCM_byte_step_weight', 1.0))

#         # switches
#         self.ua_enable = bool(ua_enable)
#         self.pc_enable = bool(pc_enable)
#         self.depth_enable = bool(depth_enable)

#         # keep original hyperparams
#         self.depth_alpha = float(depth_alpha)
#         self.depth_beta = float(depth_beta)
#         self.depth_gate = float(depth_gate)
#         self.gate_floor = float(gate_floor)
#         self.ua_mode = str(ua_mode).lower()
#         assert self.ua_mode in ['cauchy', 'gauss']
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
#         self.pc_sample_score = float(pc_sample_score)
#         self.ema_obs = float(ema_obs)
#         self.ema_pred = float(ema_pred)

#         # recommended conservative fusion params
#         self.r1_ua_lambda = float(r1_ua_lambda)
#         self.r1_depth_lambda = float(r1_depth_lambda)
#         self.r1_pc_lambda = float(r1_pc_lambda)
#         self.r1_geom_margin = float(r1_geom_margin)
#         self.r2_ua_lambda = float(r2_ua_lambda)
#         self.r2_depth_lambda = float(r2_depth_lambda)
#         self.r2_pc_lambda = float(r2_pc_lambda)
#         self.r3_ua_lambda = float(r3_ua_lambda)
#         self.r3_depth_lambda = float(r3_depth_lambda)
#         self.r3_pc_lambda = float(r3_pc_lambda)
#         self.min_pair_factor = float(min_pair_factor)
#         self.max_pair_factor = float(max_pair_factor)
#         self.pc_update_matched_only = bool(pc_update_matched_only)
#         self.pc_track_min_hits = int(pc_track_min_hits)
#         self.pc_border_skip = bool(pc_border_skip)

#         self.perspective = PerspectiveHeightModel(
#             num_bins=pc_bins,
#             ema=pc_ema,
#             min_count=pc_min_count,
#             floor=pc_floor,
#             gamma=pc_gamma,
#             ready_count=pc_warmup_count,
#         )

#         KalmanBoxTracker.count = 0

#     # -----------------------------------------------------------------------
#     # Reset helper
#     # -----------------------------------------------------------------------
#     def reset(self, reset_perspective=True, reset_id_counter=False):
#         self.trackers = []
#         self.frame_count = 0
#         if reset_perspective and self.pc_enable:
#             self.perspective.reset()
#         if reset_id_counter:
#             KalmanBoxTracker.count = 0

#     # -----------------------------------------------------------------------
#     # UA / SHS affinity
#     # -----------------------------------------------------------------------
#     def _ua_height_affinity(self,
#                             dets_xyxy,
#                             trks_xyxy,
#                             trk_mu_h,
#                             trk_sigma_h,
#                             det_scores,
#                             trk_miss,
#                             img_w,
#                             img_h):
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

#         m = max(1.0, min(float(img_w), float(img_h)) * self.ua_border_margin_frac)
#         x1 = dets_xyxy[:, 0].astype(float)
#         y1 = dets_xyxy[:, 1].astype(float)
#         x2 = dets_xyxy[:, 2].astype(float)
#         y2 = dets_xyxy[:, 3].astype(float)
#         min_dist = np.minimum(np.minimum(x1, y1), np.minimum(img_w - x2, img_h - y2))
#         closeness = np.clip(1.0 - (min_dist / m), 0.0, 1.0)
#         sc = det_scores.astype(float)
#         low_score_gap = np.clip((self.ua_low_score - sc) / max(self.ua_low_score, 1e-6), 0.0, 1.0)
#         S_det = 1.0 + self.ua_border_boost * closeness + self.ua_score_boost * low_score_gap
#         S_det = S_det[:, None]

#         sigma_eff = np.maximum(sigma_base * S_trk * S_det, eps)
#         z = np.abs(hd - mu) / (self.ua_alpha * sigma_eff + eps)
#         if self.ua_mode == 'gauss':
#             return np.exp(-0.5 * z * z)
#         return 1.0 / (1.0 + z * z)

#     # -----------------------------------------------------------------------
#     # Depth / GCD affinity
#     # -----------------------------------------------------------------------
#     def _depth_affinity(self, dets_xyxy, trks_xyxy, trk_depth_ema, img_h):
#         Nd = dets_xyxy.shape[0]
#         Nt = trks_xyxy.shape[0]
#         if Nd == 0 or Nt == 0:
#             return np.zeros((Nd, Nt), dtype=float)
#         y2d = dets_xyxy[:, 3].astype(float)[:, None]
#         y2t = np.asarray(trk_depth_ema, dtype=float)[None, :]
#         norm = max(float(img_h), 1e-6)
#         d = np.abs(y2d - y2t) / norm
#         A = np.exp(-self.depth_beta * d)
#         if self.depth_gate > 0:
#             far_mask = d > self.depth_gate
#             A[far_mask] = np.maximum(A[far_mask], self.gate_floor)
#         return A

#     # -----------------------------------------------------------------------
#     # PC / GPM affinity
#     # -----------------------------------------------------------------------
#     def _pc_affinity(self, dets_xyxy, trks_xyxy, img_h):
#         Nd = dets_xyxy.shape[0]
#         Nt = trks_xyxy.shape[0]
#         if Nd == 0 or Nt == 0:
#             return np.zeros((Nd, Nt), dtype=float)
#         if (not self.pc_enable) or (not self.perspective.ready()):
#             return np.ones((Nd, Nt), dtype=float)
#         return self.perspective.affinity_matrix(dets_xyxy, trks_xyxy, img_h)

#     # -----------------------------------------------------------------------
#     # Conservative residual fusion
#     # -----------------------------------------------------------------------
#     def _residual_factor(self, A_u=None, A_d=None, A_p=None, lam_u=0.0, lam_d=0.0, lam_p=0.0):
#         # Priors act as mild penalties / small corrections, not hard takeover.
#         factor = 1.0
#         if A_u is not None and lam_u > 0:
#             factor = factor + lam_u * (A_u - 1.0)
#         if A_d is not None and lam_d > 0:
#             factor = factor + lam_d * (A_d - 1.0)
#         if A_p is not None and lam_p > 0:
#             factor = factor + lam_p * (A_p - 1.0)
#         return np.clip(factor, self.min_pair_factor, self.max_pair_factor)

#     def _build_round_factor(self,
#                             round_id,
#                             geom,
#                             dets_xyxy,
#                             trks_xyxy,
#                             det_scores,
#                             trk_height_mu,
#                             trk_height_sigma,
#                             trk_miss,
#                             trk_depth_ema,
#                             img_w,
#                             img_h):
#         Nd = dets_xyxy.shape[0]
#         Nt = trks_xyxy.shape[0]
#         if Nd == 0 or Nt == 0:
#             return np.zeros((Nd, Nt), dtype=float)

#         A_u = None
#         A_d = None
#         A_p = None

#         if self.ua_enable:
#             A_u = self._ua_height_affinity(
#                 dets_xyxy, trks_xyxy,
#                 trk_height_mu, trk_height_sigma,
#                 det_scores, trk_miss,
#                 img_w, img_h,
#             )
#         if self.depth_enable:
#             A_d = self._depth_affinity(dets_xyxy, trks_xyxy, trk_depth_ema, img_h)
#         if self.pc_enable:
#             A_p = self._pc_affinity(dets_xyxy, trks_xyxy, img_h)

#         if round_id == 1:
#             # motion-first: weak SHS only, triggered for ambiguous geom or stale tracks.
#             # stale_mask = (np.asarray(trk_miss, dtype=float)[None, :] > 0.0)
#             # amb_mask = geom < (self.iou_threshold + self.r1_geom_margin)
#             # trigger = np.logical_or(stale_mask, amb_mask)
#             # factor = np.ones_like(geom, dtype=float)
#             # if A_u is not None and self.r1_ua_lambda > 0:
#             #     weak = self._residual_factor(A_u=A_u, lam_u=self.r1_ua_lambda)
#             #     factor = np.where(trigger, weak, factor)
#             # if A_d is not None and self.r1_depth_lambda > 0:
#             #     weak = self._residual_factor(A_d=A_d, lam_d=self.r1_depth_lambda)
#             #     factor = np.where(trigger, factor * weak, factor)
#             # if A_p is not None and self.r1_pc_lambda > 0:
#             #     weak = self._residual_factor(A_p=A_p, lam_p=self.r1_pc_lambda)
#             #     factor = np.where(trigger, factor * weak, factor)
#             # return np.clip(factor, self.min_pair_factor, self.max_pair_factor)

#         # Full participation in round-1: SHS + GCD + GPM all join.
#             return self._residual_factor(
#                 A_u=A_u, A_d=A_d, A_p=A_p,
#                 lam_u=self.r1_ua_lambda,
#                 lam_d=self.r1_depth_lambda,
#                 lam_p=self.r1_pc_lambda,
#             )
            

#         if round_id == 2:
#             # BYTE rescue: SHS + GCD main, weak GPM optional
#             return self._residual_factor(
#                 A_u=A_u, A_d=A_d, A_p=A_p,
#                 lam_u=self.r2_ua_lambda,
#                 lam_d=self.r2_depth_lambda,
#                 lam_p=self.r2_pc_lambda,
#             )

#         # round_id == 3
#         return self._residual_factor(
#             A_u=A_u, A_d=A_d, A_p=A_p,
#             lam_u=self.r3_ua_lambda,
#             lam_d=self.r3_depth_lambda,
#             lam_p=self.r3_pc_lambda,
#         )

#     # -----------------------------------------------------------------------
#     # Stable GPM updater
#     # -----------------------------------------------------------------------
#     def _update_pc_model_with_det(self, det_row, img_h, trk_obj=None, img_w=None):
#         if not self.pc_enable:
#             return
#         if not self.perspective.ready() and self.frame_count < self.perspective.ready_count // 2:
#             # allow warmup accumulation, but still be selective
#             pass
#         if trk_obj is not None and trk_obj.hit_streak < self.pc_track_min_hits:
#             return
#         score = float(det_row[4])
#         if score < self.pc_sample_score:
#             return
#         if self.pc_border_skip and img_w is not None and img_h is not None:
#             x1, y1, x2, y2 = map(float, det_row[:4])
#             m = max(1.0, min(float(img_w), float(img_h)) * self.ua_border_margin_frac)
#             min_dist = min(x1, y1, float(img_w) - x2, float(img_h) - y2)
#             if min_dist < m:
#                 return
#         y2 = float(det_row[3])
#         h = float(det_row[3] - det_row[1])
#         if h > 0 and np.isfinite(h) and np.isfinite(y2):
#             self.perspective.observe(y2, h, img_h)

#     # -----------------------------------------------------------------------
#     # Hybrid 4-point first-round association with conservative priors
#     # -----------------------------------------------------------------------
#     def _associate_4_points_conservative(self,
#                                          detections,
#                                          trackers,
#                                          iou_threshold,
#                                          lt,
#                                          rt,
#                                          lb,
#                                          rb,
#                                          previous_obs,
#                                          vdc_weight,
#                                          iou_type,
#                                          pair_factor,
#                                          use_tcm_score=False):
#         if len(trackers) == 0:
#             return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

#         Y1, X1 = speed_direction_batch_lt(detections, previous_obs)
#         Y2, X2 = speed_direction_batch_rt(detections, previous_obs)
#         Y3, X3 = speed_direction_batch_lb(detections, previous_obs)
#         Y4, X4 = speed_direction_batch_rb(detections, previous_obs)

#         cost_lt = cost_vel(Y1, X1, trackers, lt, detections, previous_obs, vdc_weight)
#         cost_rt = cost_vel(Y2, X2, trackers, rt, detections, previous_obs, vdc_weight)
#         cost_lb = cost_vel(Y3, X3, trackers, lb, detections, previous_obs, vdc_weight)
#         cost_rb = cost_vel(Y4, X4, trackers, rb, detections, previous_obs, vdc_weight)
#         angle_diff_cost = cost_lt + cost_rt + cost_lb + cost_rb

#         iou_matrix = np.array(iou_type(detections, trackers))
#         geom_adj = iou_matrix * pair_factor
#         score_matrix = geom_adj + angle_diff_cost

#         if use_tcm_score:
#             score_dif = np.array(cal_score_dif_batch(detections, trackers))
#             score_matrix = score_matrix - score_dif * self.tcm_first_step_weight

#         if min(iou_matrix.shape) > 0:
#             a = (iou_matrix > iou_threshold).astype(np.int32)
#             if a.sum(1).max() == 1 and a.sum(0).max() == 1:
#                 matched_indices = np.stack(np.where(a), axis=1)
#             else:
#                 matched_indices = linear_assignment(-score_matrix)
#         else:
#             matched_indices = np.empty(shape=(0, 2), dtype=int)

#         unmatched_detections = []
#         for d in range(len(detections)):
#             if d not in matched_indices[:, 0]:
#                 unmatched_detections.append(d)
#         unmatched_trackers = []
#         for t in range(len(trackers)):
#             if t not in matched_indices[:, 1]:
#                 unmatched_trackers.append(t)

#         matches = []
#         for m in matched_indices:
#             # gate still uses ORIGINAL geometry
#             if iou_matrix[m[0], m[1]] < iou_threshold:
#                 unmatched_detections.append(m[0])
#                 unmatched_trackers.append(m[1])
#             else:
#                 matches.append(m.reshape(1, 2))

#         if len(matches) == 0:
#             matches = np.empty((0, 2), dtype=int)
#         else:
#             matches = np.concatenate(matches, axis=0)

#         return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

#     # -----------------------------------------------------------------------
#     # Main update
#     # -----------------------------------------------------------------------
#     def update(self, output_results, img_info, img_size, frame_id=None):
#         if output_results is None:
#             return np.empty((0, 5))

#         self.frame_count += 1
#         try:
#             import torch
#             if isinstance(output_results, torch.Tensor):
#                 output_results = output_results.detach().cpu().numpy()
#         except Exception:
#             pass
#         if not isinstance(output_results, np.ndarray):
#             output_results = np.array(output_results)

#         # parse detections
#         if output_results.size == 0:
#             scores = np.zeros((0,), dtype=float)
#             bboxes = np.zeros((0, 4), dtype=float)
#         elif output_results.shape[1] == 5:
#             scores = output_results[:, 4].astype(float)
#             bboxes = output_results[:, :4].astype(float)
#         else:
#             scores = (output_results[:, 4] * output_results[:, 5]).astype(float)
#             bboxes = output_results[:, :4].astype(float)

#         img_h, img_w = _to_scalar(img_info[0]), _to_scalar(img_info[1])
#         scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
#         if len(bboxes) > 0:
#             bboxes = bboxes / max(scale, 1e-6)
#         dets_all = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1) if len(bboxes) > 0 else np.empty((0, 5))

#         inds_low = scores > 0.1
#         inds_high = scores < self.det_thresh
#         inds_second = np.logical_and(inds_low, inds_high)
#         dets_second = dets_all[inds_second]
#         remain_inds = scores > self.det_thresh
#         dets = dets_all[remain_inds]

#         # predict all trackers
#         trks = np.zeros((len(self.trackers), 6), dtype=float)
#         to_del = []
#         ret = []
#         for t, trk in enumerate(trks):
#             pos, kalman_score, simple_score = self.trackers[t].predict()
#             pos = np.asarray(pos, dtype=float)
#             trk[:] = [float(pos[0][0]), float(pos[0][1]), float(pos[0][2]), float(pos[0][3]),
#                       float(kalman_score), float(simple_score)]
#             if np.any(np.isnan(pos)):
#                 to_del.append(t)
#         trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#         for t in reversed(to_del):
#             self.trackers.pop(t)

#         velocities_lt = np.array([trk.velocity_lt if trk.velocity_lt is not None else np.array((0, 0)) for trk in self.trackers])
#         velocities_rt = np.array([trk.velocity_rt if trk.velocity_rt is not None else np.array((0, 0)) for trk in self.trackers])
#         velocities_lb = np.array([trk.velocity_lb if trk.velocity_lb is not None else np.array((0, 0)) for trk in self.trackers])
#         velocities_rb = np.array([trk.velocity_rb if trk.velocity_rb is not None else np.array((0, 0)) for trk in self.trackers])
#         last_boxes = np.array([trk.last_observation for trk in self.trackers]) if len(self.trackers) > 0 else np.empty((0, 5))
#         k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers]) if len(self.trackers) > 0 else np.empty((0, 5))

#         depth_emas = np.array([trk.depth_ema for trk in self.trackers], dtype=float) if len(self.trackers) > 0 else np.zeros((0,), dtype=float)
#         height_mus = np.array([trk.get_height_mu_sigma()[0] for trk in self.trackers], dtype=float) if len(self.trackers) > 0 else np.zeros((0,), dtype=float)
#         height_sigmas = np.array([trk.get_height_mu_sigma()[1] for trk in self.trackers], dtype=float) if len(self.trackers) > 0 else np.zeros((0,), dtype=float)
#         miss_counts = np.array([trk.time_since_update for trk in self.trackers], dtype=float) if len(self.trackers) > 0 else np.zeros((0,), dtype=float)

#         # -------------------------------------------------------------------
#         # Round 1: Hybrid motion-first + weak SHS only on ambiguous/stale pairs
#         # -------------------------------------------------------------------
#         if dets.shape[0] > 0 and trks.shape[0] > 0:
#             geom_r1 = np.array(self.asso_func(dets, trks))
#             factor_r1 = self._build_round_factor(
#                 1, geom_r1,
#                 dets[:, :4], trks[:, :4], dets[:, 4],
#                 height_mus, height_sigmas, miss_counts, depth_emas,
#                 img_w, img_h,
#             )
#             matched, unmatched_dets, unmatched_trks = self._associate_4_points_conservative(
#                 dets, trks, self.iou_threshold,
#                 velocities_lt, velocities_rt, velocities_lb, velocities_rb,
#                 k_observations, self.inertia,
#                 self.asso_func,
#                 factor_r1,
#                 use_tcm_score=self.tcm_first_step,
#             )
#         else:
#             matched = np.empty((0, 2), dtype=int)
#             unmatched_dets = np.arange(dets.shape[0])
#             unmatched_trks = np.arange(trks.shape[0])

#         for m in matched:
#             det_idx, trk_idx = int(m[0]), int(m[1])
#             self.trackers[trk_idx].update(dets[det_idx, :])
#             self._update_pc_model_with_det(dets[det_idx, :], img_h, self.trackers[trk_idx], img_w)

#         # -------------------------------------------------------------------
#         # Round 2: BYTE rescue, SHS + GCD main, weak GPM
#         # -------------------------------------------------------------------
#         if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
#             u_trks = trks[unmatched_trks]
#             geom_left = np.array(self.asso_func(dets_second, u_trks))
#             if geom_left.size > 0:
#                 factor_r2 = self._build_round_factor(
#                     2, geom_left,
#                     dets_second[:, :4], u_trks[:, :4], dets_second[:, 4],
#                     height_mus[unmatched_trks], height_sigmas[unmatched_trks],
#                     miss_counts[unmatched_trks], depth_emas[unmatched_trks],
#                     img_w, img_h,
#                 )
#                 score_left = geom_left * factor_r2
#                 if self.tcm_byte_step:
#                     score_left -= np.array(cal_score_dif_batch_two_score(dets_second, u_trks)) * self.tcm_byte_step_weight

#                 thr2 = max(0.1, self.iou_threshold - 0.05)
#                 if score_left.max() > thr2:
#                     matched_indices = linear_assignment(-score_left)
#                     to_remove_trk_indices = []
#                     to_remove_det_indices = []
#                     for m in matched_indices:
#                         det_local, trk_local = int(m[0]), int(m[1])
#                         trk_ind = int(unmatched_trks[trk_local])
#                         if geom_left[det_local, trk_local] < thr2:
#                             continue
#                         self.trackers[trk_ind].update(dets_second[det_local, :])
#                         to_remove_trk_indices.append(trk_ind)
#                         to_remove_det_indices.append(det_local)
#                         self._update_pc_model_with_det(dets_second[det_local, :], img_h, self.trackers[trk_ind], img_w)
#                     if len(to_remove_trk_indices) > 0:
#                         unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))
#                     if len(to_remove_det_indices) > 0:
#                         keep_mask = np.ones(len(dets_second), dtype=bool)
#                         keep_mask[to_remove_det_indices] = False
#                         dets_second = dets_second[keep_mask]

#         # -------------------------------------------------------------------
#         # Round 3: rematch, SHS + GCD main, weak GPM
#         # -------------------------------------------------------------------
#         if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
#             left_dets = dets[unmatched_dets]
#             left_trks = last_boxes[unmatched_trks]
#             geom_left = np.array(self.asso_func(left_dets, left_trks))
#             if geom_left.size > 0:
#                 factor_r3 = self._build_round_factor(
#                     3, geom_left,
#                     left_dets[:, :4], left_trks[:, :4], left_dets[:, 4],
#                     height_mus[unmatched_trks], height_sigmas[unmatched_trks],
#                     miss_counts[unmatched_trks], depth_emas[unmatched_trks],
#                     img_w, img_h,
#                 )
#                 score_left = geom_left * factor_r3
#                 thr3 = max(0.1, self.iou_threshold - 0.05)
#                 if score_left.max() > thr3:
#                     rematched_indices = linear_assignment(-score_left)
#                     to_remove_det_indices = []
#                     to_remove_trk_indices = []
#                     for m in rematched_indices:
#                         det_local, trk_local = int(m[0]), int(m[1])
#                         det_ind = int(unmatched_dets[det_local])
#                         trk_ind = int(unmatched_trks[trk_local])
#                         if geom_left[det_local, trk_local] < thr3:
#                             continue
#                         self.trackers[trk_ind].update(dets[det_ind, :])
#                         to_remove_det_indices.append(det_ind)
#                         to_remove_trk_indices.append(trk_ind)
#                         self._update_pc_model_with_det(dets[det_ind, :], img_h, self.trackers[trk_ind], img_w)
#                     if len(to_remove_det_indices) > 0:
#                         unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
#                     if len(to_remove_trk_indices) > 0:
#                         unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

#         # No-measurement update for unmatched tracks
#         for m in unmatched_trks:
#             self.trackers[int(m)].update(None)

#         # New trackers from unmatched high-score detections
#         for i in unmatched_dets:
#             trk = KalmanBoxTracker(
#                 dets[int(i), :],
#                 delta_t=self.delta_t,
#                 args=self.args,
#                 depth_enable=self.depth_enable,
#                 ema_obs=self.ema_obs,
#                 ema_pred=self.ema_pred,
#                 ua_enable=self.ua_enable,
#                 ua_h_ema=self.ua_h_ema,
#                 ua_h2_ema=self.ua_h2_ema,
#                 ua_proc_var_frac=self.ua_proc_var_frac,
#                 ua_sigma_floor_frac=self.ua_sigma_floor_frac,
#             )
#             self.trackers.append(trk)
#             if self.pc_enable and (not self.pc_update_matched_only):
#                 self._update_pc_model_with_det(dets[int(i), :], img_h, trk, img_w)

#         # Output and cleanup
#         i = len(self.trackers)
#         for trk in reversed(self.trackers):
#             if trk.last_observation.sum() < 0:
#                 d = trk.get_state()[0][:4]
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

#     # -----------------------------------------------------------------------
#     # Public detections path: keep close to original Hybrid-SORT behavior
#     # -----------------------------------------------------------------------
#     def update_public(self, dets, cates, scores):
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
#             cat = getattr(self.trackers[t], 'cate', 1)
#             trk[:] = [pos[0][0], pos[0][1], pos[0][2], pos[0][3], cat]
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

#         if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
#             left_dets = dets[unmatched_dets]
#             left_trks = last_boxes[unmatched_trks]
#             iou_left = np.array(iou_batch(left_dets, left_trks))
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
#                 if len(to_remove_det_indices) > 0:
#                     unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
#                 if len(to_remove_trk_indices) > 0:
#                     unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

#         for i in unmatched_dets:
#             trk = KalmanBoxTracker(
#                 dets[i, :],
#                 delta_t=self.delta_t,
#                 args=self.args,
#                 depth_enable=self.depth_enable,
#                 ema_obs=self.ema_obs,
#                 ema_pred=self.ema_pred,
#                 ua_enable=self.ua_enable,
#                 ua_h_ema=self.ua_h_ema,
#                 ua_h2_ema=self.ua_h2_ema,
#                 ua_proc_var_frac=self.ua_proc_var_frac,
#                 ua_sigma_floor_frac=self.ua_sigma_floor_frac,
#             )
#             trk.cate = cates[i]
#             self.trackers.append(trk)

#         i = len(self.trackers)
#         for trk in reversed(self.trackers):
#             if trk.last_observation.sum() > 0:
#                 d = trk.last_observation[:4]
#             else:
#                 d = trk.get_state()[0][:4]
#             if trk.time_since_update < 1:
#                 if (self.frame_count <= self.min_hits) or (trk.hit_streak >= self.min_hits):
#                     ret.append(np.concatenate((d, [trk.id + 1], [getattr(trk, 'cate', 1)], [0])).reshape(1, -1))
#                 if trk.hit_streak == self.min_hits:
#                     for prev_i in range(self.min_hits - 1):
#                         prev_observation = trk.history_observations[-(prev_i + 2)]
#                         ret.append((np.concatenate((prev_observation[:4], [trk.id + 1], [getattr(trk, 'cate', 1)], [-(prev_i + 1)]))).reshape(1, -1))
#             i -= 1
#             if trk.time_since_update > self.max_age:
#                 self.trackers.pop(i)

#         if len(ret) > 0:
#             return np.concatenate(ret)
#         return np.empty((0, 7))


