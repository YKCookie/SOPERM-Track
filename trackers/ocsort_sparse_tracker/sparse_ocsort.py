# -*- coding: utf-8 -*-
"""
#TODO best rePC-HMIoU
OC-SORT with:
- Soft-Depth Regularization
- UA-HMIoU (Uncertainty-Aware Height Modulated IoU)
- PC-HMIoU (Perspective-Consistent HMIoU)
NOTE: All original HMIoU (fixed height-ratio A_h) is REMOVED.
Both UA-HMIoU and PC-HMIoU are applied from the FIRST main matching round.

Matching pipeline in all rounds (1/2/3):
    IoU -> A_u (UA) -> A_p (PC) -> Soft-Depth blend

Includes robust numpy/torch conversions to avoid dtype issues.
"""    
    
from __future__ import print_function
import numpy as np

# association utilities from your project
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
    h = x[2] / w
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


# ---------- Kalman tracker with UA-height stats ----------
class KalmanBoxTracker(object):
    """
    OC-SORT Kalman tracker (7D state) +
    depth_ema (EMA of bottom y2) for pseudo-depth prior +
    UA-HMIoU height stats (EMA of h, h^2) with variance inflation.
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
        ua_sigma_floor_frac=0.05
    ):
        # define constant velocity model
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

        # Depth EMA (pseudo-depth y2)
        self.ema_obs = float(ema_obs)
        self.ema_pred = float(ema_pred)
        self.depth_ema = float(bbox[3])  # init with bottom y2

        # UA-HMIoU: Height stats (EMA of h, h^2) and variance
        h0 = float(bbox[3] - bbox[1])
        self.ua_h_ema = float(ua_h_ema)
        self.ua_h2_ema = float(ua_h2_ema)
        self.ua_proc_var_frac = float(ua_proc_var_frac)
        self.ua_sigma_floor_frac = float(ua_sigma_floor_frac)

        self.h_m1 = h0  # E[h]
        self.h_m2 = h0 * h0  # E[h^2]
        floor_var = (self.ua_sigma_floor_frac * h0) ** 2
        self.h_var = max(self.h_m2 - self.h_m1 * self.h_m1, floor_var)

    def get_height_mu_sigma(self):
        mu = float(self.h_m1)
        floor_var = (self.ua_sigma_floor_frac * max(mu, 1e-6)) ** 2
        var = max(float(self.h_var), floor_var)
        sigma = np.sqrt(var)
        return mu, sigma

    def _inflation_step(self):
        mu = max(float(self.h_m1), 1e-6)
        self.h_var = float(self.h_var) + (self.ua_proc_var_frac * mu) ** 2

    def update(self, bbox):
        if bbox is not None:
            # velocity for inertia
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
            # no measurement update: inflate height var a bit
            self._inflation_step()
            self.kf.update(bbox)

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        pred = convert_x_to_bbox(self.kf.x)  # (1,4)
        self.history.append(pred)

        # gently pull depth_ema towards predicted y2
        pred_y2 = float(pred[0][3])
        self.depth_ema = (1.0 - self.ema_pred) * self.depth_ema + self.ema_pred * pred_y2

        # process noise inflation for UA-height
        self._inflation_step()

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
        self.floor = float(floor)  # min affinity floor in [0,1]
        self.gamma = float(gamma)  # penalty strength
        self.reset()

    def reset(self):
        self.bin_means = np.zeros(self.num_bins, dtype=float)   # EMA of heights per bin
        self.bin_counts = np.zeros(self.num_bins, dtype=float)  # pseudo count for warmup
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
        # global fallback
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
        h_det = (dets_xyxy[:, 3] - dets_xyxy[:, 1]).astype(float)[:, None]  # (Nd,1)
        y2_trk = trks_xyxy[:, 3].astype(float)                              # (Nt,)
        h_star = self.expected_vec(y2_trk, img_h)[None, :]                  # (1,Nt)
        eps = 1e-6
        r = np.abs(h_det - h_star) / (h_star + eps)                         # (Nd,Nt)
        A_p = np.exp(-self.gamma * r)
        if self.floor is not None and self.floor > 0.0:
            A_p = np.maximum(A_p, self.floor)
        return A_p

    def ready(self):
        return self.global_count >= 50  # can be tuned


# ---------- main tracker with UA-HMIoU + PC-HMIoU (no original HMIoU) ----------
class sparse_OCSort(object):
    def __init__(
        self,
        det_thresh,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        delta_t=3,
        asso_func="iou",
        inertia=0.2,       # kept for API comp, not used in first round now
        use_byte=True,
        # Soft-depth params
        depth_alpha=0.5,
        depth_beta=2.0,
        depth_gate=0.25,
        gate_floor=0.2,
        # UA-HMIoU params
        ua_enable=True,
        ua_mode="cauchy",      # 'cauchy' or 'gauss'
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
        pc_sample_score=0.6,   # matched samples with score >= this update the PC model
        # Depth EMA
        ema_obs=0.5,
        ema_pred=0.1,
        
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

        KalmanBoxTracker.count = 0
        
        
         # 用于边界检测的状态
        self._prev_frame_id = -1
        self.last_img_size = None

    # in class sparse_OCSort:

    def reset(self, reset_perspective=True, reset_id_counter=False):
        """
        Reset the tracker state for a new sequence (or hard scene change).
        - reset_perspective: also reset the PC-HMIoU model
        - reset_id_counter: reset the global ID counter to 0
        """
        self.trackers = []
        self.frame_count = 0
        # clear boundary sentinels
        self._prev_frame_id = -1
        self.last_img_size = None
        # reset perspective prior if enabled
        if reset_perspective and getattr(self, "pc_enable", False) and hasattr(self, "perspective"):
            self.perspective.reset()
        # optionally reset global ID counter
        if reset_id_counter:
            KalmanBoxTracker.count = 0

    # 在类内新增
    def _maybe_reset_on_boundary(self, frame_id=None, img_size=None,
                                reset_on_size_change=False,  # 默认只看 frame_id
                                verbose=False):
        """
        在新序列边界仅重置透视曲线（不清轨迹/不重置ID）。
        规则：
        - 若 frame_id 回到 1，或 frame_id 发生回退（<= 上一帧），则重置
        - 可选：若 reset_on_size_change=True 且分辨率变化，也重置
        """
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

        # 更新内部记录
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
        y2d = dets_xyxy[:, 3].astype(float)[:, None]  # (Nd,1)
        y2t = trk_depth_ema.astype(float)[None, :]     # (1,Nt)
        norm = max(float(img_h), 1e-6)
        d = np.abs(y2d - y2t) / norm  # (Nd,Nt)
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
        """
        Uncertainty-aware height affinity A_u in (0,1]:
        z = |h_d - mu_h| / (alpha * sigma_eff)
        sigma_eff = sigma_trk * S_trk * S_det
          S_trk: relax when track missed (time_since_update)
          S_det: relax when detection near border / low score
        """
        # unify to numpy/scalars
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
        hd = (dets_xyxy[:, 3] - dets_xyxy[:, 1]).astype(float)[:, None]  # (Nd,1)
        mu = trk_mu_h.astype(float)[None, :]                              # (1,Nt)
        sigma_base = trk_sigma_h.astype(float)[None, :]                   # (1,Nt)

        # S_trk from miss count (cap at 2 frames)
        miss = np.clip(trk_miss.astype(float), 0.0, 2.0)[None, :]  # (1,Nt)
        S_trk = 1.0 + self.ua_miss_boost * (miss / 2.0)

        # S_det: border proximity + low score
        if Nd > 0:
            m = max(1.0, min(float(img_w), float(img_h)) * self.ua_border_margin_frac)
            x1 = dets_xyxy[:, 0].astype(float)
            y1 = dets_xyxy[:, 1].astype(float)
            x2 = dets_xyxy[:, 2].astype(float)
            y2 = dets_xyxy[:, 3].astype(float)
            min_dist = np.minimum(np.minimum(x1, y1), np.minimum(img_w - x2, img_h - y2))
            closeness = np.clip(1.0 - (min_dist / m), 0.0, 1.0)  # (Nd,)
            sc = det_scores.astype(float)
            low_score_gap = np.clip((self.ua_low_score - sc) / max(self.ua_low_score, 1e-6), 0.0, 1.0)
            S_det = 1.0 + self.ua_border_boost * closeness + self.ua_score_boost * low_score_gap  # (Nd,)
            S_det = S_det[:, None]  # (Nd,1)
        else:
            S_det = 1.0

        sigma_eff = sigma_base * S_trk       # (1,Nt)
        sigma_eff = sigma_eff * S_det        # (Nd,Nt)
        sigma_eff = np.maximum(sigma_eff, eps)

        z = np.abs(hd - mu) / (self.ua_alpha * sigma_eff + eps)
        if self.ua_mode == "gauss":
            A_u = np.exp(-0.5 * z * z)
        else:
            A_u = 1.0 / (1.0 + z * z)        # Cauchy-like
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
        if self.depth_alpha <= 0:
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

    # ---------- First round associate: IoU * A_u * A_p ----------
    def _associate_round1(
            self,
            dets, trks,
            height_mus, height_sigmas, miss_counts,
            img_w, img_h,
            iou_thr,
            depth_emas_full=None  # 新增：每条轨迹的 depth_ema，shape (Nt,)
        ):
            """
            dets: (Nd,5) [x1,y1,x2,y2,score]
            trks: (Nt,5) [x1,y1,x2,y2,0]
            returns: matched (K,2), unmatched_dets (idx), unmatched_trks (idx)
            """
            Nd = dets.shape[0]
            Nt = trks.shape[0]
            if Nd == 0 or Nt == 0:
                return np.empty((0, 2), dtype=int), np.arange(Nd), np.arange(Nt)

            dets_xy = dets[:, :4]
            trks_xy = trks[:, :4]
            det_sc = dets[:, 4]

            # 基础 IoU（用于门槛判定）
            iou = self.asso_func(dets_xy, trks_xy)
            iou = np.array(iou)

            # UA-HMIoU
            if self.ua_enable:
                A_u = self._ua_height_affinity(
                    dets_xy, trks_xy,
                    height_mus, height_sigmas,
                    det_sc, miss_counts,
                    img_w, img_h
                )
            else:
                A_u = np.ones_like(iou)

            # PC-HMIoU（带冷启动 floor）
            A_p = self._pc_affinity(dets_xy, trks_xy, img_h)

            # 用 S 排序（可选再做 Soft-Depth 融合）
            S = iou * A_u * A_p
            if (depth_emas_full is not None) and (self.depth_alpha > 0):
                A_d_r1 = self._depth_affinity(dets_xy, trks_xy, np.asarray(depth_emas_full), img_h)
                S = self._blend_iou_with_depth(S, A_d_r1)

            # 关键：门槛用“几何 IoU”，S 只用于排序
            matched = []
            if S.size > 0 and iou.max() > iou_thr:
                idxs = linear_assignment(-S)
                for d_i, t_i in idxs:
                    if iou[d_i, t_i] >= iou_thr:
                        matched.append([d_i, t_i])
            matched = np.asarray(matched, dtype=int) if len(matched) > 0 else np.empty((0, 2), dtype=int)

            matched_d = matched[:, 0] if matched.size > 0 else np.array([], dtype=int)
            matched_t = matched[:, 1] if matched.size > 0 else np.array([], dtype=int)
            unmatched_dets = np.setdiff1d(np.arange(Nd), matched_d, assume_unique=False)
            unmatched_trks = np.setdiff1d(np.arange(Nt), matched_t, assume_unique=False)

            return matched, unmatched_dets, unmatched_trks

    def update(self, output_results, img_info, img_size, frame_id=None):

        # auto reset on new sequence boundary (只重置透视曲线)
        self._maybe_reset_on_boundary(frame_id=frame_id,
                                    img_size=(img_info[0], img_info[1]),
                                    reset_on_size_change=True,  # 如需按分辨率也重置改为 True
                                    verbose=True)
        
        
        if output_results is None:
            return np.empty((0, 5))

        self.frame_count += 1

        # unify outputs to numpy
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

        # image size to scalars
        img_h, img_w = _to_scalar(img_info[0]), _to_scalar(img_info[1])

        # no detections
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

        # parse detections
        ncol = output_results.shape[1]
        if ncol >= 6:
            scores = output_results[:, 4].astype(float) * output_results[:, 5].astype(float)
            bboxes = output_results[:, :4].astype(float)
        elif ncol == 5:
            scores = output_results[:, 4].astype(float)
            bboxes = output_results[:, :4].astype(float)
        else:
            raise ValueError(f"Unexpected detection shape {output_results.shape}, expected 5 or >=6 columns.")

        # rescale to original image
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes = bboxes / max(scale, 1e-6)
        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)

        # split high/second detections
        inds_low = scores > 0.1
        inds_high = scores < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = dets[inds_second]
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]

        # predict all trackers
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

        # UA stats arrays for all trackers (aligned to self.trackers)
        height_mus_full = np.array([trk.get_height_mu_sigma()[0] for trk in self.trackers], dtype=float)
        height_sigmas_full = np.array([trk.get_height_mu_sigma()[1] for trk in self.trackers], dtype=float)
        miss_counts_full = np.array([trk.time_since_update for trk in self.trackers], dtype=float)

        # ---------- First round: MAIN matching with UA + PC ----------
        if dets.shape[0] > 0 and trks.shape[0] > 0:
            matched, unmatched_dets, unmatched_trks = self._associate_round1(
                dets, trks,
                height_mus_full, height_sigmas_full, miss_counts_full,
                img_w, img_h,
                self.iou_threshold,
                depth_emas_full = depth_emas  # 这里把 depth_emas 传进去 # 新增
            )
                     
            
            for d_i, t_i in matched:
                self.trackers[t_i].update(dets[d_i, :])
                # PC model update with stable sample
                self._update_pc_model_with_det(dets[d_i, :], img_h)
        else:
            matched = np.empty((0, 2), dtype=int)
            unmatched_dets = np.arange(dets.shape[0])
            unmatched_trks = np.arange(trks.shape[0])

        # ---------- Second round: BYTE with UA + PC + Soft-Depth ----------
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]               # (U,5)
            iou_left = self.asso_func(dets_second[:, :4], u_trks[:, :4])  # (Ds, U)
            iou_left = np.array(iou_left)

            if iou_left.size > 0:
                # A_u
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
                # A_p
                A_p = self._pc_affinity(dets_second[:, :4], u_trks[:, :4], img_h)
                
                if self.pc_enable:
                    Ap_dbg = A_p
                    if self.frame_count % 50 == 0:
                        print(f"[PC] frame={self.frame_count} count={self.perspective.global_count:.0f} "
                            f"ready={self.perspective.ready()} A_p(mean/min/max)="
                            f"{Ap_dbg.mean():.3f}/{Ap_dbg.min():.3f}/{Ap_dbg.max():.3f}")
                
                iou_geom = iou_left * A_u * A_p
            else:
                iou_geom = iou_left

            # Soft-depth
            if iou_geom.size > 0:
                u_trk_emas = depth_emas[unmatched_trks]
                A_d = self._depth_affinity(dets_second[:, :4], u_trks[:, :4], u_trk_emas, img_h)
                iou_adj = self._blend_iou_with_depth(iou_geom, A_d)
            else:
                iou_adj = iou_geom

            thr2 = max(0.1, self.iou_threshold - 0.07)
            if iou_adj.size > 0 and iou_adj.max() > thr2:
                matched_indices = linear_assignment(-iou_adj)
                to_remove_trk_indices = []
                to_remove_det_indices = []
                for m2 in matched_indices:
                    det_local, trk_local = m2[0], m2[1]
                    # if iou_adj[det_local, trk_local] < thr2:
                    #     continue
                    
                    if iou_left[det_local, trk_local] < thr2:
                         continue        
                    
                    trk_ind = unmatched_trks[trk_local]
                    self.trackers[trk_ind].update(dets_second[det_local, :])
                    to_remove_trk_indices.append(trk_ind)
                    to_remove_det_indices.append(det_local)
                    self._update_pc_model_with_det(dets_second[det_local, :], img_h)
                if len(to_remove_trk_indices) > 0:
                    unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))
                # remove those dets from dets_second indexing space to avoid duplicate when computing setdiff later
                if len(to_remove_det_indices) > 0:
                    keep_mask = np.ones(len(dets_second), dtype=bool)
                    keep_mask[to_remove_det_indices] = False
                    dets_second = dets_second[keep_mask]

        # ---------- Third round: Re-association with UA + PC + Soft-Depth + max(IoU(last, pred)) ----------
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]             # (Dl,5)
            left_trks_last = last_boxes[unmatched_trks]  # (Tl,5)
            left_trks_pred = trks[unmatched_trks]        # (Tl,5)

            iou_last = self.asso_func(left_dets[:, :4], left_trks_last[:, :4])
            iou_pred = self.asso_func(left_dets[:, :4], left_trks_pred[:, :4])
            iou_last = np.array(iou_last)
            iou_pred = np.array(iou_pred)
            iou_max = iou_last if iou_pred.size == 0 else np.maximum(iou_last, iou_pred)

            if iou_max.size > 0:
                dets_xy = left_dets[:, :4]
                trks_xy = left_trks_pred[:, :4]

                # UA
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

                # PC
                A_p = self._pc_affinity(dets_xy, trks_xy, img_h)
                
                if self.pc_enable:
                    Ap_dbg = A_p
                    if self.frame_count % 50 == 0:
                        print(f"[PC] frame={self.frame_count} count={self.perspective.global_count:.0f} "
                            f"ready={self.perspective.ready()} A_p(mean/min/max)="
                            f"{Ap_dbg.mean():.3f}/{Ap_dbg.min():.3f}/{Ap_dbg.max():.3f}")
                        

                iou_geom = iou_max * A_u * A_p

                # Soft-depth
                trk_emas = depth_emas[unmatched_trks]
                A_d = self._depth_affinity(dets_xy, trks_xy, trk_emas, img_h)
                iou_adj = self._blend_iou_with_depth(iou_geom, A_d)
            else:
                iou_adj = iou_max

            thr3 = max(0.1, self.iou_threshold - 0.07)
            if iou_adj.size > 0 and iou_adj.max() > thr3:
                rematched_indices = linear_assignment(-iou_adj)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m3 in rematched_indices:
                    det_local, trk_local = m3[0], m3[1]
                    # if iou_adj[det_local, trk_local] < thr3:
                    #     continue
                    
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

        # unmatched trackers: no measurement update
        for m in unmatched_trks:
            self.trackers[m].update(None)

        # create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(
                dets[i, :],
                delta_t=self.delta_t,
                ema_obs=self.ema_obs,
                ema_pred=self.ema_pred,
                ua_h_ema=self.ua_h_ema,
                ua_h2_ema=self.ua_h2_ema,
                ua_proc_var_frac=self.ua_proc_var_frac,
                ua_sigma_floor_frac=self.ua_sigma_floor_frac
            )
            self.trackers.append(trk)
            self._update_pc_model_with_det(dets[i, :], img_h)

        # output and cleanup
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
        # keep baseline behavior for public detections (no UA/PC used here for simplicity)
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
                ua_sigma_floor_frac=self.ua_sigma_floor_frac
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


# # # -*- coding: utf-8 -*-
# # """
# # #TODO best rePC-HMIoU,For MOT20 没用的样子
# # OC-SORT with:
# # - Soft-Depth Regularization
# # - UA-HMIoU (Uncertainty-Aware Height Modulated IoU)
# # - PC-HMIoU (Perspective-Consistent HMIoU)
# # NOTE: All original HMIoU (fixed height-ratio A_h) is REMOVED.
# # Both UA-HMIoU and PC-HMIoU are applied from the FIRST main matching round.

# # Matching pipeline in all rounds (1/2/3):
# #     IoU -> A_u (UA) -> A_p (PC) -> Soft-Depth blend

# # Includes robust numpy/torch conversions to avoid dtype issues.
# # """    
    
# from __future__ import print_function
# import numpy as np

# # association utilities from your project
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
#     h = x[2] / w
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


# # ---------- Kalman tracker with UA-height stats ----------
# class KalmanBoxTracker(object):
#     """
#     OC-SORT Kalman tracker (7D state) +
#     depth_ema (EMA of bottom y2) for pseudo-depth prior +
#     UA-HMIoU height stats (EMA of h, h^2) with variance inflation.
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
#         ua_sigma_floor_frac=0.05
#     ):
#         # define constant velocity model
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

#         # Depth EMA (pseudo-depth y2)
#         self.ema_obs = float(ema_obs)
#         self.ema_pred = float(ema_pred)
#         self.depth_ema = float(bbox[3])  # init with bottom y2

#         # UA-HMIoU: Height stats (EMA of h, h^2) and variance
#         h0 = float(bbox[3] - bbox[1])
#         self.ua_h_ema = float(ua_h_ema)
#         self.ua_h2_ema = float(ua_h2_ema)
#         self.ua_proc_var_frac = float(ua_proc_var_frac)
#         self.ua_sigma_floor_frac = float(ua_sigma_floor_frac)

#         self.h_m1 = h0  # E[h]
#         self.h_m2 = h0 * h0  # E[h^2]
#         floor_var = (self.ua_sigma_floor_frac * h0) ** 2
#         self.h_var = max(self.h_m2 - self.h_m1 * self.h_m1, floor_var)

#     def get_height_mu_sigma(self):
#         mu = float(self.h_m1)
#         floor_var = (self.ua_sigma_floor_frac * max(mu, 1e-6)) ** 2
#         var = max(float(self.h_var), floor_var)
#         sigma = np.sqrt(var)
#         return mu, sigma

#     def _inflation_step(self):
#         mu = max(float(self.h_m1), 1e-6)
#         self.h_var = float(self.h_var) + (self.ua_proc_var_frac * mu) ** 2

#     def update(self, bbox):
#         if bbox is not None:
#             # velocity for inertia
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
#             # no measurement update: inflate height var a bit
#             self._inflation_step()
#             self.kf.update(bbox)

#     def predict(self):
#         if (self.kf.x[6] + self.kf.x[2]) <= 0:
#             self.kf.x[6] *= 0.0
#         self.kf.predict()
#         self.age += 1
#         if self.time_since_update > 0:
#             self.hit_streak = 0
#         self.time_since_update += 1
#         pred = convert_x_to_bbox(self.kf.x)  # (1,4)
#         self.history.append(pred)

#         # gently pull depth_ema towards predicted y2
#         pred_y2 = float(pred[0][3])
#         self.depth_ema = (1.0 - self.ema_pred) * self.depth_ema + self.ema_pred * pred_y2

#         # process noise inflation for UA-height
#         self._inflation_step()

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
#         self.floor = float(floor)  # min affinity floor in [0,1]
#         self.gamma = float(gamma)  # penalty strength
#         self.reset()

#     def reset(self):
#         self.bin_means = np.zeros(self.num_bins, dtype=float)   # EMA of heights per bin
#         self.bin_counts = np.zeros(self.num_bins, dtype=float)  # pseudo count for warmup
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
#         # global fallback
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
#         h_det = (dets_xyxy[:, 3] - dets_xyxy[:, 1]).astype(float)[:, None]  # (Nd,1)
#         y2_trk = trks_xyxy[:, 3].astype(float)                              # (Nt,)
#         h_star = self.expected_vec(y2_trk, img_h)[None, :]                  # (1,Nt)
#         eps = 1e-6
#         r = np.abs(h_det - h_star) / (h_star + eps)                         # (Nd,Nt)
#         A_p = np.exp(-self.gamma * r)
#         if self.floor is not None and self.floor > 0.0:
#             A_p = np.maximum(A_p, self.floor)
#         return A_p

#     def ready(self):
#         return self.global_count >= 50  # can be tuned


# # ---------- main tracker with UA-HMIoU + PC-HMIoU (no original HMIoU) ----------
# class sparse_OCSort(object):
#     def __init__(
#         self,
#         det_thresh,
#         max_age=30,
#         min_hits=3,
#         iou_threshold=0.3,
#         delta_t=3,
#         asso_func="iou",
#         inertia=0.2,       # kept for API comp, not used in first round now
#         use_byte=True,
#         # Soft-depth params
#         depth_alpha=0.5,
#         depth_beta=2.0,
#         depth_gate=0.25,
#         gate_floor=0.2,
#         # UA-HMIoU params
#         ua_enable=True,
#         ua_mode="cauchy",      # 'cauchy' or 'gauss'
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
#         pc_sample_score=0.6,   # matched samples with score >= this update the PC model
#         # Depth EMA
#         ema_obs=0.5,
#         ema_pred=0.1,
        
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

#         KalmanBoxTracker.count = 0
        
        
#          # 用于边界检测的状态
#         self._prev_frame_id = -1
#         self.last_img_size = None

#     # in class sparse_OCSort:

#     def reset(self, reset_perspective=True, reset_id_counter=False):
#         """
#         Reset the tracker state for a new sequence (or hard scene change).
#         - reset_perspective: also reset the PC-HMIoU model
#         - reset_id_counter: reset the global ID counter to 0
#         """
#         self.trackers = []
#         self.frame_count = 0
#         # clear boundary sentinels
#         self._prev_frame_id = -1
#         self.last_img_size = None
#         # reset perspective prior if enabled
#         if reset_perspective and getattr(self, "pc_enable", False) and hasattr(self, "perspective"):
#             self.perspective.reset()
#         # optionally reset global ID counter
#         if reset_id_counter:
#             KalmanBoxTracker.count = 0

#     # 在类内新增
#     def _maybe_reset_on_boundary(self, frame_id=None, img_size=None,
#                                 reset_on_size_change=False,  # 默认只看 frame_id
#                                 verbose=False):
#         """
#         在新序列边界仅重置透视曲线（不清轨迹/不重置ID）。
#         规则：
#         - 若 frame_id 回到 1，或 frame_id 发生回退（<= 上一帧），则重置
#         - 可选：若 reset_on_size_change=True 且分辨率变化，也重置
#         """
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
#                         f"fid_back={fid_back}, size_changed={size_changed})")

#         # 更新内部记录
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
#         y2d = dets_xyxy[:, 3].astype(float)[:, None]  # (Nd,1)
#         y2t = trk_depth_ema.astype(float)[None, :]     # (1,Nt)
#         norm = max(float(img_h), 1e-6)
#         d = np.abs(y2d - y2t) / norm  # (Nd,Nt)
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
#         """
#         Uncertainty-aware height affinity A_u in (0,1]:
#         z = |h_d - mu_h| / (alpha * sigma_eff)
#         sigma_eff = sigma_trk * S_trk * S_det
#           S_trk: relax when track missed (time_since_update)
#           S_det: relax when detection near border / low score
#         """
#         # unify to numpy/scalars
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
#         hd = (dets_xyxy[:, 3] - dets_xyxy[:, 1]).astype(float)[:, None]  # (Nd,1)
#         mu = trk_mu_h.astype(float)[None, :]                              # (1,Nt)
#         sigma_base = trk_sigma_h.astype(float)[None, :]                   # (1,Nt)

#         # S_trk from miss count (cap at 2 frames)
#         miss = np.clip(trk_miss.astype(float), 0.0, 2.0)[None, :]  # (1,Nt)
#         S_trk = 1.0 + self.ua_miss_boost * (miss / 2.0)

#         # S_det: border proximity + low score
#         if Nd > 0:
#             m = max(1.0, min(float(img_w), float(img_h)) * self.ua_border_margin_frac)
#             x1 = dets_xyxy[:, 0].astype(float)
#             y1 = dets_xyxy[:, 1].astype(float)
#             x2 = dets_xyxy[:, 2].astype(float)
#             y2 = dets_xyxy[:, 3].astype(float)
#             min_dist = np.minimum(np.minimum(x1, y1), np.minimum(img_w - x2, img_h - y2))
#             closeness = np.clip(1.0 - (min_dist / m), 0.0, 1.0)  # (Nd,)
#             sc = det_scores.astype(float)
#             low_score_gap = np.clip((self.ua_low_score - sc) / max(self.ua_low_score, 1e-6), 0.0, 1.0)
#             S_det = 1.0 + self.ua_border_boost * closeness + self.ua_score_boost * low_score_gap  # (Nd,)
#             S_det = S_det[:, None]  # (Nd,1)
#         else:
#             S_det = 1.0

#         sigma_eff = sigma_base * S_trk       # (1,Nt)
#         sigma_eff = sigma_eff * S_det        # (Nd,Nt)
#         sigma_eff = np.maximum(sigma_eff, eps)

#         z = np.abs(hd - mu) / (self.ua_alpha * sigma_eff + eps)
#         if self.ua_mode == "gauss":
#             A_u = np.exp(-0.5 * z * z)
#         else:
#             A_u = 1.0 / (1.0 + z * z)        # Cauchy-like
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
#         if self.depth_alpha <= 0:
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

#     # ---------- First round associate: IoU * A_u * A_p ----------
#     def _associate_round1(
#             self,
#             dets, trks,
#             height_mus, height_sigmas, miss_counts,
#             img_w, img_h,
#             iou_thr,
#             depth_emas_full=None  # 新增：每条轨迹的 depth_ema，shape (Nt,)
#         ):
#             """
#             dets: (Nd,5) [x1,y1,x2,y2,score]
#             trks: (Nt,5) [x1,y1,x2,y2,0]
#             returns: matched (K,2), unmatched_dets (idx), unmatched_trks (idx)
#             """
#             Nd = dets.shape[0]
#             Nt = trks.shape[0]
#             if Nd == 0 or Nt == 0:
#                 return np.empty((0, 2), dtype=int), np.arange(Nd), np.arange(Nt)

#             dets_xy = dets[:, :4]
#             trks_xy = trks[:, :4]
#             det_sc = dets[:, 4]

#             # 基础 IoU（用于门槛判定）
#             iou = self.asso_func(dets_xy, trks_xy)
#             iou = np.array(iou)

#             # UA-HMIoU
#             if self.ua_enable:
#                 A_u = self._ua_height_affinity(
#                     dets_xy, trks_xy,
#                     height_mus, height_sigmas,
#                     det_sc, miss_counts,
#                     img_w, img_h
#                 )
#             else:
#                 A_u = np.ones_like(iou)

#             # PC-HMIoU（带冷启动 floor）
#             A_p = self._pc_affinity(dets_xy, trks_xy, img_h)

#             # 用 S 排序（可选再做 Soft-Depth 融合）
#             S = iou * A_u * A_p
#             if (depth_emas_full is not None) and (self.depth_alpha > 0):
#                 A_d_r1 = self._depth_affinity(dets_xy, trks_xy, np.asarray(depth_emas_full), img_h)
#                 S = self._blend_iou_with_depth(S, A_d_r1)

#             # 关键：门槛用“几何 IoU”，S 只用于排序
#             matched = []
#             if S.size > 0 and iou.max() > iou_thr:
#                 idxs = linear_assignment(-S)
#                 for d_i, t_i in idxs:
#                     if iou[d_i, t_i] >= iou_thr:
#                         matched.append([d_i, t_i])
#             matched = np.asarray(matched, dtype=int) if len(matched) > 0 else np.empty((0, 2), dtype=int)

#             matched_d = matched[:, 0] if matched.size > 0 else np.array([], dtype=int)
#             matched_t = matched[:, 1] if matched.size > 0 else np.array([], dtype=int)
#             unmatched_dets = np.setdiff1d(np.arange(Nd), matched_d, assume_unique=False)
#             unmatched_trks = np.setdiff1d(np.arange(Nt), matched_t, assume_unique=False)

#             return matched, unmatched_dets, unmatched_trks

#     def update(self, output_results, img_info, img_size, frame_id=None):

#         # auto reset on new sequence boundary (只重置透视曲线)
#         self._maybe_reset_on_boundary(frame_id=frame_id,
#                                     img_size=(img_info[0], img_info[1]),
#                                     reset_on_size_change=True,  # 如需按分辨率也重置改为 True
#                                     verbose=True)
        
        
#         if output_results is None:
#             return np.empty((0, 5))

#         self.frame_count += 1

#         # unify outputs to numpy
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

#         # image size to scalars
#         img_h, img_w = _to_scalar(img_info[0]), _to_scalar(img_info[1])

#         # no detections
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

#         # parse detections
#         ncol = output_results.shape[1]
#         if ncol >= 6:
#             scores = output_results[:, 4].astype(float) * output_results[:, 5].astype(float)
#             bboxes = output_results[:, :4].astype(float)
#         elif ncol == 5:
#             scores = output_results[:, 4].astype(float)
#             bboxes = output_results[:, :4].astype(float)
#         else:
#             raise ValueError(f"Unexpected detection shape {output_results.shape}, expected 5 or >=6 columns.")

#         # rescale to original image
#         scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
#         bboxes = bboxes / max(scale, 1e-6)
#         dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)

#         # split high/second detections
#         # inds_low = scores > 0.1
#         #For MOT20
#         inds_low = scores > 0.05
        
        
#         inds_high = scores < self.det_thresh
#         inds_second = np.logical_and(inds_low, inds_high)
#         dets_second = dets[inds_second]
#         remain_inds = scores > self.det_thresh
#         dets = dets[remain_inds]

#         # predict all trackers
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

#         # UA stats arrays for all trackers (aligned to self.trackers)
#         height_mus_full = np.array([trk.get_height_mu_sigma()[0] for trk in self.trackers], dtype=float)
#         height_sigmas_full = np.array([trk.get_height_mu_sigma()[1] for trk in self.trackers], dtype=float)
#         miss_counts_full = np.array([trk.time_since_update for trk in self.trackers], dtype=float)

#         # ---------- First round: MAIN matching with UA + PC ----------
#         if dets.shape[0] > 0 and trks.shape[0] > 0:
#             matched, unmatched_dets, unmatched_trks = self._associate_round1(
#                 dets, trks,
#                 height_mus_full, height_sigmas_full, miss_counts_full,
#                 img_w, img_h,
#                 self.iou_threshold,
#                 depth_emas_full = depth_emas  # 这里把 depth_emas 传进去 # 新增
#             )
                     
            
#             for d_i, t_i in matched:
#                 self.trackers[t_i].update(dets[d_i, :])
#                 # PC model update with stable sample
#                 self._update_pc_model_with_det(dets[d_i, :], img_h)
#         else:
#             matched = np.empty((0, 2), dtype=int)
#             unmatched_dets = np.arange(dets.shape[0])
#             unmatched_trks = np.arange(trks.shape[0])

#         # ---------- Second round: BYTE with UA + PC + Soft-Depth ----------
#         if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
#             u_trks = trks[unmatched_trks]               # (U,5)
#             iou_left = self.asso_func(dets_second[:, :4], u_trks[:, :4])  # (Ds, U)
#             iou_left = np.array(iou_left)

#             if iou_left.size > 0:
#                 # A_u
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
#                 # # A_p
#                 # A_p = self._pc_affinity(dets_second[:, :4], u_trks[:, :4], img_h)
#                 #For MOT20
#                 A_p = np.ones_like(iou_left)
                
#                 if self.pc_enable:
#                     Ap_dbg = A_p
#                     if self.frame_count % 50 == 0:
#                         print(f"[PC] frame={self.frame_count} count={self.perspective.global_count:.0f} "
#                             f"ready={self.perspective.ready()} A_p(mean/min/max)="
#                             f"{Ap_dbg.mean():.3f}/{Ap_dbg.min():.3f}/{Ap_dbg.max():.3f}")
                
#                 iou_geom = iou_left * A_u * A_p
#             else:
#                 iou_geom = iou_left

#             # Soft-depth
#             if iou_geom.size > 0:
#                 u_trk_emas = depth_emas[unmatched_trks]
#                 A_d = self._depth_affinity(dets_second[:, :4], u_trks[:, :4], u_trk_emas, img_h)
#                 iou_adj = self._blend_iou_with_depth(iou_geom, A_d)
#             else:
#                 iou_adj = iou_geom

#             # thr2 = max(0.1, self.iou_threshold - 0.07)
#             thr2 = max(0.08, self.iou_threshold - 0.12)
#             if iou_adj.size > 0 and iou_adj.max() > thr2:
#                 matched_indices = linear_assignment(-iou_adj)
#                 to_remove_trk_indices = []
#                 to_remove_det_indices = []
#                 for m2 in matched_indices:
#                     det_local, trk_local = m2[0], m2[1]
#                     # if iou_adj[det_local, trk_local] < thr2:
#                     #     continue
                    
#                     if iou_left[det_local, trk_local] < thr2:
#                          continue        
                    
#                     trk_ind = unmatched_trks[trk_local]
#                     self.trackers[trk_ind].update(dets_second[det_local, :])
#                     to_remove_trk_indices.append(trk_ind)
#                     to_remove_det_indices.append(det_local)
#                     self._update_pc_model_with_det(dets_second[det_local, :], img_h)
#                 if len(to_remove_trk_indices) > 0:
#                     unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))
#                 # remove those dets from dets_second indexing space to avoid duplicate when computing setdiff later
#                 if len(to_remove_det_indices) > 0:
#                     keep_mask = np.ones(len(dets_second), dtype=bool)
#                     keep_mask[to_remove_det_indices] = False
#                     dets_second = dets_second[keep_mask]

#         # # ---------- Third round: Re-association with UA + PC + Soft-Depth + max(IoU(last, pred)) ----------
#         # if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
#         #     left_dets = dets[unmatched_dets]             # (Dl,5)
#         #     left_trks_last = last_boxes[unmatched_trks]  # (Tl,5)
#         #     left_trks_pred = trks[unmatched_trks]        # (Tl,5)

#         #     iou_last = self.asso_func(left_dets[:, :4], left_trks_last[:, :4])
#         #     iou_pred = self.asso_func(left_dets[:, :4], left_trks_pred[:, :4])
#         #     iou_last = np.array(iou_last)
#         #     iou_pred = np.array(iou_pred)
#         #     iou_max = iou_last if iou_pred.size == 0 else np.maximum(iou_last, iou_pred)

#         #     if iou_max.size > 0:
#         #         dets_xy = left_dets[:, :4]
#         #         trks_xy = left_trks_pred[:, :4]

#         #         # UA
#         #         if self.ua_enable:
#         #             l_mu = height_mus_full[unmatched_trks]
#         #             l_sigma = height_sigmas_full[unmatched_trks]
#         #             l_miss = miss_counts_full[unmatched_trks]
#         #             det_sc = left_dets[:, 4]
#         #             A_u = self._ua_height_affinity(
#         #                 dets_xy, trks_xy,
#         #                 l_mu, l_sigma,
#         #                 det_sc, l_miss,
#         #                 img_w, img_h
#         #             )
#         #         else:
#         #             A_u = np.ones_like(iou_max)

#         #         # # PC
#         #         # A_p = self._pc_affinity(dets_xy, trks_xy, img_h)
#         #         # For MOT20
#         #         A_p = np.ones_like(iou_max)
                
#         #         if self.pc_enable:
#         #             Ap_dbg = A_p
#         #             if self.frame_count % 50 == 0:
#         #                 print(f"[PC] frame={self.frame_count} count={self.perspective.global_count:.0f} "
#         #                     f"ready={self.perspective.ready()} A_p(mean/min/max)="
#         #                     f"{Ap_dbg.mean():.3f}/{Ap_dbg.min():.3f}/{Ap_dbg.max():.3f}")
                        

#         #         iou_geom = iou_max * A_u * A_p

#         #         # Soft-depth
#         #         trk_emas = depth_emas[unmatched_trks]
#         #         A_d = self._depth_affinity(dets_xy, trks_xy, trk_emas, img_h)
#         #         iou_adj = self._blend_iou_with_depth(iou_geom, A_d)
#         #     else:
#         #         iou_adj = iou_max

#         #     # thr3 = max(0.1, self.iou_threshold - 0.07)
#         #     # For MOT20
#         #     thr3 = max(0.08, self.iou_threshold - 0.12)
            
#         #     if iou_adj.size > 0 and iou_adj.max() > thr3:
#         #         rematched_indices = linear_assignment(-iou_adj)
#         #         to_remove_det_indices = []
#         #         to_remove_trk_indices = []
#         #         for m3 in rematched_indices:
#         #             det_local, trk_local = m3[0], m3[1]
#         #             # if iou_adj[det_local, trk_local] < thr3:
#         #             #     continue
                    
#         #             if iou_max[det_local, trk_local] < thr3:
#         #                   continue
                    
#         #             det_ind = unmatched_dets[det_local]
#         #             trk_ind = unmatched_trks[trk_local]
#         #             self.trackers[trk_ind].update(dets[det_ind, :])
#         #             to_remove_det_indices.append(det_ind)
#         #             to_remove_trk_indices.append(trk_ind)
#         #             self._update_pc_model_with_det(dets[det_ind, :], img_h)
#         #         if len(to_remove_det_indices) > 0:
#         #             unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
#         #         if len(to_remove_trk_indices) > 0:
#         #             unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))
        
        
#         # ---------- Third round: Re-association with UA + PC + Soft-Depth + adaptive IoU(last vs pred) ----------
#         if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
#             left_dets = dets[unmatched_dets]             # (Dl,5)
#             left_trks_last = last_boxes[unmatched_trks]  # (Tl,5)
#             left_trks_pred = trks[unmatched_trks]        # (Tl,5)

#             # 1) 基础几何 IoU
#             iou_last = self.asso_func(left_dets[:, :4], left_trks_last[:, :4])
#             iou_pred = self.asso_func(left_dets[:, :4], left_trks_pred[:, :4])
#             iou_last = np.array(iou_last)
#             iou_pred = np.array(iou_pred)

#             # 2) 自适应选择：对于 time_since_update <= recent_k 的轨迹，优先使用 last；否则用 max(last, pred)
#             recent_k = 1
#             miss_sel = miss_counts_full[unmatched_trks]          # (Tl,)
#             mask_recent = (miss_sel <= recent_k)                 # (Tl,) True 表示“刚更新过”

#             if iou_pred.size == 0:
#                 iou_ref = iou_last
#                 trks_xy_ref = left_trks_last[:, :4]
#             else:
#                 # 构造参考几何矩阵 iou_ref
#                 iou_ref = np.maximum(iou_last, iou_pred)
#                 # 对于“刚更新”的轨迹，用 last 覆盖列
#                 if mask_recent.any():
#                     iou_ref[:, mask_recent] = iou_last[:, mask_recent]

#                 # 同时构造参考轨迹框 trks_xy_ref（用于 UA/PC/Depth 计算）
#                 trks_xy_ref = left_trks_pred[:, :4].copy()
#                 trks_xy_ref[mask_recent] = left_trks_last[:, :4][mask_recent]

#             # 3) 先验构造
#             dets_xy = left_dets[:, :4]

#             # UA
#             if self.ua_enable:
#                 l_mu = height_mus_full[unmatched_trks]
#                 l_sigma = height_sigmas_full[unmatched_trks]
#                 l_miss = miss_counts_full[unmatched_trks]
#                 det_sc = left_dets[:, 4]
#                 A_u = self._ua_height_affinity(
#                     dets_xy, trks_xy_ref,
#                     l_mu, l_sigma,
#                     det_sc, l_miss,
#                     img_w, img_h
#                 )
#             else:
#                 A_u = np.ones_like(iou_ref)

#             # PC（如需更保守，可将 A_p = np.ones_like(iou_ref)）
#             A_p = self._pc_affinity(dets_xy, trks_xy_ref, img_h)

#             iou_geom = iou_ref * A_u * A_p

#             # Soft-Depth
#             trk_emas = depth_emas[unmatched_trks]
#             A_d = self._depth_affinity(dets_xy, trks_xy_ref, trk_emas, img_h)
#             iou_adj = self._blend_iou_with_depth(iou_geom, A_d)

#             # 4) 接受条件仍仅看“几何 iou_ref”，与排序解耦
#             thr3 = max(0.1, self.iou_threshold - 0.07)
#             if iou_adj.size > 0 and iou_adj.max() > thr3:
#                 rematched_indices = linear_assignment(-iou_adj)
#                 to_remove_det_indices = []
#                 to_remove_trk_indices = []
#                 for m3 in rematched_indices:
#                     det_local, trk_local = m3[0], m3[1]
#                     if iou_ref[det_local, trk_local] < thr3:
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

#         # unmatched trackers: no measurement update
#         for m in unmatched_trks:
#             self.trackers[m].update(None)

#         # create new trackers for unmatched detections
#         for i in unmatched_dets:
#             trk = KalmanBoxTracker(
#                 dets[i, :],
#                 delta_t=self.delta_t,
#                 ema_obs=self.ema_obs,
#                 ema_pred=self.ema_pred,
#                 ua_h_ema=self.ua_h_ema,
#                 ua_h2_ema=self.ua_h2_ema,
#                 ua_proc_var_frac=self.ua_proc_var_frac,
#                 ua_sigma_floor_frac=self.ua_sigma_floor_frac
#             )
#             self.trackers.append(trk)
#             self._update_pc_model_with_det(dets[i, :], img_h)

#         # output and cleanup
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
#         # keep baseline behavior for public detections (no UA/PC used here for simplicity)
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
#                 ua_sigma_floor_frac=self.ua_sigma_floor_frac
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



# # -*- coding: utf-8 -*-
# """

# OC-SORT with:
# - Soft-Depth Regularization
# - UA-HMIoU (Uncertainty-Aware Height Modulated IoU)
# - PC-HMIoU (Perspective-Consistent HMIoU)
# NOTE: All original HMIoU (fixed height-ratio A_h) is REMOVED.
# Both UA-HMIoU and PC-HMIoU are applied from the FIRST main matching round.

# Matching pipeline in all rounds (1/2/3):
#     IoU -> A_u (UA) -> A_p (PC) -> Soft-Depth blend

# Includes robust numpy/torch conversions to avoid dtype issues.

# #TODO UPDATE:
# - #TODO update_public now uses the same three methods (UA-HMIoU, PC-HMIoU, Soft-Depth)
#   across its matching steps, with category-consistency gating.
# """    
    
# from __future__ import print_function
# import numpy as np

# # association utilities from your project
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
#     h = x[2] / w
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


# # ---------- Kalman tracker with UA-height stats ----------
# class KalmanBoxTracker(object):
#     """
#     OC-SORT Kalman tracker (7D state) +
#     depth_ema (EMA of bottom y2) for pseudo-depth prior +
#     UA-HMIoU height stats (EMA of h, h^2) with variance inflation.
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
#         ua_sigma_floor_frac=0.05
#     ):
#         # define constant velocity model
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

#         # Depth EMA (pseudo-depth y2)
#         self.ema_obs = float(ema_obs)
#         self.ema_pred = float(ema_pred)
#         self.depth_ema = float(bbox[3])  # init with bottom y2

#         # UA-HMIoU: Height stats (EMA of h, h^2) and variance
#         h0 = float(bbox[3] - bbox[1])
#         self.ua_h_ema = float(ua_h_ema)
#         self.ua_h2_ema = float(ua_h2_ema)
#         self.ua_proc_var_frac = float(ua_proc_var_frac)
#         self.ua_sigma_floor_frac = float(ua_sigma_floor_frac)

#         self.h_m1 = h0  # E[h]
#         self.h_m2 = h0 * h0  # E[h^2]
#         floor_var = (self.ua_sigma_floor_frac * h0) ** 2
#         self.h_var = max(self.h_m2 - self.h_m1 * self.h_m1, floor_var)

#     def get_height_mu_sigma(self):
#         mu = float(self.h_m1)
#         floor_var = (self.ua_sigma_floor_frac * max(mu, 1e-6)) ** 2
#         var = max(float(self.h_var), floor_var)
#         sigma = np.sqrt(var)
#         return mu, sigma

#     def _inflation_step(self):
#         mu = max(float(self.h_m1), 1e-6)
#         self.h_var = float(self.h_var) + (self.ua_proc_var_frac * mu) ** 2

#     def update(self, bbox):
#         if bbox is not None:
#             # velocity for inertia
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
#             # no measurement update: inflate height var a bit
#             self._inflation_step()
#             self.kf.update(bbox)

#     def predict(self):
#         if (self.kf.x[6] + self.kf.x[2]) <= 0:
#             self.kf.x[6] *= 0.0
#         self.kf.predict()
#         self.age += 1
#         if self.time_since_update > 0:
#             self.hit_streak = 0
#         self.time_since_update += 1
#         pred = convert_x_to_bbox(self.kf.x)  # (1,4)
#         self.history.append(pred)

#         # gently pull depth_ema towards predicted y2
#         pred_y2 = float(pred[0][3])
#         self.depth_ema = (1.0 - self.ema_pred) * self.depth_ema + self.ema_pred * pred_y2

#         # process noise inflation for UA-height
#         self._inflation_step()

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
#         self.floor = float(floor)  # min affinity floor in [0,1]
#         self.gamma = float(gamma)  # penalty strength
#         self.reset()

#     def reset(self):
#         self.bin_means = np.zeros(self.num_bins, dtype=float)   # EMA of heights per bin
#         self.bin_counts = np.zeros(self.num_bins, dtype=float)  # pseudo count for warmup
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
#         # global fallback
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
#         h_det = (dets_xyxy[:, 3] - dets_xyxy[:, 1]).astype(float)[:, None]  # (Nd,1)
#         y2_trk = trks_xyxy[:, 3].astype(float)                              # (Nt,)
#         h_star = self.expected_vec(y2_trk, img_h)[None, :]                  # (1,Nt)
#         eps = 1e-6
#         r = np.abs(h_det - h_star) / (h_star + eps)                         # (Nd,Nt)
#         A_p = np.exp(-self.gamma * r)
#         if self.floor is not None and self.floor > 0.0:
#             A_p = np.maximum(A_p, self.floor)
#         return A_p

#     def ready(self):
#         return self.global_count >= 50  # can be tuned


# # ---------- main tracker with UA-HMIoU + PC-HMIoU (no original HMIoU) ----------
# class sparse_OCSort(object):
#     def __init__(
#         self,
#         det_thresh,
#         max_age=30,
#         min_hits=3,
#         iou_threshold=0.3,
#         delta_t=3,
#         asso_func="iou",
#         inertia=0.2,       # kept for API comp, not used in first round now
#         use_byte=True,
#         # Soft-depth params
#         depth_alpha=0.5,
#         depth_beta=2.0,
#         depth_gate=0.25,
#         gate_floor=0.2,
#         # UA-HMIoU params
#         ua_enable=True,
#         ua_mode="cauchy",      # 'cauchy' or 'gauss'
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
#         pc_sample_score=0.6,   # matched samples with score >= this update the PC model
#         # Depth EMA
#         ema_obs=0.5,
#         ema_pred=0.1,
        
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

#         KalmanBoxTracker.count = 0
        
        
#          # 用于边界检测的状态
#         self._prev_frame_id = -1
#         self.last_img_size = None

#     # in class sparse_OCSort:

#     def reset(self, reset_perspective=True, reset_id_counter=False):
#         """
#         Reset the tracker state for a new sequence (or hard scene change).
#         - reset_perspective: also reset the PC-HMIoU model
#         - reset_id_counter: reset the global ID counter to 0
#         """
#         self.trackers = []
#         self.frame_count = 0
#         # clear boundary sentinels
#         self._prev_frame_id = -1
#         self.last_img_size = None
#         # reset perspective prior if enabled
#         if reset_perspective and getattr(self, "pc_enable", False) and hasattr(self, "perspective"):
#             self.perspective.reset()
#         # optionally reset global ID counter
#         if reset_id_counter:
#             KalmanBoxTracker.count = 0

#     # 在类内新增
#     def _maybe_reset_on_boundary(self, frame_id=None, img_size=None,
#                                 reset_on_size_change=False,  # 默认只看 frame_id
#                                 verbose=False):
#         """
#         在新序列边界仅重置透视曲线（不清轨迹/不重置ID）。
#         规则：
#         - 若 frame_id 回到 1，或 frame_id 发生回退（<= 上一帧），则重置
#         - 可选：若 reset_on_size_change=True 且分辨率变化，也重置
#         """
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
#                         f"fid_back={fid_back}, size_changed={size_changed})")

#         # 更新内部记录
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
#         y2d = dets_xyxy[:, 3].astype(float)[:, None]  # (Nd,1)
#         y2t = trk_depth_ema.astype(float)[None, :]     # (1,Nt)
#         norm = max(float(img_h), 1e-6)
#         d = np.abs(y2d - y2t) / norm  # (Nd,Nt)
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
#         """
#         Uncertainty-aware height affinity A_u in (0,1]:
#         z = |h_d - mu_h| / (alpha * sigma_eff)
#         sigma_eff = sigma_trk * S_trk * S_det
#           S_trk: relax when track missed (time_since_update)
#           S_det: relax when detection near border / low score
#         """
#         # unify to numpy/scalars
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
#         hd = (dets_xyxy[:, 3] - dets_xyxy[:, 1]).astype(float)[:, None]  # (Nd,1)
#         mu = trk_mu_h.astype(float)[None, :]                              # (1,Nt)
#         sigma_base = trk_sigma_h.astype(float)[None, :]                   # (1,Nt)

#         # S_trk from miss count (cap at 2 frames)
#         miss = np.clip(trk_miss.astype(float), 0.0, 2.0)[None, :]  # (1,Nt)
#         S_trk = 1.0 + self.ua_miss_boost * (miss / 2.0)

#         # S_det: border proximity + low score
#         if Nd > 0:
#             m = max(1.0, min(float(img_w), float(img_h)) * self.ua_border_margin_frac)
#             x1 = dets_xyxy[:, 0].astype(float)
#             y1 = dets_xyxy[:, 1].astype(float)
#             x2 = dets_xyxy[:, 2].astype(float)
#             y2 = dets_xyxy[:, 3].astype(float)
#             min_dist = np.minimum(np.minimum(x1, y1), np.minimum(img_w - x2, img_h - y2))
#             closeness = np.clip(1.0 - (min_dist / m), 0.0, 1.0)  # (Nd,)
#             sc = det_scores.astype(float)
#             low_score_gap = np.clip((self.ua_low_score - sc) / max(self.ua_low_score, 1e-6), 0.0, 1.0)
#             S_det = 1.0 + self.ua_border_boost * closeness + self.ua_score_boost * low_score_gap  # (Nd,)
#             S_det = S_det[:, None]  # (Nd,1)
#         else:
#             S_det = 1.0

#         sigma_eff = sigma_base * S_trk       # (1,Nt)
#         sigma_eff = sigma_eff * S_det        # (Nd,Nt)
#         sigma_eff = np.maximum(sigma_eff, eps)

#         z = np.abs(hd - mu) / (self.ua_alpha * sigma_eff + eps)
#         if self.ua_mode == "gauss":
#             A_u = np.exp(-0.5 * z * z)
#         else:
#             A_u = 1.0 / (1.0 + z * z)        # Cauchy-like
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
#         if self.depth_alpha <= 0:
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

#     # ---------- First round associate: IoU * A_u * A_p ----------
#     def _associate_round1(
#             self,
#             dets, trks,
#             height_mus, height_sigmas, miss_counts,
#             img_w, img_h,
#             iou_thr,
#             depth_emas_full=None  # 新增：每条轨迹的 depth_ema，shape (Nt,)
#         ):
#             """
#             dets: (Nd,5) [x1,y1,x2,y2,score]
#             trks: (Nt,5) [x1,y1,x2,y2,0]
#             returns: matched (K,2), unmatched_dets (idx), unmatched_trks (idx)
#             """
#             Nd = dets.shape[0]
#             Nt = trks.shape[0]
#             if Nd == 0 or Nt == 0:
#                 return np.empty((0, 2), dtype=int), np.arange(Nd), np.arange(Nt)

#             dets_xy = dets[:, :4]
#             trks_xy = trks[:, :4]
#             det_sc = dets[:, 4]

#             # 基础 IoU（用于门槛判定）
#             iou = self.asso_func(dets_xy, trks_xy)
#             iou = np.array(iou)

#             # UA-HMIoU
#             if self.ua_enable:
#                 A_u = self._ua_height_affinity(
#                     dets_xy, trks_xy,
#                     height_mus, height_sigmas,
#                     det_sc, miss_counts,
#                     img_w, img_h
#                 )
#             else:
#                 A_u = np.ones_like(iou)

#             # PC-HMIoU（带冷启动 floor）
#             A_p = self._pc_affinity(dets_xy, trks_xy, img_h)

#             # 用 S 排序（可选再做 Soft-Depth 融合）
#             S = iou * A_u * A_p
#             if (depth_emas_full is not None) and (self.depth_alpha > 0):
#                 A_d_r1 = self._depth_affinity(dets_xy, trks_xy, np.asarray(depth_emas_full), img_h)
#                 S = self._blend_iou_with_depth(S, A_d_r1)

#             # 关键：门槛用“几何 IoU”，S 只用于排序
#             matched = []
#             if S.size > 0 and iou.max() > iou_thr:
#                 idxs = linear_assignment(-S)
#                 for d_i, t_i in idxs:
#                     if iou[d_i, t_i] >= iou_thr:
#                         matched.append([d_i, t_i])
#             matched = np.asarray(matched, dtype=int) if len(matched) > 0 else np.empty((0, 2), dtype=int)

#             matched_d = matched[:, 0] if matched.size > 0 else np.array([], dtype=int)
#             matched_t = matched[:, 1] if matched.size > 0 else np.array([], dtype=int)
#             unmatched_dets = np.setdiff1d(np.arange(Nd), matched_d, assume_unique=False)
#             unmatched_trks = np.setdiff1d(np.arange(Nt), matched_t, assume_unique=False)

#             return matched, unmatched_dets, unmatched_trks

#     def update(self, output_results, img_info, img_size, frame_id=None):

#         # auto reset on new sequence boundary (只重置透视曲线)
#         self._maybe_reset_on_boundary(frame_id=frame_id,
#                                     img_size=(img_info[0], img_info[1]),
#                                     reset_on_size_change=True,  # 如需按分辨率也重置改为 True
#                                     verbose=True)
        
        
#         if output_results is None:
#             return np.empty((0, 5))

#         self.frame_count += 1

#         # unify outputs to numpy
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

#         # image size to scalars
#         img_h, img_w = _to_scalar(img_info[0]), _to_scalar(img_info[1])

#         # no detections
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

#         # parse detections
#         ncol = output_results.shape[1]
#         if ncol >= 6:
#             scores = output_results[:, 4].astype(float) * output_results[:, 5].astype(float)
#             bboxes = output_results[:, :4].astype(float)
#         elif ncol == 5:
#             scores = output_results[:, 4].astype(float)
#             bboxes = output_results[:, :4].astype(float)
#         else:
#             raise ValueError(f"Unexpected detection shape {output_results.shape}, expected 5 or >=6 columns.")

#         # rescale to original image
#         scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
#         bboxes = bboxes / max(scale, 1e-6)
#         dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)

#         # split high/second detections
#         inds_low = scores > 0.1
#         inds_high = scores < self.det_thresh
#         inds_second = np.logical_and(inds_low, inds_high)
#         dets_second = dets[inds_second]
#         remain_inds = scores > self.det_thresh
#         dets = dets[remain_inds]

#         # predict all trackers
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

#         # UA stats arrays for all trackers (aligned to self.trackers)
#         height_mus_full = np.array([trk.get_height_mu_sigma()[0] for trk in self.trackers], dtype=float)
#         height_sigmas_full = np.array([trk.get_height_mu_sigma()[1] for trk in self.trackers], dtype=float)
#         miss_counts_full = np.array([trk.time_since_update for trk in self.trackers], dtype=float)

#         # ---------- First round: MAIN matching with UA + PC ----------
#         if dets.shape[0] > 0 and trks.shape[0] > 0:
#             matched, unmatched_dets, unmatched_trks = self._associate_round1(
#                 dets, trks,
#                 height_mus_full, height_sigmas_full, miss_counts_full,
#                 img_w, img_h,
#                 self.iou_threshold,
#                 depth_emas_full = depth_emas  # 这里把 depth_emas 传进去 # 新增
#             )
                     
            
#             for d_i, t_i in matched:
#                 self.trackers[t_i].update(dets[d_i, :])
#                 # PC model update with stable sample
#                 self._update_pc_model_with_det(dets[d_i, :], img_h)
#         else:
#             matched = np.empty((0, 2), dtype=int)
#             unmatched_dets = np.arange(dets.shape[0])
#             unmatched_trks = np.arange(trks.shape[0])

#         # ---------- Second round: BYTE with UA + PC + Soft-Depth ----------
#         if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
#             u_trks = trks[unmatched_trks]               # (U,5)
#             iou_left = self.asso_func(dets_second[:, :4], u_trks[:, :4])  # (Ds, U)
#             iou_left = np.array(iou_left)

#             if iou_left.size > 0:
#                 # A_u
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
#                 # A_p
#                 A_p = self._pc_affinity(dets_second[:, :4], u_trks[:, :4], img_h)
                
#                 if self.pc_enable:
#                     Ap_dbg = A_p
#                     if self.frame_count % 50 == 0:
#                         print(f"[PC] frame={self.frame_count} count={self.perspective.global_count:.0f} "
#                             f"ready={self.perspective.ready()} A_p(mean/min/max)="
#                             f"{Ap_dbg.mean():.3f}/{Ap_dbg.min():.3f}/{Ap_dbg.max():.3f}")
                
#                 iou_geom = iou_left * A_u * A_p
#             else:
#                 iou_geom = iou_left

#             # Soft-depth
#             if iou_geom.size > 0:
#                 u_trk_emas = depth_emas[unmatched_trks]
#                 A_d = self._depth_affinity(dets_second[:, :4], u_trks[:, :4], u_trk_emas, img_h)
#                 iou_adj = self._blend_iou_with_depth(iou_geom, A_d)
#             else:
#                 iou_adj = iou_geom

#             thr2 = max(0.1, self.iou_threshold - 0.07)
#             if iou_adj.size > 0 and iou_adj.max() > thr2:
#                 matched_indices = linear_assignment(-iou_adj)
#                 to_remove_trk_indices = []
#                 to_remove_det_indices = []
#                 for m2 in matched_indices:
#                     det_local, trk_local = m2[0], m2[1]
#                     # if iou_adj[det_local, trk_local] < thr2:
#                     #     continue
                    
#                     if iou_left[det_local, trk_local] < thr2:
#                          continue        
                    
#                     trk_ind = unmatched_trks[trk_local]
#                     self.trackers[trk_ind].update(dets_second[det_local, :])
#                     to_remove_trk_indices.append(trk_ind)
#                     to_remove_det_indices.append(det_local)
#                     self._update_pc_model_with_det(dets_second[det_local, :], img_h)
#                 if len(to_remove_trk_indices) > 0:
#                     unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))
#                 # remove those dets from dets_second indexing space to avoid duplicate when computing setdiff later
#                 if len(to_remove_det_indices) > 0:
#                     keep_mask = np.ones(len(dets_second), dtype=bool)
#                     keep_mask[to_remove_det_indices] = False
#                     dets_second = dets_second[keep_mask]

#         # ---------- Third round: Re-association with UA + PC + Soft-Depth + max(IoU(last, pred)) ----------
#         if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
#             left_dets = dets[unmatched_dets]             # (Dl,5)
#             left_trks_last = last_boxes[unmatched_trks]  # (Tl,5)
#             left_trks_pred = trks[unmatched_trks]        # (Tl,5)

#             iou_last = self.asso_func(left_dets[:, :4], left_trks_last[:, :4])
#             iou_pred = self.asso_func(left_dets[:, :4], left_trks_pred[:, :4])
#             iou_last = np.array(iou_last)
#             iou_pred = np.array(iou_pred)
#             iou_max = iou_last if iou_pred.size == 0 else np.maximum(iou_last, iou_pred)

#             if iou_max.size > 0:
#                 dets_xy = left_dets[:, :4]
#                 trks_xy = left_trks_pred[:, :4]

#                 # UA
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

#                 # PC
#                 A_p = self._pc_affinity(dets_xy, trks_xy, img_h)
                
#                 if self.pc_enable:
#                     Ap_dbg = A_p
#                     if self.frame_count % 50 == 0:
#                         print(f"[PC] frame={self.frame_count} count={self.perspective.global_count:.0f} "
#                             f"ready={self.perspective.ready()} A_p(mean/min/max)="
#                             f"{Ap_dbg.mean():.3f}/{Ap_dbg.min():.3f}/{Ap_dbg.max():.3f}")
                        

#                 iou_geom = iou_max * A_u * A_p

#                 # Soft-depth
#                 trk_emas = depth_emas[unmatched_trks]
#                 A_d = self._depth_affinity(dets_xy, trks_xy, trk_emas, img_h)
#                 iou_adj = self._blend_iou_with_depth(iou_geom, A_d)
#             else:
#                 iou_adj = iou_max

#             thr3 = max(0.1, self.iou_threshold - 0.07)
#             if iou_adj.size > 0 and iou_adj.max() > thr3:
#                 rematched_indices = linear_assignment(-iou_adj)
#                 to_remove_det_indices = []
#                 to_remove_trk_indices = []
#                 for m3 in rematched_indices:
#                     det_local, trk_local = m3[0], m3[1]
#                     # if iou_adj[det_local, trk_local] < thr3:
#                     #     continue
                    
#                     if iou_max[det_local, trk_local] < thr3:
#                           continue
                    
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

#         # unmatched trackers: no measurement update
#         for m in unmatched_trks:
#             self.trackers[m].update(None)

#         # create new trackers for unmatched detections
#         for i in unmatched_dets:
#             trk = KalmanBoxTracker(
#                 dets[i, :],
#                 delta_t=self.delta_t,
#                 ema_obs=self.ema_obs,
#                 ema_pred=self.ema_pred,
#                 ua_h_ema=self.ua_h_ema,
#                 ua_h2_ema=self.ua_h2_ema,
#                 ua_proc_var_frac=self.ua_proc_var_frac,
#                 ua_sigma_floor_frac=self.ua_sigma_floor_frac
#             )
#             self.trackers.append(trk)
#             self._update_pc_model_with_det(dets[i, :], img_h)

#         # output and cleanup
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
#         """
#         Public detections tracking with UA-HMIoU + PC-HMIoU + Soft-Depth,
#         plus category-consistency gating.

#         dets: (N,4) [x1,y1,x2,y2] in image pixel coords
#         cates: (N,) integer categories
#         scores: (N,) detection confidences
#         """
#         self.frame_count += 1

#         # Compose full dets with score column kept (needed by UA)
#         dets = _to_numpy(dets, dtype=float)
#         cates = _to_numpy(cates, dtype=int).reshape(-1)
#         scores = _to_numpy(scores, dtype=float).reshape(-1)
#         assert dets.ndim == 2 and dets.shape[1] == 4, f"Expected dets (N,4), got {dets.shape}"
#         assert cates.shape[0] == dets.shape[0] == scores.shape[0], "dets/cates/scores size mismatch"

#         dets_full_all = np.concatenate([dets, scores[:, None]], axis=1)
#         cates_all = cates

#         # split high/second detections as in update()
#         remain_mask = scores > self.det_thresh
#         second_mask = (scores > 0.1) & (scores < self.det_thresh)

#         dets_main = dets_full_all[remain_mask]
#         cates_main = cates_all[remain_mask]

#         dets_second = dets_full_all[second_mask]
#         cates_second = cates_all[second_mask]

#         # predict all trackers
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

#         # prepare per-tracker stats
#         last_boxes = np.array([trk.last_observation for trk in self.trackers]) if len(self.trackers) > 0 else np.zeros((0,5))
#         depth_emas = np.array([trk.depth_ema for trk in self.trackers], dtype=float) if len(self.trackers) > 0 else np.zeros((0,), dtype=float)
#         height_mus_full = np.array([trk.get_height_mu_sigma()[0] for trk in self.trackers], dtype=float) if len(self.trackers) > 0 else np.zeros((0,), dtype=float)
#         height_sigmas_full = np.array([trk.get_height_mu_sigma()[1] for trk in self.trackers], dtype=float) if len(self.trackers) > 0 else np.zeros((0,), dtype=float)
#         miss_counts_full = np.array([trk.time_since_update for trk in self.trackers], dtype=float) if len(self.trackers) > 0 else np.zeros((0,), dtype=float)
#         trk_cates = np.array([getattr(trk, "cate", -1) for trk in self.trackers], dtype=int) if len(self.trackers) > 0 else np.zeros((0,), dtype=int)

#         # image size for PC/Depth; fallback to 1.0 if unknown
#         if self.last_img_size is not None:
#             img_h, img_w = _to_scalar(self.last_img_size[0]), _to_scalar(self.last_img_size[1])
#         else:
#             img_h, img_w = 1.0, 1.0

#         # ---------- Round 1: MAIN (with categories) ----------
#         matched_main = np.empty((0,2), dtype=int)
#         unmatched_dets_main = np.arange(dets_main.shape[0])
#         unmatched_trks = np.arange(trks.shape[0])

#         if dets_main.shape[0] > 0 and trks.shape[0] > 0:
#             dets_xy = dets_main[:, :4]
#             trks_xy = trks[:, :4]
#             iou = np.array(self.asso_func(dets_xy, trks_xy))
#             # UA
#             if self.ua_enable:
#                 A_u = self._ua_height_affinity(
#                     dets_xy, trks_xy,
#                     height_mus_full, height_sigmas_full,
#                     dets_main[:, 4], miss_counts_full,
#                     img_w, img_h
#                 )
#             else:
#                 A_u = np.ones_like(iou)
#             # PC
#             A_p = self._pc_affinity(dets_xy, trks_xy, img_h)
#             # Soft-Depth
#             A_d = self._depth_affinity(dets_xy, trks_xy, depth_emas, img_h)
#             S = iou * A_u * A_p
#             S = self._blend_iou_with_depth(S, A_d)

#             # category mask: allow if trk_cate < 0 (unset) or equal to det cate
#             if trk_cates.shape[0] > 0:
#                 same_cat = (cates_main[:, None] == trk_cates[None, :]) | (trk_cates[None, :] < 0)
#                 S = np.where(same_cat, S, 0.0)

#             # Hungarian on -S, accept only if pure geometric IoU >= thr and category ok
#             matched = []
#             if S.size > 0 and iou.max() > self.iou_threshold:
#                 idxs = linear_assignment(-S)
#                 for d_i, t_i in idxs:
#                     if iou[d_i, t_i] >= self.iou_threshold:
#                         # also ensure category-allowed
#                         if trk_cates.shape[0] == 0 or (trk_cates[t_i] < 0 or trk_cates[t_i] == cates_main[d_i]):
#                             matched.append([d_i, t_i])
#             matched_main = np.asarray(matched, dtype=int) if len(matched) > 0 else np.empty((0,2), dtype=int)

#             md = matched_main[:,0] if matched_main.size>0 else np.array([], dtype=int)
#             mt = matched_main[:,1] if matched_main.size>0 else np.array([], dtype=int)
#             unmatched_dets_main = np.setdiff1d(np.arange(dets_main.shape[0]), md, assume_unique=False)
#             unmatched_trks = np.setdiff1d(np.arange(trks.shape[0]), mt, assume_unique=False)

#             # apply updates for matched pairs (round-1)
#             for d_i, t_i in matched_main:
#                 self.trackers[t_i].update(dets_main[d_i, :])
#                 # set/update category for the tracker
#                 setattr(self.trackers[t_i], "cate", int(cates_main[d_i]))
#                 # update PC model
#                 self._update_pc_model_with_det(dets_main[d_i, :], img_h)

#         # ---------- Round 2: BYTE (low-score) with UA + PC + Soft-Depth + category gating ----------
#         if self.use_byte and dets_second.shape[0] > 0 and unmatched_trks.shape[0] > 0:
#             u_trks = trks[unmatched_trks]  # (U,5)
#             iou_left = np.array(self.asso_func(dets_second[:, :4], u_trks[:, :4]))  # (Ds, U)

#             if iou_left.size > 0:
#                 # A_u
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

#                 # A_p
#                 A_p = self._pc_affinity(dets_second[:, :4], u_trks[:, :4], img_h)

#                 # Soft-Depth
#                 u_trk_emas = depth_emas[unmatched_trks]
#                 A_d = self._depth_affinity(dets_second[:, :4], u_trks[:, :4], u_trk_emas, img_h)

#                 iou_geom = iou_left * A_u * A_p
#                 iou_adj = self._blend_iou_with_depth(iou_geom, A_d)

#                 # category gating (allow if tracker cate unset or equal)
#                 u_trk_cates = trk_cates[unmatched_trks]
#                 cat_mask = (cates_second[:, None] == u_trk_cates[None, :]) | (u_trk_cates[None, :] < 0)
#                 iou_adj = np.where(cat_mask, iou_adj, 0.0)
#             else:
#                 iou_adj = iou_left

#             thr2 = max(0.1, self.iou_threshold - 0.07)
#             if iou_adj.size > 0 and iou_adj.max() > thr2:
#                 matched_indices = linear_assignment(-iou_adj)
#                 to_remove_trk_indices = []
#                 to_remove_det_indices = []
#                 for det_local, trk_local in matched_indices:
#                     # geometric IoU gate and category gate
#                     if iou_left[det_local, trk_local] < thr2:
#                         continue
#                     trk_ind = unmatched_trks[trk_local]
#                     # also ensure category consistency
#                     if trk_cates.shape[0] > 0:
#                         trk_c = trk_cates[trk_ind]
#                         if not (trk_c < 0 or trk_c == cates_second[det_local]):
#                             continue
#                     # apply update
#                     self.trackers[trk_ind].update(dets_second[det_local, :])
#                     setattr(self.trackers[trk_ind], "cate", int(cates_second[det_local]))
#                     self._update_pc_model_with_det(dets_second[det_local, :], img_h)
#                     to_remove_trk_indices.append(trk_ind)
#                     to_remove_det_indices.append(det_local)

#                 if len(to_remove_trk_indices) > 0:
#                     unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))
#                 if len(to_remove_det_indices) > 0:
#                     keep_mask = np.ones(len(dets_second), dtype=bool)
#                     keep_mask[to_remove_det_indices] = False
#                     dets_second = dets_second[keep_mask]
#                     cates_second = cates_second[keep_mask]

#         # ---------- Round 3: Re-association with UA + PC + Soft-Depth + max(IoU(last, pred)) + category gating ----------
#         if unmatched_dets_main.shape[0] > 0 and unmatched_trks.shape[0] > 0:
#             left_dets = dets_main[unmatched_dets_main]            # (Dl,5)
#             left_cates = cates_main[unmatched_dets_main]          # (Dl,)
#             left_trks_last = last_boxes[unmatched_trks]           # (Tl,5)
#             left_trks_pred = trks[unmatched_trks]                 # (Tl,5)

#             iou_last = np.array(self.asso_func(left_dets[:, :4], left_trks_last[:, :4]))
#             iou_pred = np.array(self.asso_func(left_dets[:, :4], left_trks_pred[:, :4]))
#             iou_max = iou_last if iou_pred.size == 0 else np.maximum(iou_last, iou_pred)

#             if iou_max.size > 0:
#                 dets_xy = left_dets[:, :4]
#                 trks_xy = left_trks_pred[:, :4]

#                 # UA
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

#                 # PC
#                 A_p = self._pc_affinity(dets_xy, trks_xy, img_h)

#                 # Soft-Depth
#                 trk_emas = depth_emas[unmatched_trks]
#                 A_d = self._depth_affinity(dets_xy, trks_xy, trk_emas, img_h)

#                 iou_geom = iou_max * A_u * A_p
#                 iou_adj = self._blend_iou_with_depth(iou_geom, A_d)

#                 # category gating (allow if tracker cate unset or equal)
#                 u_trk_cates = trk_cates[unmatched_trks]
#                 cat_mask = (left_cates[:, None] == u_trk_cates[None, :]) | (u_trk_cates[None, :] < 0)
#                 iou_adj = np.where(cat_mask, iou_adj, 0.0)
#             else:
#                 iou_adj = iou_max

#             thr3 = max(0.1, self.iou_threshold - 0.07)
#             if iou_adj.size > 0 and iou_adj.max() > thr3:
#                 rematched_indices = linear_assignment(-iou_adj)
#                 to_remove_det_indices = []
#                 to_remove_trk_indices = []
#                 for det_local, trk_local in rematched_indices:
#                     if iou_max[det_local, trk_local] < thr3:
#                         continue
#                     det_ind = unmatched_dets_main[det_local]
#                     trk_ind = unmatched_trks[trk_local]
#                     # ensure category consistency
#                     trk_c = trk_cates[trk_ind]
#                     if not (trk_c < 0 or trk_c == cates_main[det_ind]):
#                         continue
#                     # apply update
#                     self.trackers[trk_ind].update(dets_main[det_ind, :])
#                     setattr(self.trackers[trk_ind], "cate", int(cates_main[det_ind]))
#                     self._update_pc_model_with_det(dets_main[det_ind, :], img_h)
#                     to_remove_det_indices.append(det_ind)
#                     to_remove_trk_indices.append(trk_ind)

#                 if len(to_remove_det_indices) > 0:
#                     unmatched_dets_main = np.setdiff1d(unmatched_dets_main, np.array(to_remove_det_indices))
#                 if len(to_remove_trk_indices) > 0:
#                     unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

#         # unmatched trackers: no measurement update (inflate UA variance)
#         for t_idx in unmatched_trks:
#             self.trackers[t_idx].update(None)

#         # create new trackers for unmatched detections (from main and second pools)
#         for i_local in unmatched_dets_main:
#             trk = KalmanBoxTracker(
#                 dets_main[i_local, :],
#                 delta_t=self.delta_t,
#                 ema_obs=self.ema_obs,
#                 ema_pred=self.ema_pred,
#                 ua_h_ema=self.ua_h_ema,
#                 ua_h2_ema=self.ua_h2_ema,
#                 ua_proc_var_frac=self.ua_proc_var_frac,
#                 ua_sigma_floor_frac=self.ua_sigma_floor_frac
#             )
#             trk.cate = int(cates_main[i_local])
#             self.trackers.append(trk)
#             self._update_pc_model_with_det(dets_main[i_local, :], img_h)

#         # any remaining second detections become new tracks too
#         if dets_second.shape[0] > 0:
#             for j in range(dets_second.shape[0]):
#                 trk = KalmanBoxTracker(
#                     dets_second[j, :],
#                     delta_t=self.delta_t,
#                     ema_obs=self.ema_obs,
#                     ema_pred=self.ema_pred,
#                     ua_h_ema=self.ua_h_ema,
#                     ua_h2_ema=self.ua_h2_ema,
#                     ua_proc_var_frac=self.ua_proc_var_frac,
#                     ua_sigma_floor_frac=self.ua_sigma_floor_frac
#                 )
#                 trk.cate = int(cates_second[j])
#                 self.trackers.append(trk)
#                 self._update_pc_model_with_det(dets_second[j, :], img_h)

#         # output and cleanup (same as original update_public behavior)
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