# """
#     This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
#     Modified: Replace Kalman filter with Diffusion (D2MP) for motion prediction.
# """
# from __future__ import print_function

# import os
# import numpy as np
# import torch

# from .association import *
# from .diffusion import VarianceSchedule, HMINet, D2MP_OB

# def _to_int(x):
#     # 将 1D 或 0D 的 torch.Tensor 转为 Python int；否则直接 int(x)
#     if isinstance(x, torch.Tensor):
#         return int(x.reshape(-1)[0].item())
#     return int(x)


# # -----------------------------
# # Helpers kept from original OC-SORT
# # -----------------------------
# def k_previous_obs(observations, cur_age, k):
#     if len(observations) == 0:
#         return np.array([-1, -1, -1, -1, -1], dtype=np.float32)
#     for i in range(k):
#         dt = k - i
#         if cur_age - dt in observations:
#             return observations[cur_age - dt]
#     max_age = max(observations.keys())
#     return observations[max_age]


# def convert_bbox_to_z(bbox):
#     w = bbox[2] - bbox[0]
#     h = bbox[3] - bbox[1]
#     x = bbox[0] + w/2.
#     y = bbox[1] + h/2.
#     s = w * h
#     r = w / float(h + 1e-6)
#     return np.array([x, y, s, r]).reshape((4, 1))


# def convert_x_to_bbox(x, score=None):
#     w = np.sqrt(x[2] * x[3])
#     h = x[2] / w
#     if(score is None):
#         return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
#     else:
#         return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


# def speed_direction(bbox1, bbox2):
#     cx1, cy1 = (bbox1[0]+bbox1[2]) / 2.0, (bbox1[1]+bbox1[3])/2.0
#     cx2, cy2 = (bbox2[0]+bbox2[2]) / 2.0, (bbox2[1]+bbox2[3])/2.0
#     speed = np.array([cy2-cy1, cx2-cx1], dtype=np.float32)
#     norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
#     return speed / norm


# """
#     We support multiple ways for association cost calculation, by default
#     we use IoU. GIoU may have better performance in some situations. We note 
#     that we hardly normalize the cost by all methods to (0,1) which may not be 
#     the best practice.
# """
# ASSO_FUNCS = {  "iou": iou_batch,
#                 "giou": giou_batch,
#                 "ciou": ciou_batch,
#                 "diou": diou_batch,
#                 "ct_dist": ct_dist}


# # -----------------------------
# # D2MP config + utilities
# # -----------------------------
# class D2MPConfig:
#     def __init__(self):
#         # diffusion forward eps
#         self.eps = 1e-5
#         # bbox format used during training: 'xyxy' or 'cxcywh'
#         self.bbox_format = 'xyxy'
#         # normalization after unitizing to [0,1]
#         self.bbox_mean = [0.5, 0.5, 0.5, 0.5]
#         self.bbox_std  = [0.5, 0.5, 0.5, 0.5]
#         # will be inferred from weight if possible
#         self.context_dim = None
#         self.hist_len = 5
#         # diffusion schedule defaults (train-time)
#         self.num_steps = 20 #扩散采样步数较多 + 按 tracklet 单个调用
#         self.beta_1 = 1e-4
#         self.beta_T = 5e-2
#         self.var_mode = 'linear'
#         # model structure defaults (train-time)
#         self.point_dim = 4
#         self.tf_layer = 3
#         # deterministic seed for stability (optional)
#         self.seed = 12345  # or None


# d2mp_config = D2MPConfig()


# def xyxy_to_cxcywh(xyxy):
#     x1, y1, x2, y2 = xyxy
#     w = max(x2 - x1, 1e-6)
#     h = max(y2 - y1, 1e-6)
#     cx = x1 + w / 2.0
#     cy = y1 + h / 2.0
#     return np.array([cx, cy, w, h], dtype=np.float32)


# def cxcywh_to_xyxy(cxcywh):
#     cx, cy, w, h = cxcywh
#     x1 = cx - w / 2.0
#     y1 = cy - h / 2.0
#     x2 = cx + w / 2.0
#     y2 = cy + h / 2.0
#     return np.array([x1, y1, x2, y2], dtype=np.float32)


# # def to_unit_xyxy(xyxy, img_size):
# #     # img_size: (H, W)
# #     H, W = img_size
# #     x1, y1, x2, y2 = xyxy
# #     return np.array([x1 / W, y1 / H, x2 / W, y2 / H], dtype=np.float32)


# # def to_pixel_xyxy(unit_xyxy, img_size):
# #     H, W = img_size
# #     x1, y1, x2, y2 = unit_xyxy
# #     return np.array([x1 * W, y1 * H, x2 * W, y2 * H], dtype=np.float32)


# # def clamp_xyxy(xyxy, img_size):
# #     H, W = img_size
# #     x1, y1, x2, y2 = xyxy
# #     x1 = max(0, min(W - 1, x1))
# #     y1 = max(0, min(H - 1, y1))
# #     x2 = max(0, min(W - 1, x2))
# #     y2 = max(0, min(H - 1, y2))
# #     if x2 < x1: x2 = x1
# #     if y2 < y1: y2 = y1
# #     return np.array([x1, y1, x2, y2], dtype=np.float32)

# def to_unit_xyxy(xyxy, img_size):
#     H, W = img_size
#     H, W = _to_int(H), _to_int(W)
#     x1, y1, x2, y2 = xyxy
#     return np.array([x1 / W, y1 / H, x2 / W, y2 / H], dtype=np.float32)

# def to_pixel_xyxy(unit_xyxy, img_size):
#     H, W = img_size
#     H, W = _to_int(H), _to_int(W)
#     x1, y1, x2, y2 = unit_xyxy
#     return np.array([x1 * W, y1 * H, x2 * W, y2 * H], dtype=np.float32)

# def clamp_xyxy(xyxy, img_size):
#     H, W = img_size
#     H, W = _to_int(H), _to_int(W)
#     x1, y1, x2, y2 = xyxy
#     x1 = max(0, min(W - 1, x1))
#     y1 = max(0, min(H - 1, y1))
#     x2 = max(0, min(W - 1, x2))
#     y2 = max(0, min(H - 1, y2))
#     if x2 < x1: x2 = x1
#     if y2 < y1: y2 = y1
#     return np.array([x1, y1, x2, y2], dtype=np.float32)

# # -----------------------------
# # Global D2MP model init (lazy)
# # -----------------------------
# _global_d2mp = None


# def _infer_context_dim_and_hist_len_from_state(state):
#     # Try to infer context_dim from concat1._hyper_bias.weight in HMINet
#     # expected key like 'net.concat1._hyper_bias.weight' (or without 'net.')
#     ctx_dim = None
#     for k, v in state.items():
#         if k.endswith('concat1._hyper_bias.weight') and v.ndim == 2:
#             ctx_dim = int(v.shape[1]) - 3  # minus time_emb (3 dims)
#             break
#     if ctx_dim is not None and ctx_dim > 0:
#         d2mp_config.context_dim = ctx_dim
#         # hist_len is context_dim // 4 if context is flattened L*4
#         if ctx_dim % 4 == 0:
#             d2mp_config.hist_len = ctx_dim // 4
#         # else leave default and trust user's configuration
#     # else leave defaults


# # def init_d2mp_model():
# #     global _global_d2mp
# #     if _global_d2mp is not None:
# #         return _global_d2mp

# #     # optional: deterministic behavior
# #     if d2mp_config.seed is not None:
# #         torch.manual_seed(d2mp_config.seed)
# #         np.random.seed(d2mp_config.seed)

# #     # build weight path robustly
# #     this_dir = os.path.dirname(os.path.abspath(__file__))
# #     weight_path = "/public/home/cookie/anaconda3/envs/ocsort_env/OC_SORT/trackers/diffmot_tracker/weights/MOT_epoch800.pt"

# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     state = torch.load(weight_path, map_location=device)

# #     # unwrap common containers
# #     if isinstance(state, dict):
# #         # try common wrappers
# #         if 'state_dict' in state and isinstance(state['state_dict'], dict):
# #             state = state['state_dict']
# #         elif 'model' in state and isinstance(state['model'], dict):
# #             state = state['model']

# #     # strip 'module.' prefix if any
# #     needs_strip = any(k.startswith('module.') for k in state.keys())
# #     if needs_strip:
# #         state = {k.replace('module.', '', 1): v for k, v in state.items()}

# #     # infer context_dim/hist_len if possible
# #     _infer_context_dim_and_hist_len_from_state(state)

# #     # fallback if not inferred
# #     context_dim = d2mp_config.context_dim if d2mp_config.context_dim is not None else (d2mp_config.hist_len * 4)

# #     var_sched = VarianceSchedule(
# #         num_steps=d2mp_config.num_steps,
# #         mode=d2mp_config.var_mode,
# #         beta_1=d2mp_config.beta_1,
# #         beta_T=d2mp_config.beta_T
# #     )
# #     hminet = HMINet(
# #         point_dim=d2mp_config.point_dim,
# #         context_dim=context_dim,
# #         tf_layer=d2mp_config.tf_layer
# #     )
# #     model = D2MP_OB(
# #         net=hminet,
# #         var_sched=var_sched,
# #         config=d2mp_config
# #     )

# #     # try strict load, fallback
# #     try:
# #         model.load_state_dict(state, strict=True)
# #     except Exception as e:
# #         print(f"[D2MP] strict load_state_dict failed: {e}\nTrying non-strict loading...")
# #         model.load_state_dict(state, strict=False)

# #     model.eval().to(device)
# #     _global_d2mp = model
# #     return _global_d2mp

# def init_d2mp_model():
#     """
#     Lazy init global D2MP model with robust checkpoint parsing.
#     - Uses absolute weight path provided by you.
#     - Unwraps common wrappers: state_dict / model / ddpm
#     - Strips 'module.' prefixes if present
#     - Extracts 'net' and optional 'var_sched' weights
#     - Infers context_dim / hist_len from weights if possible
#     """
#     global _global_d2mp
#     if _global_d2mp is not None:
#         return _global_d2mp

#     # Optional determinism
#     if d2mp_config.seed is not None:
#         torch.manual_seed(d2mp_config.seed)
#         np.random.seed(d2mp_config.seed)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Use your absolute path
#     weight_path = "/public/home/cookie/anaconda3/envs/ocsort_env/OC_SORT/trackers/diffmot_tracker/weights/MOT_epoch800.pt"

#     # 1) Load raw checkpoint (prefer weights_only=True; fallback for older PyTorch)
#     try:
#         state_raw = torch.load(weight_path, map_location=device, weights_only=True)
#     except TypeError:
#         state_raw = torch.load(weight_path, map_location=device)

#     # 2) Unwrap common containers
#     if isinstance(state_raw, dict):
#         if 'state_dict' in state_raw and isinstance(state_raw['state_dict'], dict):
#             state_raw = state_raw['state_dict']
#         elif 'model' in state_raw and isinstance(state_raw['model'], dict):
#             state_raw = state_raw['model']
#         elif 'ddpm' in state_raw and isinstance(state_raw['ddpm'], dict):
#             inner = state_raw['ddpm']
#             if isinstance(inner, dict) and 'state_dict' in inner and isinstance(inner['state_dict'], dict):
#                 state_raw = inner['state_dict']
#             else:
#                 state_raw = inner

#     # 3) Strip 'module.' prefix if any (DDP saved checkpoints)
#     if isinstance(state_raw, dict) and any(isinstance(k, str) and k.startswith('module.') for k in state_raw.keys()):
#         state_raw = {k.replace('module.', '', 1): v for k, v in state_raw.items()}

#     # 4) Build a flat state dict with expected prefixes: 'net.' and optional 'var_sched.'
#     state = {}
#     if isinstance(state_raw, dict):
#         # Case A: nested under 'net'
#         if 'net' in state_raw and isinstance(state_raw['net'], dict):
#             for k, v in state_raw['net'].items():
#                 key = f'net.{k}' if not k.startswith('net.') else k
#                 state[key] = v
#         # Case B: already flattened with 'net.' prefix
#         elif any(isinstance(k, str) and k.startswith('net.') for k in state_raw.keys()):
#             for k, v in state_raw.items():
#                 if k.startswith('net.'):
#                     state[k] = v
#         else:
#             # Case C: raw layer names without 'net.' (heuristic)
#             for k, v in state_raw.items():
#                 if any(k.startswith(p) for p in ['optimizer', 'epoch', 'ema', 'amp', 'scaler', 'global_step']):
#                     continue
#                 if k.startswith('var_sched.'):
#                     state[k] = v
#                 else:
#                     state[f'net.{k}'] = v

#         # Optional: var_sched (nested or flattened)
#         if 'var_sched' in state_raw and isinstance(state_raw['var_sched'], dict):
#             for k, v in state_raw['var_sched'].items():
#                 key = f'var_sched.{k}' if not k.startswith('var_sched.') else k
#                 state[key] = v
#         elif any(isinstance(k, str) and k.startswith('var_sched.') for k in state_raw.keys()):
#             for k, v in state_raw.items():
#                 if k.startswith('var_sched.'):
#                     state[k] = v

#     # 5) Logging (optional but helpful)
#     num_net = sum(1 for k in state.keys() if isinstance(k, str) and k.startswith('net.'))
#     num_sched = sum(1 for k in state.keys() if isinstance(k, str) and k.startswith('var_sched.'))
#     if num_net == 0:
#         print('[D2MP] WARNING: Could not find net weights in checkpoint. '
#               'Tried keys: state_dict/model/ddpm with subkeys net/var_sched. '
#               'Please inspect the checkpoint structure.')
#     else:
#         print(f'[D2MP] Parsed weights: net keys = {num_net}, var_sched keys = {num_sched}')

#     # 6) Infer context_dim / hist_len from extracted weights if possible
#     _infer_context_dim_and_hist_len_from_state(state)
#     context_dim = d2mp_config.context_dim if d2mp_config.context_dim is not None else (d2mp_config.hist_len * 4)

#     # 7) Build model
#     var_sched = VarianceSchedule(
#         num_steps=d2mp_config.num_steps,
#         mode=d2mp_config.var_mode,
#         beta_1=d2mp_config.beta_1,
#         beta_T=d2mp_config.beta_T
#     )
#     hminet = HMINet(
#         point_dim=d2mp_config.point_dim,
#         context_dim=context_dim,
#         tf_layer=d2mp_config.tf_layer
#     )
#     model = D2MP_OB(
#         net=hminet,
#         var_sched=var_sched,
#         config=d2mp_config
#     )

#     # 8) Load weights (non-strict to allow missing buffers like var_sched.*)
#     try:
#         model.load_state_dict(state, strict=False)
#     except Exception as e:
#         print(f"[D2MP] load_state_dict failed: {e}")

#     model.eval().to(device)
#     _global_d2mp = model
#     return _global_d2mp

# # -----------------------------
# # Diffusion-based tracker
# # -----------------------------
# class DiffusionBoxTracker(object):
#     """
#     D2MP-based tracker replacing Kalman filter.
#     Maintains same external interface for OC-SORT association.
#     """
#     count = 0

#     def __init__(self, bbox, img_size, delta_t=3):
#         """
#         bbox: [x1, y1, x2, y2, score] in pixel coords
#         img_size: (H, W)
#         """
#         self.img_size = img_size  # (H, W)
#         # internal state
#         self.history_bboxes = [bbox[:4].astype(np.float32)]
#         self.last_bbox = bbox[:4].astype(np.float32)  # last predicted/observed bbox (pixel)
#         # OC-SORT required fields
#         self.time_since_update = 0
#         self.id = DiffusionBoxTracker.count
#         DiffusionBoxTracker.count += 1
#         self.hits = 0
#         self.hit_streak = 0
#         self.age = 0

#         # placeholders and history for OC-SORT
#         self.last_observation = np.array([-1, -1, -1, -1, -1], dtype=np.float32)  # same shape as OC-SORT
#         self.observations = dict()
#         self.history_observations = []
#         self.velocity = None
#         self.delta_t = delta_t
#         # immediately update with the init bbox as an observation
#         self.update(bbox, img_size=img_size)

#     # def set_img_size(self, img_size):
#     #     self.img_size = img_size
#     def set_img_size(self, img_size):
#         H, W = img_size
#         H, W = _to_int(H), _to_int(W)
#         self.img_size = (H, W)

#     # normalization helpers
#     def _normalize_tensor(self, bbox_tensor):
#         mean = torch.tensor(d2mp_config.bbox_mean, device=bbox_tensor.device, dtype=bbox_tensor.dtype)
#         std  = torch.tensor(d2mp_config.bbox_std,  device=bbox_tensor.device, dtype=bbox_tensor.dtype)
#         return (bbox_tensor - mean) / std

#     def _denormalize_tensor(self, norm_tensor):
#         mean = torch.tensor(d2mp_config.bbox_mean, device=norm_tensor.device, dtype=norm_tensor.dtype)
#         std  = torch.tensor(d2mp_config.bbox_std,  device=norm_tensor.device, dtype=norm_tensor.dtype)
#         return norm_tensor * std + mean

#     def update(self, bbox, img_size=None):
#         """
#         Update with detection bbox (x1,y1,x2,y2,score) or None.
#         """
#         if img_size is not None:
#             self.set_img_size(img_size)

#         if bbox is not None:
#             bbox = bbox.astype(np.float32)
#             # estimate velocity using observation delta_t steps away
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

#             # maintain obs history like original OC-SORT
#             self.last_observation = bbox.copy()
#             self.observations[self.age] = bbox.copy()
#             self.history_observations.append(bbox.copy())

#             # maintain bbox history for D2MP (store only xyxy)
#             self.last_bbox = bbox[:4].copy()
#             self.history_bboxes.append(self.last_bbox.copy())
#             # keep last hist_len frames
#             if len(self.history_bboxes) > d2mp_config.hist_len:
#                 self.history_bboxes = self.history_bboxes[-d2mp_config.hist_len:]

#             # OC-SORT counters
#             self.time_since_update = 0
#             self.hits += 1
#             self.hit_streak += 1
#         else:
#             # no observation this frame
#             self.time_since_update += 1

#     def predict(self):
#         """
#         Predict next bbox with D2MP. Does NOT overwrite last_observation.
#         Returns shape (1,4) in pixel coords.
#         """
#         # model and device
#         model = init_d2mp_model()
#         device = next(model.parameters()).device

#         # history prep
#         H, W = self.img_size
#         hist = self.history_bboxes[-d2mp_config.hist_len:]
#         if len(hist) < d2mp_config.hist_len:
#             hist = hist + [hist[-1]] * (d2mp_config.hist_len - len(hist))

#         # pixel -> unit
#         hist_unit = [to_unit_xyxy(b, (H, W)) for b in hist]
#         hist_unit = np.stack(hist_unit, axis=0)  # (L,4)

#         # format conversion if trained on cxcywh
#         if d2mp_config.bbox_format == 'cxcywh':
#             hist_unit = np.stack([xyxy_to_cxcywh(b) for b in hist_unit], axis=0)

#         # normalize
#         inp = torch.tensor(hist_unit, dtype=torch.float32, device=device)  # (L,4)
#         inp = self._normalize_tensor(inp)  # normalize per coordinate
#         context = inp.flatten().unsqueeze(0)  # (1, L*4)

#         # sampling
#         with torch.no_grad():
#             pred = model.sample(
#                 context=context,
#                 sample=1,
#                 bestof=False,
#                 point_dim=d2mp_config.point_dim,
#                 ret_traj=False
#             )  # expected shape: (S=1, B=1, 4)

#         if isinstance(pred, (list, tuple)):
#             pred = pred[0]
#         pred = pred[0, 0] if pred.ndim == 3 else pred.squeeze(0)  # (4,)
#         pred = self._denormalize_tensor(pred)  # back to unit-scale bbox (format per config)

#         # format back to xyxy if needed (still unit scale)
#         pred_unit = pred.detach().cpu().numpy()
#         if d2mp_config.bbox_format == 'cxcywh':
#             pred_unit = cxcywh_to_xyxy(pred_unit)

#         # unit -> pixel, clamp
#         pred_px = to_pixel_xyxy(pred_unit, (H, W))
#         pred_px = clamp_xyxy(pred_px, (H, W))

#         # advance OC-SORT counters (as in KalmanBoxTracker.predict)
#         self.age += 1
#         if self.time_since_update > 0:
#             self.hit_streak = 0
#         self.time_since_update += 1

#         # cache last predicted bbox (separate from last_observation)
#         self.last_bbox = pred_px.astype(np.float32)

#         return self.last_bbox.reshape(1, 4)

#     def get_state(self):
#         """
#         Returns the current bbox estimate (prediction), shape (1,4)
#         """
#         return self.last_bbox.reshape(1, 4)


# # -----------------------------
# # OCSort using DiffusionBoxTracker
# # -----------------------------
# class Diff_OCSort(object):
#     def __init__(self, det_thresh, max_age=30, min_hits=3, 
#         iou_threshold=0.3, delta_t=3, asso_func="iou", inertia=0.2, use_byte=False):
#         """
#         Sets key parameters for SORT
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

#         # reset diffusion-based tracker id counter
#         DiffusionBoxTracker.count = 0

#         # remember last image size (for update_public fallback)
#         self.last_img_size = None  # (H, W)

#     def update(self, output_results, img_info, img_size):
#         """
#         Params:
#           output_results: numpy array of detections [[x1,y1,x2,y2,score], ...]
#           img_info: (img_h, img_w, ...)
#           img_size: (test_input_h, test_input_w) - for scaling back
#         """
#         if output_results is None:
#             return np.empty((0, 5))

#         self.frame_count += 1
#         # post_process detections
#         if output_results.shape[1] == 5:
#             scores = output_results[:, 4]
#             bboxes = output_results[:, :4]
#         else:
#             output_results = output_results.cpu().numpy()
#             scores = output_results[:, 4] * output_results[:, 5]
#             bboxes = output_results[:, :4]  # x1y1x2y2
#         # img_h, img_w = img_info[0], img_info[1]
#         # self.last_img_size = (img_h, img_w)  # remember for public update
#         # scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
#         img_h, img_w = _to_int(img_info[0]), _to_int(img_info[1])
#         self.last_img_size = (img_h, img_w)
#         scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        
#         bboxes = bboxes / scale
#         dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)

#         inds_low = scores > 0.1
#         inds_high = scores < self.det_thresh
#         inds_second = np.logical_and(inds_low, inds_high)
#         dets_second = dets[inds_second]
#         remain_inds = scores > self.det_thresh
#         dets = dets[remain_inds]

#         # predict from existing trackers
#         trks = np.zeros((len(self.trackers), 5), dtype=np.float32)
#         to_del = []
#         ret = []
#         for t, trk in enumerate(self.trackers):
#             pos = trk.predict()[0]  # (4,)
#             trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
#             if np.any(np.isnan(pos)):
#                 to_del.append(t)
#         trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#         for t in reversed(to_del):
#             self.trackers.pop(t)

#         velocities = np.array(
#             [trk.velocity if trk.velocity is not None else np.array((0, 0), dtype=np.float32) for trk in self.trackers])
#         last_boxes = np.array([trk.last_observation for trk in self.trackers])
#         k_observations = np.array(
#             [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

#         # First round of association
#         matched, unmatched_dets, unmatched_trks = associate(
#             dets, trks, self.iou_threshold, velocities, k_observations, self.inertia)
#         for m in matched:
#             # pass img_size for trackers to keep resolution
#             self.trackers[m[1]].update(dets[m[0], :], img_size=(img_h, img_w))

#         # Second round of association by OCR (BYTE optional)
#         if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
#             u_trks = trks[unmatched_trks]
#             iou_left = self.asso_func(dets_second, u_trks)
#             iou_left = np.array(iou_left)
#             if iou_left.max() > self.iou_threshold:
#                 matched_indices = linear_assignment(-iou_left)
#                 to_remove_trk_indices = []
#                 for m in matched_indices:
#                     det_ind, trk_ind = m[0], unmatched_trks[m[1]]
#                     if iou_left[m[0], m[1]] < self.iou_threshold:
#                         continue
#                     self.trackers[trk_ind].update(dets_second[det_ind, :], img_size=(img_h, img_w))
#                     to_remove_trk_indices.append(trk_ind)
#                 unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

#         if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
#             left_dets = dets[unmatched_dets]
#             left_trks = last_boxes[unmatched_trks]
#             iou_left = self.asso_func(left_dets, left_trks)
#             iou_left = np.array(iou_left)
#             if iou_left.max() > self.iou_threshold:
#                 rematched_indices = linear_assignment(-iou_left)
#                 to_remove_det_indices = []
#                 to_remove_trk_indices = []
#                 for m in rematched_indices:
#                     det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
#                     if iou_left[m[0], m[1]] < self.iou_threshold:
#                         continue
#                     self.trackers[trk_ind].update(dets[det_ind, :], img_size=(img_h, img_w))
#                     to_remove_det_indices.append(det_ind)
#                     to_remove_trk_indices.append(trk_ind)
#                 unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
#                 unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

#         for m in unmatched_trks:
#             self.trackers[m].update(None)

#         # create and initialise new trackers
#         for i in unmatched_dets:
#             trk = DiffusionBoxTracker(dets[i, :], img_size=(img_h, img_w), delta_t=self.delta_t)
#             self.trackers.append(trk)

#         i = len(self.trackers)
#         for trk in reversed(self.trackers):
#             if trk.last_observation.sum() < 0:
#                 d = trk.get_state()[0]
#             else:
#                 # optional: use recent observation or predicted
#                 d = trk.last_observation[:4]
#             if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
#                 # +1 as MOT benchmark requires positive
#                 ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
#             i -= 1
#             # remove dead tracklet
#             if (trk.time_since_update > self.max_age):
#                 self.trackers.pop(i)
#         if (len(ret) > 0):
#             return np.concatenate(ret)
#         return np.empty((0, 5))

#     def update_public(self, dets, cates, scores):
#         """
#         Public detections variant.
#         Note: will use last known image size from update(); if None, falls back to (1080, 1920).
#         """
#         self.frame_count += 1

#         det_scores = np.ones((dets.shape[0], 1), dtype=np.float32)
#         dets = np.concatenate((dets, det_scores), axis=1)

#         remain_inds = scores > self.det_thresh
#         cates = cates[remain_inds]
#         dets = dets[remain_inds]

#         trks = np.zeros((len(self.trackers), 5), dtype=np.float32)
#         to_del = []
#         ret = []
#         for t, trk in enumerate(self.trackers):
#             pos = trk.predict()[0]
#             cat = getattr(trk, 'cate', 0)
#             trks[t, :] = [pos[0], pos[1], pos[2], pos[3], cat]
#             if np.any(np.isnan(pos)):
#                 to_del.append(t)
#         trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#         for t in reversed(to_del):
#             self.trackers.pop(t)

#         velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0,0), dtype=np.float32) for trk in self.trackers])
#         last_boxes = np.array([trk.last_observation for trk in self.trackers])
#         k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

#         matched, unmatched_dets, unmatched_trks = associate_kitti\
#               (dets, trks, cates, self.iou_threshold, velocities, k_observations, self.inertia)
          
#         for m in matched:
#             # use last known image size or a sensible default
#             img_size = self.last_img_size if self.last_img_size is not None else (1080, 1920)
#             self.trackers[m[1]].update(dets[m[0], :], img_size=img_size)
          
#         if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
#             left_dets = dets[unmatched_dets]
#             left_trks = last_boxes[unmatched_trks]
#             left_dets_c = left_dets.copy()
#             left_trks_c = left_trks.copy()

#             iou_left = self.asso_func(left_dets_c, left_trks_c)
#             iou_left = np.array(iou_left)
#             det_cates_left = cates[unmatched_dets]
#             trk_cates_left = trks[unmatched_trks][:,4]
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
#                     img_size = self.last_img_size if self.last_img_size is not None else (1080, 1920)
#                     self.trackers[trk_ind].update(dets[det_ind, :], img_size=img_size)
#                     to_remove_det_indices.append(det_ind)
#                     to_remove_trk_indices.append(trk_ind) 
#                 unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
#                 unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

#         for i in unmatched_dets:
#             img_size = self.last_img_size if self.last_img_size is not None else (1080, 1920)
#             trk = DiffusionBoxTracker(dets[i,:], img_size=img_size, delta_t=self.delta_t)
#             trk.cate = cates[i]
#             self.trackers.append(trk)
#         i = len(self.trackers)

#         for trk in reversed(self.trackers):
#             if trk.last_observation.sum() > 0:
#                 d = trk.last_observation[:4]
#             else:
#                 d = trk.get_state()[0]
#             if (trk.time_since_update < 1):
#                 if (self.frame_count <= self.min_hits) or (trk.hit_streak >= self.min_hits):
#                     # id+1 as MOT benchmark requires positive
#                     ret.append(np.concatenate((d, [trk.id+1], [getattr(trk, 'cate', 0)], [0])).reshape(1,-1)) 
#                 if trk.hit_streak == self.min_hits:
#                     # Head Padding (HP)
#                     for prev_i in range(self.min_hits - 1):
#                         prev_observation = trk.history_observations[-(prev_i+2)]
#                         ret.append((np.concatenate((prev_observation[:4], [trk.id+1], [getattr(trk, 'cate', 0)], 
#                             [-(prev_i+1)]))).reshape(1,-1))
#             i -= 1 
#             if (trk.time_since_update > self.max_age):
#                 self.trackers.pop(i)
        
#         if(len(ret)>0):
#             return np.concatenate(ret)
#         return np.empty((0, 7))


# """
#     This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
#     Modified: Replace Kalman filter with Diffusion (D2MP) for motion prediction.
# """
# from __future__ import print_function

# import os
# import numpy as np
# import torch

# from .association import *
# from .diffusion import VarianceSchedule, HMINet, D2MP_OB

# # 全局推理模式，避免误触梯度
# torch.set_grad_enabled(False)

# # Ampere 及以上建议开启 TF32（提升矩阵乘性能，几乎不影响精度）
# if torch.cuda.is_available():
#     try:
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True
#         # 2.0+ 可设 matmul 精度
#         if hasattr(torch, "set_float32_matmul_precision"):
#             torch.set_float32_matmul_precision('high')
#     except Exception:
#         pass

# def _to_int(x):
#     # 将 1D 或 0D 的 torch.Tensor 转为 Python int；否则直接 int(x)
#     if isinstance(x, torch.Tensor):
#         return int(x.reshape(-1)[0].item())
#     return int(x)


# # -----------------------------
# # Helpers kept from original OC-SORT
# # -----------------------------
# def k_previous_obs(observations, cur_age, k):
#     if len(observations) == 0:
#         return np.array([-1, -1, -1, -1, -1], dtype=np.float32)
#     for i in range(k):
#         dt = k - i
#         if cur_age - dt in observations:
#             return observations[cur_age - dt]
#     max_age = max(observations.keys())
#     return observations[max_age]


# def convert_bbox_to_z(bbox):
#     w = bbox[2] - bbox[0]
#     h = bbox[3] - bbox[1]
#     x = bbox[0] + w/2.
#     y = bbox[1] + h/2.
#     s = w * h
#     r = w / float(h + 1e-6)
#     return np.array([x, y, s, r]).reshape((4, 1))


# def convert_x_to_bbox(x, score=None):
#     w = np.sqrt(x[2] * x[3])
#     h = x[2] / w
#     if(score is None):
#         return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
#     else:
#         return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


# def speed_direction(bbox1, bbox2):
#     cx1, cy1 = (bbox1[0]+bbox1[2]) / 2.0, (bbox1[1]+bbox1[3])/2.0
#     cx2, cy2 = (bbox2[0]+bbox2[2]) / 2.0, (bbox2[1]+bbox2[3])/2.0
#     speed = np.array([cy2-cy1, cx2-cx1], dtype=np.float32)
#     norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
#     return speed / norm


# """
#     We support multiple ways for association cost calculation, by default
#     we use IoU. GIoU may have better performance in some situations. We note 
#     that we hardly normalize the cost by all methods to (0,1) which may not be 
#     the best practice.
# """
# ASSO_FUNCS = {  "iou": iou_batch,
#                 "giou": giou_batch,
#                 "ciou": ciou_batch,
#                 "diou": diou_batch,
#                 "ct_dist": ct_dist}


# # -----------------------------
# # D2MP config + utilities
# # -----------------------------
# class D2MPConfig:
#     def __init__(self):
#         # diffusion forward eps
#         self.eps = 1e-5
#         # bbox format used during training: 'xyxy' or 'cxcywh'
#         self.bbox_format = 'xyxy'
#         # normalization after unitizing to [0,1]
#         self.bbox_mean = [0.5, 0.5, 0.5, 0.5]
#         self.bbox_std  = [0.5, 0.5, 0.5, 0.5]
#         # will be inferred from weight if possible
#         self.context_dim = None
#         self.hist_len = 5
#         # diffusion schedule defaults (train-time)
#         self.num_steps = 1 #扩散采样步数较多 + 按 tracklet 单个调用
#         self.beta_1 = 1e-4
#         self.beta_T = 5e-2
#         self.var_mode = 'linear'
#         # model structure defaults (train-time)
#         self.point_dim = 4
#         self.tf_layer = 3
#         # deterministic seed for stability (optional)
#         self.seed = 12345  # or None


# d2mp_config = D2MPConfig()


# def xyxy_to_cxcywh(xyxy):
#     x1, y1, x2, y2 = xyxy
#     w = max(x2 - x1, 1e-6)
#     h = max(y2 - y1, 1e-6)
#     cx = x1 + w / 2.0
#     cy = y1 + h / 2.0
#     return np.array([cx, cy, w, h], dtype=np.float32)


# def cxcywh_to_xyxy(cxcywh):
#     cx, cy, w, h = cxcywh
#     x1 = cx - w / 2.0
#     y1 = cy - h / 2.0
#     x2 = cx + w / 2.0
#     y2 = cy + h / 2.0
#     return np.array([x1, y1, x2, y2], dtype=np.float32)

# def to_unit_xyxy(xyxy, img_size):
#     H, W = img_size
#     H, W = _to_int(H), _to_int(W)
#     x1, y1, x2, y2 = xyxy
#     return np.array([x1 / W, y1 / H, x2 / W, y2 / H], dtype=np.float32)

# def to_pixel_xyxy(unit_xyxy, img_size):
#     H, W = img_size
#     H, W = _to_int(H), _to_int(W)
#     x1, y1, x2, y2 = unit_xyxy
#     return np.array([x1 * W, y1 * H, x2 * W, y2 * H], dtype=np.float32)

# def clamp_xyxy(xyxy, img_size):
#     H, W = img_size
#     H, W = _to_int(H), _to_int(W)
#     x1, y1, x2, y2 = xyxy
#     x1 = max(0, min(W - 1, x1))
#     y1 = max(0, min(H - 1, y1))
#     x2 = max(0, min(W - 1, x2))
#     y2 = max(0, min(H - 1, y2))
#     if x2 < x1: x2 = x1
#     if y2 < y1: y2 = y1
#     return np.array([x1, y1, x2, y2], dtype=np.float32)

# # -----------------------------
# # Global D2MP model init (lazy)
# # -----------------------------
# _global_d2mp = None


# def _infer_context_dim_and_hist_len_from_state(state):
#     # Try to infer context_dim from concat1._hyper_bias.weight in HMINet
#     # expected key like 'net.concat1._hyper_bias.weight' (or without 'net.')
#     ctx_dim = None
#     for k, v in state.items():
#         if k.endswith('concat1._hyper_bias.weight') and v.ndim == 2:
#             ctx_dim = int(v.shape[1]) - 3  # minus time_emb (3 dims)
#             break
#     if ctx_dim is not None and ctx_dim > 0:
#         d2mp_config.context_dim = ctx_dim
#         # hist_len is context_dim // 4 if context is flattened L*4
#         if ctx_dim % 4 == 0:
#             d2mp_config.hist_len = ctx_dim // 4
#         # else leave default and trust user's configuration
#     # else leave defaults


# # def init_d2mp_model():
# #     global _global_d2mp
# #     if _global_d2mp is not None:
# #         return _global_d2mp

# #     # optional: deterministic behavior
# #     if d2mp_config.seed is not None:
# #         torch.manual_seed(d2mp_config.seed)
# #         np.random.seed(d2mp_config.seed)

# #     # build weight path robustly
# #     this_dir = os.path.dirname(os.path.abspath(__file__))
# #     weight_path = "/public/home/cookie/anaconda3/envs/ocsort_env/OC_SORT/trackers/diffmot_tracker/weights/MOT_epoch800.pt"

# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     state = torch.load(weight_path, map_location=device)

# #     # unwrap common containers
# #     if isinstance(state, dict):
# #         # try common wrappers
# #         if 'state_dict' in state and isinstance(state['state_dict'], dict):
# #             state = state['state_dict']
# #         elif 'model' in state and isinstance(state['model'], dict):
# #             state = state['model']

# #     # strip 'module.' prefix if any
# #     needs_strip = any(k.startswith('module.') for k in state.keys())
# #     if needs_strip:
# #         state = {k.replace('module.', '', 1): v for k, v in state.items()}

# #     # infer context_dim/hist_len if possible
# #     _infer_context_dim_and_hist_len_from_state(state)

# #     # fallback if not inferred
# #     context_dim = d2mp_config.context_dim if d2mp_config.context_dim is not None else (d2mp_config.hist_len * 4)

# #     var_sched = VarianceSchedule(
# #         num_steps=d2mp_config.num_steps,
# #         mode=d2mp_config.var_mode,
# #         beta_1=d2mp_config.beta_1,
# #         beta_T=d2mp_config.beta_T
# #     )
# #     hminet = HMINet(
# #         point_dim=d2mp_config.point_dim,
# #         context_dim=context_dim,
# #         tf_layer=d2mp_config.tf_layer
# #     )
# #     model = D2MP_OB(
# #         net=hminet,
# #         var_sched=var_sched,
# #         config=d2mp_config
# #     )

# #     # try strict load, fallback
# #     try:
# #         model.load_state_dict(state, strict=True)
# #     except Exception as e:
# #         print(f"[D2MP] strict load_state_dict failed: {e}\nTrying non-strict loading...")
# #         model.load_state_dict(state, strict=False)

# #     model.eval().to(device)
# #     _global_d2mp = model
# #     return _global_d2mp

# def init_d2mp_model():
#     """
#     Lazy init global D2MP model with robust checkpoint parsing.
#     - Uses absolute weight path provided by you.
#     - Unwraps common wrappers: state_dict / model / ddpm
#     - Strips 'module.' prefixes if present
#     - Extracts 'net' and optional 'var_sched' weights
#     - Infers context_dim / hist_len from weights if possible
#     """
#     global _global_d2mp
#     if _global_d2mp is not None:
#         return _global_d2mp

#     # Optional determinism
#     if d2mp_config.seed is not None:
#         torch.manual_seed(d2mp_config.seed)
#         np.random.seed(d2mp_config.seed)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Use your absolute path
#     weight_path = "/public/home/cookie/anaconda3/envs/ocsort_env/OC_SORT/trackers/diffmot_tracker/weights/MOT_epoch800.pt"

#     # 1) Load raw checkpoint (prefer weights_only=True; fallback for older PyTorch)
#     try:
#         state_raw = torch.load(weight_path, map_location=device, weights_only=True)
#     except TypeError:
#         state_raw = torch.load(weight_path, map_location=device)

#     # 2) Unwrap common containers
#     if isinstance(state_raw, dict):
#         if 'state_dict' in state_raw and isinstance(state_raw['state_dict'], dict):
#             state_raw = state_raw['state_dict']
#         elif 'model' in state_raw and isinstance(state_raw['model'], dict):
#             state_raw = state_raw['model']
#         elif 'ddpm' in state_raw and isinstance(state_raw['ddpm'], dict):
#             inner = state_raw['ddpm']
#             if isinstance(inner, dict) and 'state_dict' in inner and isinstance(inner['state_dict'], dict):
#                 state_raw = inner['state_dict']
#             else:
#                 state_raw = inner

#     # 3) Strip 'module.' prefix if any (DDP saved checkpoints)
#     if isinstance(state_raw, dict) and any(isinstance(k, str) and k.startswith('module.') for k in state_raw.keys()):
#         state_raw = {k.replace('module.', '', 1): v for k, v in state_raw.items()}

#     # 4) Build a flat state dict with expected prefixes: 'net.' and optional 'var_sched.'
#     state = {}
#     if isinstance(state_raw, dict):
#         # Case A: nested under 'net'
#         if 'net' in state_raw and isinstance(state_raw['net'], dict):
#             for k, v in state_raw['net'].items():
#                 key = f'net.{k}' if not k.startswith('net.') else k
#                 state[key] = v
#         # Case B: already flattened with 'net.' prefix
#         elif any(isinstance(k, str) and k.startswith('net.') for k in state_raw.keys()):
#             for k, v in state_raw.items():
#                 if k.startswith('net.'):
#                     state[k] = v
#         else:
#             # Case C: raw layer names without 'net.' (heuristic)
#             for k, v in state_raw.items():
#                 if any(k.startswith(p) for p in ['optimizer', 'epoch', 'ema', 'amp', 'scaler', 'global_step']):
#                     continue
#                 if k.startswith('var_sched.'):
#                     state[k] = v
#                 else:
#                     state[f'net.{k}'] = v

#         # Optional: var_sched (nested or flattened)
#         if 'var_sched' in state_raw and isinstance(state_raw['var_sched'], dict):
#             for k, v in state_raw['var_sched'].items():
#                 key = f'var_sched.{k}' if not k.startswith('var_sched.') else k
#                 state[key] = v
#         elif any(isinstance(k, str) and k.startswith('var_sched.') for k in state_raw.keys()):
#             for k, v in state_raw.items():
#                 if k.startswith('var_sched.'):
#                     state[k] = v

#     # 5) Logging (optional but helpful)
#     num_net = sum(1 for k in state.keys() if isinstance(k, str) and k.startswith('net.'))
#     num_sched = sum(1 for k in state.keys() if isinstance(k, str) and k.startswith('var_sched.'))
#     if num_net == 0:
#         print('[D2MP] WARNING: Could not find net weights in checkpoint. '
#               'Tried keys: state_dict/model/ddpm with subkeys net/var_sched. '
#               'Please inspect the checkpoint structure.')
#     else:
#         print(f'[D2MP] Parsed weights: net keys = {num_net}, var_sched keys = {num_sched}')

#     # 6) Infer context_dim / hist_len from extracted weights if possible
#     _infer_context_dim_and_hist_len_from_state(state)
#     context_dim = d2mp_config.context_dim if d2mp_config.context_dim is not None else (d2mp_config.hist_len * 4)

#     # 7) Build model
#     var_sched = VarianceSchedule(
#         num_steps=d2mp_config.num_steps,
#         mode=d2mp_config.var_mode,
#         beta_1=d2mp_config.beta_1,
#         beta_T=d2mp_config.beta_T
#     )
#     hminet = HMINet(
#         point_dim=d2mp_config.point_dim,
#         context_dim=context_dim,
#         tf_layer=d2mp_config.tf_layer
#     )
#     model = D2MP_OB(
#         net=hminet,
#         var_sched=var_sched,
#         config=d2mp_config
#     )

#     # 8) Load weights (non-strict to allow missing buffers like var_sched.*)
#     try:
#         model.load_state_dict(state, strict=False)
#     except Exception as e:
#         print(f"[D2MP] load_state_dict failed: {e}")

#     model.eval().to(device)
#     _global_d2mp = model
#     return _global_d2mp

# # -----------------------------
# # Diffusion-based tracker
# # -----------------------------
# class DiffusionBoxTracker(object):
#     """
#     D2MP-based tracker replacing Kalman filter.
#     Maintains same external interface for OC-SORT association.
#     """
#     count = 0

#     def __init__(self, bbox, img_size, delta_t=3):
#         """
#         bbox: [x1, y1, x2, y2, score] in pixel coords
#         img_size: (H, W)
#         """
#         self.img_size = img_size  # (H, W)
#         # internal state
#         self.history_bboxes = [bbox[:4].astype(np.float32)]
#         self.last_bbox = bbox[:4].astype(np.float32)  # last predicted/observed bbox (pixel)
#         # OC-SORT required fields
#         self.time_since_update = 0
#         self.id = DiffusionBoxTracker.count
#         DiffusionBoxTracker.count += 1
#         self.hits = 0
#         self.hit_streak = 0
#         self.age = 0

#         # placeholders and history for OC-SORT
#         self.last_observation = np.array([-1, -1, -1, -1, -1], dtype=np.float32)  # same shape as OC-SORT
#         self.observations = dict()
#         self.history_observations = []
#         self.velocity = None
#         self.delta_t = delta_t
#         # immediately update with the init bbox as an observation
#         self.update(bbox, img_size=img_size)

#     # def set_img_size(self, img_size):
#     #     self.img_size = img_size
#     def set_img_size(self, img_size):
#         H, W = img_size
#         H, W = _to_int(H), _to_int(W)
#         self.img_size = (H, W)

#     # normalization helpers
#     def _normalize_tensor(self, bbox_tensor):
#         mean = torch.tensor(d2mp_config.bbox_mean, device=bbox_tensor.device, dtype=bbox_tensor.dtype)
#         std  = torch.tensor(d2mp_config.bbox_std,  device=bbox_tensor.device, dtype=bbox_tensor.dtype)
#         return (bbox_tensor - mean) / std

#     def _denormalize_tensor(self, norm_tensor):
#         mean = torch.tensor(d2mp_config.bbox_mean, device=norm_tensor.device, dtype=norm_tensor.dtype)
#         std  = torch.tensor(d2mp_config.bbox_std,  device=norm_tensor.device, dtype=norm_tensor.dtype)
#         return norm_tensor * std + mean

#     def update(self, bbox, img_size=None):
#         """
#         Update with detection bbox (x1,y1,x2,y2,score) or None.
#         """
#         if img_size is not None:
#             self.set_img_size(img_size)

#         if bbox is not None:
#             bbox = bbox.astype(np.float32)
#             # estimate velocity using observation delta_t steps away
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

#             # maintain obs history like original OC-SORT
#             self.last_observation = bbox.copy()
#             self.observations[self.age] = bbox.copy()
#             self.history_observations.append(bbox.copy())

#             # maintain bbox history for D2MP (store only xyxy)
#             self.last_bbox = bbox[:4].copy()
#             self.history_bboxes.append(self.last_bbox.copy())
#             # keep last hist_len frames
#             if len(self.history_bboxes) > d2mp_config.hist_len:
#                 self.history_bboxes = self.history_bboxes[-d2mp_config.hist_len:]

#             # OC-SORT counters
#             self.time_since_update = 0
#             self.hits += 1
#             self.hit_streak += 1
#         else:
#             # no observation this frame
#             self.time_since_update += 1

#     def predict(self):
#         """
#         Predict next bbox with D2MP. Does NOT overwrite last_observation.
#         Returns shape (1,4) in pixel coords.
#         """
#         # model and device
#         model = init_d2mp_model()
#         device = next(model.parameters()).device

#         # history prep
#         H, W = self.img_size
#         hist = self.history_bboxes[-d2mp_config.hist_len:]
#         if len(hist) < d2mp_config.hist_len:
#             hist = hist + [hist[-1]] * (d2mp_config.hist_len - len(hist))

#         # pixel -> unit
#         hist_unit = [to_unit_xyxy(b, (H, W)) for b in hist]
#         hist_unit = np.stack(hist_unit, axis=0)  # (L,4)

#         # format conversion if trained on cxcywh
#         if d2mp_config.bbox_format == 'cxcywh':
#             hist_unit = np.stack([xyxy_to_cxcywh(b) for b in hist_unit], axis=0)

#         # normalize
#         inp = torch.tensor(hist_unit, dtype=torch.float32, device=device)  # (L,4)
#         inp = self._normalize_tensor(inp)  # normalize per coordinate
#         context = inp.flatten().unsqueeze(0)  # (1, L*4)

#         # sampling
#         # with torch.no_grad():
        
#         with torch.inference_mode():
#             pred = model.sample(
#                 context=context,
#                 sample=1,
#                 bestof=False,
#                 point_dim=d2mp_config.point_dim,
#                 ret_traj=False
#             )  # expected shape: (S=1, B=1, 4)

#         if isinstance(pred, (list, tuple)):
#             pred = pred[0]
#         pred = pred[0, 0] if pred.ndim == 3 else pred.squeeze(0)  # (4,)
#         pred = self._denormalize_tensor(pred)  # back to unit-scale bbox (format per config)

#         # format back to xyxy if needed (still unit scale)
#         pred_unit = pred.detach().cpu().numpy()
#         if d2mp_config.bbox_format == 'cxcywh':
#             pred_unit = cxcywh_to_xyxy(pred_unit)

#         # unit -> pixel, clamp
#         pred_px = to_pixel_xyxy(pred_unit, (H, W))
#         pred_px = clamp_xyxy(pred_px, (H, W))

#         # advance OC-SORT counters (as in KalmanBoxTracker.predict)
#         self.age += 1
#         if self.time_since_update > 0:
#             self.hit_streak = 0
#         self.time_since_update += 1

#         # cache last predicted bbox (separate from last_observation)
#         self.last_bbox = pred_px.astype(np.float32)

#         return self.last_bbox.reshape(1, 4)

#     def get_state(self):
#         """
#         Returns the current bbox estimate (prediction), shape (1,4)
#         """
#         return self.last_bbox.reshape(1, 4)


# # -----------------------------
# # OCSort using DiffusionBoxTracker
# # -----------------------------
# class Diff_OCSort(object):
#     def __init__(self, det_thresh, max_age=30, min_hits=3, 
#         iou_threshold=0.3, delta_t=3, asso_func="iou", inertia=0.2, use_byte=False):
#         """
#         Sets key parameters for SORT
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

#         # reset diffusion-based tracker id counter
#         DiffusionBoxTracker.count = 0

#         # remember last image size (for update_public fallback)
#         self.last_img_size = None  # (H, W)


#     def _batched_predict(self, trackers, img_size):
#         """
#         对一帧内所有 trackers 批量预测，返回 (N, 4) numpy 数组（像素坐标）。
#         同时同步每个 tracker 的 age/time_since_update/hit_streak/last_bbox。
#         """
#         if len(trackers) == 0:
#             return np.zeros((0, 4), dtype=np.float32)

#         model = init_d2mp_model()
#         device = next(model.parameters()).device
#         H, W = img_size
#         # 构建 batch context
#         L = d2mp_config.hist_len
#         contexts = []
#         meta = []  # 保存每个 tracker 的信息以便回填
#         for trk in trackers:
#             trk.set_img_size((H, W))
#             hist = trk.history_bboxes[-L:]
#             if len(hist) < L:
#                 hist = hist + [hist[-1]] * (L - len(hist))
#             # pixel -> unit
#             hist_unit = [to_unit_xyxy(b, (H, W)) for b in hist]
#             hist_unit = np.stack(hist_unit, axis=0)  # (L,4)
#             # 格式转换
#             if d2mp_config.bbox_format == 'cxcywh':
#                 hist_unit = np.stack([xyxy_to_cxcywh(b) for b in hist_unit], axis=0)

#             contexts.append(hist_unit)
#             meta.append(trk)

#         # 组装成 tensor
#         contexts = torch.tensor(np.stack(contexts, axis=0), dtype=torch.float32, device=device)  # (B,L,4)
#         # 归一化
#         mean = torch.tensor(d2mp_config.bbox_mean, device=device, dtype=contexts.dtype).view(1,1,4)
#         std  = torch.tensor(d2mp_config.bbox_std,  device=device, dtype=contexts.dtype).view(1,1,4)
#         contexts = (contexts - mean) / std  # (B,L,4)
#         contexts = contexts.flatten(start_dim=1)  # (B, L*4)

#         # 混合精度（可选）
#         if getattr(d2mp_config, "use_fp16", False) and device.type == "cuda":
#             contexts = contexts.half()
#             model = model.half()

#         # 采样
#         with torch.inference_mode():
#             pred = model.sample(
#                 context=contexts,      # (B, L*4)
#                 sample=1,
#                 bestof=False,
#                 point_dim=d2mp_config.point_dim,
#                 ret_traj=False
#             )  # 期望形状: (1, B, 4) 或 (B,4)

#         if isinstance(pred, (list, tuple)):
#             pred = pred[0]
#         if pred.ndim == 3:
#             pred = pred[0]  # (B,4)
#         # 反归一化（单位坐标）
#         if pred.dtype != contexts.dtype:
#             pred = pred.to(contexts.dtype)
#         pred = pred.view(len(trackers), 4)
#         pred = pred * std.view(1,4) + mean.view(1,4)

#         # 单位 -> 像素，必要时格式还原
#         pred_np = pred.detach().float().cpu().numpy()  # (B,4), 单位坐标
#         if d2mp_config.bbox_format == 'cxcywh':
#             pred_np = np.stack([cxcywh_to_xyxy(p) for p in pred_np], axis=0)
#         preds_px = np.stack([to_pixel_xyxy(p, (H, W)) for p in pred_np], axis=0)
#         preds_px = np.stack([clamp_xyxy(p, (H, W)) for p in preds_px], axis=0).astype(np.float32)

#         # 同步 tracker 的内部计数器与 last_bbox（对齐单个 predict 的行为）
#         for trk, px in zip(meta, preds_px):
#             trk.age += 1
#             if trk.time_since_update > 0:
#                 trk.hit_streak = 0
#             trk.time_since_update += 1
#             trk.last_bbox = px.astype(np.float32)

#         return preds_px


#     def update(self, output_results, img_info, img_size):
#         """
#         Params:
#           output_results: numpy array of detections [[x1,y1,x2,y2,score], ...]
#           img_info: (img_h, img_w, ...)
#           img_size: (test_input_h, test_input_w) - for scaling back
#         """
#         if output_results is None:
#             return np.empty((0, 5))

#         self.frame_count += 1
#         # post_process detections
#         if output_results.shape[1] == 5:
#             scores = output_results[:, 4]
#             bboxes = output_results[:, :4]
#         else:
#             output_results = output_results.cpu().numpy()
#             scores = output_results[:, 4] * output_results[:, 5]
#             bboxes = output_results[:, :4]  # x1y1x2y2
#         # img_h, img_w = img_info[0], img_info[1]
#         # self.last_img_size = (img_h, img_w)  # remember for public update
#         # scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
#         img_h, img_w = _to_int(img_info[0]), _to_int(img_info[1])
#         self.last_img_size = (img_h, img_w)
#         scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        
#         bboxes = bboxes / scale
#         dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)

#         inds_low = scores > 0.1
#         inds_high = scores < self.det_thresh
#         inds_second = np.logical_and(inds_low, inds_high)
#         dets_second = dets[inds_second]
#         remain_inds = scores > self.det_thresh
#         dets = dets[remain_inds]

#         # predict from existing trackers
#         # trks = np.zeros((len(self.trackers), 5), dtype=np.float32)
#         # to_del = []
#         # ret = []
#         # for t, trk in enumerate(self.trackers):
#         #     pos = trk.predict()[0]  # (4,)
#         #     trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
#         #     if np.any(np.isnan(pos)):
#         #         to_del.append(t)
        
#         trks = np.zeros((len(self.trackers), 5), dtype=np.float32)
#         to_del = []
#         ret = []
#         if len(self.trackers) > 0:
#             preds_px = self._batched_predict(self.trackers, img_size=(img_h, img_w))  # (N,4)
#             for t in range(len(self.trackers)):
#                 pos = preds_px[t]
#                 trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
#                 if np.any(np.isnan(pos)):
#                     to_del.append(t)
        
#         trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#         for t in reversed(to_del):
#             self.trackers.pop(t)

#         velocities = np.array(
#             [trk.velocity if trk.velocity is not None else np.array((0, 0), dtype=np.float32) for trk in self.trackers])
#         last_boxes = np.array([trk.last_observation for trk in self.trackers])
#         k_observations = np.array(
#             [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

#         # First round of association
#         matched, unmatched_dets, unmatched_trks = associate(
#             dets, trks, self.iou_threshold, velocities, k_observations, self.inertia)
#         for m in matched:
#             # pass img_size for trackers to keep resolution
#             self.trackers[m[1]].update(dets[m[0], :], img_size=(img_h, img_w))

#         # Second round of association by OCR (BYTE optional)
#         if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
#             u_trks = trks[unmatched_trks]
#             iou_left = self.asso_func(dets_second, u_trks)
#             iou_left = np.array(iou_left)
#             if iou_left.max() > self.iou_threshold:
#                 matched_indices = linear_assignment(-iou_left)
#                 to_remove_trk_indices = []
#                 for m in matched_indices:
#                     det_ind, trk_ind = m[0], unmatched_trks[m[1]]
#                     if iou_left[m[0], m[1]] < self.iou_threshold:
#                         continue
#                     self.trackers[trk_ind].update(dets_second[det_ind, :], img_size=(img_h, img_w))
#                     to_remove_trk_indices.append(trk_ind)
#                 unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

#         if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
#             left_dets = dets[unmatched_dets]
#             left_trks = last_boxes[unmatched_trks]
#             iou_left = self.asso_func(left_dets, left_trks)
#             iou_left = np.array(iou_left)
#             if iou_left.max() > self.iou_threshold:
#                 rematched_indices = linear_assignment(-iou_left)
#                 to_remove_det_indices = []
#                 to_remove_trk_indices = []
#                 for m in rematched_indices:
#                     det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
#                     if iou_left[m[0], m[1]] < self.iou_threshold:
#                         continue
#                     self.trackers[trk_ind].update(dets[det_ind, :], img_size=(img_h, img_w))
#                     to_remove_det_indices.append(det_ind)
#                     to_remove_trk_indices.append(trk_ind)
#                 unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
#                 unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

#         for m in unmatched_trks:
#             self.trackers[m].update(None)

#         # create and initialise new trackers
#         for i in unmatched_dets:
#             trk = DiffusionBoxTracker(dets[i, :], img_size=(img_h, img_w), delta_t=self.delta_t)
#             self.trackers.append(trk)

#         i = len(self.trackers)
#         for trk in reversed(self.trackers):
#             if trk.last_observation.sum() < 0:
#                 d = trk.get_state()[0]
#             else:
#                 # optional: use recent observation or predicted
#                 d = trk.last_observation[:4]
#             if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
#                 # +1 as MOT benchmark requires positive
#                 ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
#             i -= 1
#             # remove dead tracklet
#             if (trk.time_since_update > self.max_age):
#                 self.trackers.pop(i)
#         if (len(ret) > 0):
#             return np.concatenate(ret)
#         return np.empty((0, 5))

#     def update_public(self, dets, cates, scores):
#         """
#         Public detections variant.
#         Note: will use last known image size from update(); if None, falls back to (1080, 1920).
#         """
#         self.frame_count += 1

#         det_scores = np.ones((dets.shape[0], 1), dtype=np.float32)
#         dets = np.concatenate((dets, det_scores), axis=1)

#         remain_inds = scores > self.det_thresh
#         cates = cates[remain_inds]
#         dets = dets[remain_inds]

#         trks = np.zeros((len(self.trackers), 5), dtype=np.float32)
#         to_del = []
#         ret = []
#         for t, trk in enumerate(self.trackers):
#             pos = trk.predict()[0]
#             cat = getattr(trk, 'cate', 0)
#             trks[t, :] = [pos[0], pos[1], pos[2], pos[3], cat]
#             if np.any(np.isnan(pos)):
#                 to_del.append(t)
#         trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#         for t in reversed(to_del):
#             self.trackers.pop(t)

#         velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0,0), dtype=np.float32) for trk in self.trackers])
#         last_boxes = np.array([trk.last_observation for trk in self.trackers])
#         k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

#         matched, unmatched_dets, unmatched_trks = associate_kitti\
#               (dets, trks, cates, self.iou_threshold, velocities, k_observations, self.inertia)
          
#         for m in matched:
#             # use last known image size or a sensible default
#             img_size = self.last_img_size if self.last_img_size is not None else (1080, 1920)
#             self.trackers[m[1]].update(dets[m[0], :], img_size=img_size)
          
#         if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
#             left_dets = dets[unmatched_dets]
#             left_trks = last_boxes[unmatched_trks]
#             left_dets_c = left_dets.copy()
#             left_trks_c = left_trks.copy()

#             iou_left = self.asso_func(left_dets_c, left_trks_c)
#             iou_left = np.array(iou_left)
#             det_cates_left = cates[unmatched_dets]
#             trk_cates_left = trks[unmatched_trks][:,4]
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
#                     img_size = self.last_img_size if self.last_img_size is not None else (1080, 1920)
#                     self.trackers[trk_ind].update(dets[det_ind, :], img_size=img_size)
#                     to_remove_det_indices.append(det_ind)
#                     to_remove_trk_indices.append(trk_ind) 
#                 unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
#                 unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

#         for i in unmatched_dets:
#             img_size = self.last_img_size if self.last_img_size is not None else (1080, 1920)
#             trk = DiffusionBoxTracker(dets[i,:], img_size=img_size, delta_t=self.delta_t)
#             trk.cate = cates[i]
#             self.trackers.append(trk)
#         i = len(self.trackers)

#         for trk in reversed(self.trackers):
#             if trk.last_observation.sum() > 0:
#                 d = trk.last_observation[:4]
#             else:
#                 d = trk.get_state()[0]
#             if (trk.time_since_update < 1):
#                 if (self.frame_count <= self.min_hits) or (trk.hit_streak >= self.min_hits):
#                     # id+1 as MOT benchmark requires positive
#                     ret.append(np.concatenate((d, [trk.id+1], [getattr(trk, 'cate', 0)], [0])).reshape(1,-1)) 
#                 if trk.hit_streak == self.min_hits:
#                     # Head Padding (HP)
#                     for prev_i in range(self.min_hits - 1):
#                         prev_observation = trk.history_observations[-(prev_i+2)]
#                         ret.append((np.concatenate((prev_observation[:4], [trk.id+1], [getattr(trk, 'cate', 0)], 
#                             [-(prev_i+1)]))).reshape(1,-1))
#             i -= 1 
#             if (trk.time_since_update > self.max_age):
#                 self.trackers.pop(i)
        
#         if(len(ret)>0):
#             return np.concatenate(ret)
#         return np.empty((0, 7))



# """
#     This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
#     Modified: Replace Kalman filter with Diffusion (D2MP) for motion prediction.
#     Diffmot-compat: Prefer model.generate(conds, ...) if available,
#     else fallback to model.sample(context, ...).
# """
# from __future__ import print_function

# import os
# import collections
# import numpy as np
# import torch

# from .association import *
# from .diffusion import VarianceSchedule, HMINet, D2MP_OB

# # 全局推理模式，避免误触梯度
# torch.set_grad_enabled(False)

# # Ampere 及以上建议开启 TF32（提升矩阵乘性能，几乎不影响精度）
# if torch.cuda.is_available():
#     try:
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True
#         if hasattr(torch, "set_float32_matmul_precision"):
#             torch.set_float32_matmul_precision('high')
#     except Exception:
#         pass


# def _to_int(x):
#     if isinstance(x, torch.Tensor):
#         return int(x.reshape(-1)[0].item())
#     return int(x)


# # -----------------------------
# # Helpers kept from original OC-SORT
# # -----------------------------
# def k_previous_obs(observations, cur_age, k):
#     if len(observations) == 0:
#         return np.array([-1, -1, -1, -1, -1], dtype=np.float32)
#     for i in range(k):
#         dt = k - i
#         if cur_age - dt in observations:
#             return observations[cur_age - dt]
#     max_age = max(observations.keys())
#     return observations[max_age]


# def convert_bbox_to_z(bbox):
#     w = bbox[2] - bbox[0]
#     h = bbox[3] - bbox[1]
#     x = bbox[0] + w / 2.0
#     y = bbox[1] + h / 2.0
#     s = w * h
#     r = w / float(h + 1e-6)
#     return np.array([x, y, s, r]).reshape((4, 1))


# def convert_x_to_bbox(x, score=None):
#     w = np.sqrt(x[2] * x[3])
#     h = x[2] / w
#     if score is None:
#         return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
#     else:
#         return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


# def speed_direction(bbox1, bbox2):
#     cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
#     cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
#     speed = np.array([cy2 - cy1, cx2 - cx1], dtype=np.float32)
#     norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
#     return speed / norm


# ASSO_FUNCS = {
#     "iou": iou_batch,
#     "giou": giou_batch,
#     "ciou": ciou_batch,
#     "diou": diou_batch,
#     "ct_dist": ct_dist,
# }


# # -----------------------------
# # D2MP config + utilities
# # -----------------------------
# class D2MPConfig:
#     def __init__(self):
#         # 数值稳定项（D2MP_OB 需要）
#         self.eps = 1e-5

#         # bbox 表达，Diffmot 训练默认 cxcywh
#         self.bbox_format = 'cxcywh'

#         # 对 generate 分支无额外归一化，sample 分支仍可用
#         # self.bbox_mean = [0.0, 0.0, 0.0, 0.0]
#         # self.bbox_std  = [1.0, 1.0, 1.0, 1.0]
        
#         self.bbox_mean = [0.5, 0.5, 0.5, 0.5]
#         self.bbox_std  = [0.5, 0.5, 0.5, 0.5]

#         self.context_dim = None
#         self.hist_len = 5

#         # 扩散调度与步数（尽量对齐训练）
#         self.num_steps = 1
#         self.beta_1 = 1e-4
#         self.beta_T = 5e-2
#         self.var_mode = 'linear'  # 若你确认支持 'cosine' 再改

#         # 模型结构
#         self.point_dim = 4
#         self.tf_layer  = 3

#         # 其他开关
#         self.seed = 12345
#         self.use_fp16 = False

#         # 稳健性
#         self.pred_blend = 0.5
#         self.pred_gate_iou = 0.05
#         self.use_const_v_fallback = True


# d2mp_config = D2MPConfig()


# def xyxy_to_cxcywh(xyxy):
#     x1, y1, x2, y2 = xyxy
#     w = max(x2 - x1, 1e-6)
#     h = max(y2 - y1, 1e-6)
#     cx = x1 + w / 2.0
#     cy = y1 + h / 2.0
#     return np.array([cx, cy, w, h], dtype=np.float32)


# def cxcywh_to_xyxy(cxcywh):
#     cx, cy, w, h = cxcywh
#     x1 = cx - w / 2.0
#     y1 = cy - h / 2.0
#     x2 = cx + w / 2.0
#     y2 = cy + h / 2.0
#     return np.array([x1, y1, x2, y2], dtype=np.float32)


# def to_unit_xyxy(xyxy, img_size):
#     H, W = img_size
#     H, W = _to_int(H), _to_int(W)
#     x1, y1, x2, y2 = xyxy
#     return np.array([x1 / W, y1 / H, x2 / W, y2 / H], dtype=np.float32)


# def to_pixel_xyxy(unit_xyxy, img_size):
#     H, W = img_size
#     H, W = _to_int(H), _to_int(W)
#     x1, y1, x2, y2 = unit_xyxy
#     return np.array([x1 * W, y1 * H, x2 * W, y2 * H], dtype=np.float32)


# def clamp_xyxy(xyxy, img_size):
#     H, W = img_size
#     H, W = _to_int(H), _to_int(W)
#     x1, y1, x2, y2 = xyxy
#     x1 = max(0, min(W - 1, x1))
#     y1 = max(0, min(H - 1, y1))
#     x2 = max(0, min(W - 1, x2))
#     y2 = max(0, min(H - 1, y2))
#     if x2 < x1:
#         x2 = x1
#     if y2 < y1:
#         y2 = y1
#     return np.array([x1, y1, x2, y2], dtype=np.float32)


# # -----------------------------
# # Global D2MP model init (lazy)
# # -----------------------------
# _global_d2mp = None


# def _infer_context_dim_and_hist_len_from_state(state):
#     # Try to infer context_dim from concat1._hyper_bias.weight in HMINet
#     ctx_dim = None
#     for k, v in state.items():
#         if isinstance(k, str) and k.endswith('concat1._hyper_bias.weight') and hasattr(v, "ndim") and v.ndim == 2:
#             ctx_dim = int(v.shape[1]) - 3  # - time_emb(3)
#             break
#     if ctx_dim is not None and ctx_dim > 0:
#         d2mp_config.context_dim = ctx_dim
#         if ctx_dim % 4 == 0:
#             d2mp_config.hist_len = ctx_dim // 4


# def init_d2mp_model():
#     """
#     Lazy init global D2MP model with robust checkpoint parsing.
#     """
#     global _global_d2mp
#     if _global_d2mp is not None:
#         return _global_d2mp

#     if d2mp_config.seed is not None:
#         torch.manual_seed(d2mp_config.seed)
#         np.random.seed(d2mp_config.seed)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Use your absolute path
#     weight_path = "/public/home/cookie/anaconda3/envs/ocsort_env/OC_SORT/trackers/diffmot_tracker/weights/MOT_epoch800.pt"

#     # 1) Load raw checkpoint
#     try:
#         state_raw = torch.load(weight_path, map_location=device, weights_only=True)
#     except TypeError:
#         state_raw = torch.load(weight_path, map_location=device)

#     # 2) Unwrap common containers
#     if isinstance(state_raw, dict):
#         if 'state_dict' in state_raw and isinstance(state_raw['state_dict'], dict):
#             state_raw = state_raw['state_dict']
#         elif 'model' in state_raw and isinstance(state_raw['model'], dict):
#             state_raw = state_raw['model']
#         elif 'ddpm' in state_raw and isinstance(state_raw['ddpm'], dict):
#             inner = state_raw['ddpm']
#             if isinstance(inner, dict) and 'state_dict' in inner and isinstance(inner['state_dict'], dict):
#                 state_raw = inner['state_dict']
#             else:
#                 state_raw = inner

#     # 3) Strip 'module.' prefix if any
#     if isinstance(state_raw, dict) and any(isinstance(k, str) and k.startswith('module.') for k in state_raw.keys()):
#         state_raw = {k.replace('module.', '', 1): v for k, v in state_raw.items()}

#     # 4) Build flat state dict
#     state = {}
#     if isinstance(state_raw, dict):
#         if 'net' in state_raw and isinstance(state_raw['net'], dict):
#             for k, v in state_raw['net'].items():
#                 key = f'net.{k}' if not k.startswith('net.') else k
#                 state[key] = v
#         elif any(isinstance(k, str) and k.startswith('net.') for k in state_raw.keys()):
#             for k, v in state_raw.items():
#                 if k.startswith('net.'):
#                     state[k] = v
#         else:
#             for k, v in state_raw.items():
#                 if any(k.startswith(p) for p in ['optimizer', 'epoch', 'ema', 'amp', 'scaler', 'global_step']):
#                     continue
#                 if k.startswith('var_sched.'):
#                     state[k] = v
#                 else:
#                     state[f'net.{k}'] = v

#         if 'var_sched' in state_raw and isinstance(state_raw['var_sched'], dict):
#             for k, v in state_raw['var_sched'].items():
#                 key = f'var_sched.{k}' if not k.startswith('var_sched.') else k
#                 state[key] = v
#         elif any(isinstance(k, str) and k.startswith('var_sched.') for k in state_raw.keys()):
#             for k, v in state_raw.items():
#                 if k.startswith('var_sched.'):
#                     state[k] = v

#     # 5) Logging
#     num_net = sum(1 for k in state.keys() if isinstance(k, str) and k.startswith('net.'))
#     num_sched = sum(1 for k in state.keys() if isinstance(k, str) and k.startswith('var_sched.'))
#     if num_net == 0:
#         print('[D2MP] WARNING: Could not find net weights in checkpoint.')
#     else:
#         print(f'[D2MP] Parsed weights: net keys = {num_net}, var_sched keys = {num_sched}')

#     # 6) Infer context_dim / hist_len
#     _infer_context_dim_and_hist_len_from_state(state)
#     context_dim = d2mp_config.context_dim if d2mp_config.context_dim is not None else (d2mp_config.hist_len * 4)

#     # 7) Build model
#     var_sched = VarianceSchedule(
#         num_steps=d2mp_config.num_steps,
#         mode=d2mp_config.var_mode,
#         beta_1=d2mp_config.beta_1,
#         beta_T=d2mp_config.beta_T
#     )
#     hminet = HMINet(
#         point_dim=d2mp_config.point_dim,
#         context_dim=context_dim,
#         tf_layer=d2mp_config.tf_layer
#     )
    
#     if not hasattr(d2mp_config, "eps"):
#         d2mp_config.eps = 1e-5
    
#     model = D2MP_OB(
#         net=hminet,
#         var_sched=var_sched,
#         config=d2mp_config
#     )

#     # 8) Load weights
#     try:
#         model.load_state_dict(state, strict=False)
#     except Exception as e:
#         print(f"[D2MP] load_state_dict failed: {e}")

#     model.eval().to(device)
#     if getattr(d2mp_config, "use_fp16", False) and device.type == "cuda":
#         model.half()

#     print(f"[D2MP] cfg => format={d2mp_config.bbox_format}, hist_len={d2mp_config.hist_len}, "
#           f"num_steps={d2mp_config.num_steps}, var_mode={d2mp_config.var_mode}, "
#           f"use_fp16={d2mp_config.use_fp16}")
#     _global_d2mp = model
#     return _global_d2mp


# # -----------------------------
# # Diffusion-based tracker
# # -----------------------------
# class DiffusionBoxTracker(object):
#     """
#     D2MP-based tracker replacing Kalman filter.
#     Maintains same external interface for OC-SORT association.
#     """
#     count = 0

#     def __init__(self, bbox, img_size, delta_t=3):
#         """
#         bbox: [x1, y1, x2, y2, score] in pixel coords
#         img_size: (H, W)
#         """
#         self.img_size = img_size  # (H, W)
#         # OC-SORT counters
#         self.time_since_update = 0
#         self.id = DiffusionBoxTracker.count
#         DiffusionBoxTracker.count += 1
#         self.hits = 0
#         self.hit_streak = 0
#         self.age = 0

#         # histories
#         self.history_bboxes = [bbox[:4].astype(np.float32)]
#         self.last_bbox = bbox[:4].astype(np.float32)
#         self.last_observation = np.array([-1, -1, -1, -1, -1], dtype=np.float32)
#         self.observations = dict()
#         self.history_observations = []
#         self.velocity = None
#         self.delta_t = delta_t

#         # Diffmot 兼容：维护 conds（像素域）
#         self.prev_accept_xywh = None
#         self.diff_conds = collections.deque([], maxlen=5)

#         # 初始化立即作为一次观测
#         self.update(bbox, img_size=img_size)

#     def set_img_size(self, img_size):
#         H, W = img_size
#         H, W = _to_int(H), _to_int(W)
#         self.img_size = (H, W)

#     # normalization helpers for sample fallback
#     def _normalize_tensor(self, bbox_tensor):
#         mean = torch.tensor(d2mp_config.bbox_mean, device=bbox_tensor.device, dtype=bbox_tensor.dtype)
#         std = torch.tensor(d2mp_config.bbox_std, device=bbox_tensor.device, dtype=bbox_tensor.dtype)
#         return (bbox_tensor - mean) / std

#     def _denormalize_tensor(self, norm_tensor):
#         mean = torch.tensor(d2mp_config.bbox_mean, device=norm_tensor.device, dtype=norm_tensor.dtype)
#         std = torch.tensor(d2mp_config.bbox_std, device=norm_tensor.device, dtype=norm_tensor.dtype)
#         return norm_tensor * std + mean

#     def update(self, bbox, img_size=None):
#         """
#         Update with detection bbox (x1,y1,x2,y2,score) or None.
#         """
#         if img_size is not None:
#             self.set_img_size(img_size)

#         if bbox is not None:
#             bbox = bbox.astype(np.float32)
#             # velocity (direction) using observation delta_t steps away
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

#             # maintain obs history like original OC-SORT
#             self.last_observation = bbox.copy()
#             self.observations[self.age] = bbox.copy()
#             self.history_observations.append(bbox.copy())

#             # maintain bbox history for D2MP (store xyxy)
#             self.last_bbox = bbox[:4].copy()
#             self.history_bboxes.append(self.last_bbox.copy())
#             if len(self.history_bboxes) > d2mp_config.hist_len:
#                 self.history_bboxes = self.history_bboxes[-d2mp_config.hist_len:]

#             # Diffmot conds: [xywh, delta_xywh] in pixels
#             x1, y1, x2, y2 = self.last_bbox
#             cx = 0.5 * (x1 + x2)
#             cy = 0.5 * (y1 + y2)
#             w = max(x2 - x1, 1.0)
#             h = max(y2 - y1, 1.0)
#             xywh = np.array([cx, cy, w, h], dtype=np.float32)
#             if self.prev_accept_xywh is None:
#                 delta = np.zeros(4, dtype=np.float32)
#             else:
#                 delta = (xywh - self.prev_accept_xywh).astype(np.float32)
#             tmp_conds = np.concatenate([xywh, delta]).astype(np.float32)
#             self.diff_conds.append(tmp_conds)
#             self.prev_accept_xywh = xywh.copy()

#             # OC-SORT counters
#             self.time_since_update = 0
#             self.hits += 1
#             self.hit_streak += 1
#         else:
#             # no observation this frame
#             self.time_since_update += 1

#     def predict(self):
#         """
#         Predict next bbox with D2MP (single-track fallback).
#         主流程使用 batched 版本；public 模式等仍会用到该方法。
#         Returns shape (1,4) in pixel coords.
#         """
#         model = init_d2mp_model()
#         device = next(model.parameters()).device

#         # sample 分支（保持原逻辑）
#         H, W = self.img_size
#         L = d2mp_config.hist_len
#         hist = self.history_bboxes[-L:]
#         if len(hist) < L:
#             hist = hist + [hist[-1]] * (L - len(hist))

#         hist_unit = [to_unit_xyxy(b, (H, W)) for b in hist]
#         hist_unit = np.stack(hist_unit, axis=0)  # (L,4)

#         if d2mp_config.bbox_format == 'xyxy':  # 原cxcywh
#             hist_unit = np.stack([xyxy_to_cxcywh(b) for b in hist_unit], axis=0)

#         inp = torch.tensor(hist_unit, dtype=torch.float32, device=device)  # (L,4)
#         inp = self._normalize_tensor(inp)
#         context = inp.flatten().unsqueeze(0)  # (1, L*4)

#         with torch.inference_mode():
#             pred = model.sample(
#                 context=context,
#                 sample=1,
#                 bestof=False,
#                 point_dim=d2mp_config.point_dim,
#                 ret_traj=False
#             )

#         if isinstance(pred, (list, tuple)):
#             pred = pred[0]
#         pred = pred[0, 0] if pred.ndim == 3 else pred.squeeze(0)  # (4,)
#         pred = self._denormalize_tensor(pred)

#         pred_unit = pred.detach().cpu().numpy()
#         if d2mp_config.bbox_format == 'xyxy': #cxcywh
#             pred_unit = cxcywh_to_xyxy(pred_unit)

#         pred_px = to_pixel_xyxy(pred_unit, (H, W))
#         pred_px = clamp_xyxy(pred_px, (H, W))

#         # advance counters
#         self.age += 1
#         if self.time_since_update > 0:
#             self.hit_streak = 0
#         self.time_since_update += 1

#         self.last_bbox = pred_px.astype(np.float32)
#         return self.last_bbox.reshape(1, 4)

#     def get_state(self):
#         return self.last_bbox.reshape(1, 4)


# # -----------------------------
# # OCSort using DiffusionBoxTracker
# # -----------------------------
# class Diff_OCSort(object):
#     def __init__(self, det_thresh, max_age=30, min_hits=3,
#                  iou_threshold=0.3, delta_t=3, asso_func="iou", inertia=0.0, use_byte=False):
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

#         DiffusionBoxTracker.count = 0
#         self.last_img_size = None  # (H,W)

#         # 诊断用
#         self._dbg_pred_cnt = 0
#         self._dbg_iou_running = []

#     def _iou_single(self, a, b):
#         x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
#         x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
#         iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
#         inter = iw * ih
#         area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
#         area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
#         union = area_a + area_b - inter + 1e-6
#         return inter / union

#     def _batched_predict(self, trackers, img_size):
#         """
#         - 如模型具备 generate(): 走 Diffmot 语义（conds 像素域，输出偏移）
#         - 否则：回退到 sample(context)（unit 域绝对框）
#         返回 (N,4) xyxy 像素坐标，并同步计数器和 last_bbox。
#         """
#         if len(trackers) == 0:
#             return np.zeros((0, 4), dtype=np.float32)

#         model = init_d2mp_model()
#         device = next(model.parameters()).device
#         H, W = img_size

#         has_gen = callable(getattr(model, "generate", None))
#         print(f"[D2MP] Predict branch: {'generate' if has_gen else 'sample'} | num_steps={d2mp_config.num_steps}, hist_len={d2mp_config.hist_len}")
        
#         # 优先使用 Diffmot 风格的 generate 接口
#         if hasattr(model, "generate"):
#             dets_norm = []
#             conds_list = []
#             metas = []
#             for trk in trackers:
#                 trk.set_img_size((H, W))
#                 # 基线使用“上次接受的观测”
#                 if getattr(trk, "prev_accept_xywh", None) is not None:
#                     cx, cy, w, h = trk.prev_accept_xywh
#                 else:
#                     x1, y1, x2, y2 = trk.last_observation[:4] if trk.last_observation.sum() > 0 else trk.last_bbox
#                     cx = 0.5 * (x1 + x2)
#                     cy = 0.5 * (y1 + y2)
#                     w = max(1.0, x2 - x1)
#                     h = max(1.0, y2 - y1)
#                 dets_norm.append([cx / W, cy / H, w / W, h / H])

#                 # conds（像素域，含 delta）
#                 if not hasattr(trk, "diff_conds") or len(trk.diff_conds) == 0:
#                     delta = np.zeros(4, dtype=np.float32)
#                     cond = np.array([cx, cy, w, h, *delta], dtype=np.float32)
#                     conds = [cond]
#                 else:
#                     conds = list(trk.diff_conds)
#                 conds_list.append(conds)
#                 metas.append(trk)

#             dets_norm = np.asarray(dets_norm, dtype=np.float32)

#             with torch.inference_mode():
#                 multi_track_pred = model.generate(
#                     conds_list, sample=1, bestof=True, img_w=W, img_h=H
#                 )

#             if isinstance(multi_track_pred, (list, tuple)):
#                 multi_track_pred = multi_track_pred[0]
#             pred = multi_track_pred
#             pred = np.asarray(pred)
#             if pred.ndim == 3:
#                 pred = pred.mean(0)  # (N,4) 归一化域 residual cxcywh

#             track_pred = pred + dets_norm  # 归一化域 cxcywh

#             # 归一化 -> 像素，并转 xyxy
#             track_pred_px = track_pred.copy()
#             track_pred_px[:, 0::2] *= float(W)  # cx,w
#             track_pred_px[:, 1::2] *= float(H)  # cy,h
#             x_tl = track_pred_px[:, 0] - track_pred_px[:, 2] / 2.0
#             y_tl = track_pred_px[:, 1] - track_pred_px[:, 3] / 2.0
#             w_px = track_pred_px[:, 2]; h_px = track_pred_px[:, 3]
#             x1 = x_tl; y1 = y_tl; x2 = x_tl + w_px; y2 = y_tl + h_px
#             preds_px = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
#             preds_px[:, 0] = np.clip(preds_px[:, 0], 0, W - 1)
#             preds_px[:, 1] = np.clip(preds_px[:, 1], 0, H - 1)
#             preds_px[:, 2] = np.clip(preds_px[:, 2], 0, W - 1)
#             preds_px[:, 3] = np.clip(preds_px[:, 3], 0, H - 1)
#         else:
#             # 回退到 sample(context) 路径
#             L = d2mp_config.hist_len
#             contexts = []
#             metas = []
#             for trk in trackers:
#                 trk.set_img_size((H, W))
#                 hist = trk.history_bboxes[-L:]
#                 if len(hist) < L:
#                     hist = hist + [hist[-1]] * (L - len(hist))
#                 hist_unit = [to_unit_xyxy(b, (H, W)) for b in hist]
#                 hist_unit = np.stack(hist_unit, axis=0)  # (L,4)
#                 if d2mp_config.bbox_format == 'xyxy': # cxcywh
#                     hist_unit = np.stack([xyxy_to_cxcywh(b) for b in hist_unit], axis=0)
#                 contexts.append(hist_unit)
#                 metas.append(trk)

#             contexts = torch.tensor(np.stack(contexts, axis=0), dtype=torch.float32, device=device)  # (B,L,4)
#             mean = torch.tensor(d2mp_config.bbox_mean, device=device, dtype=contexts.dtype).view(1, 1, 4)
#             std = torch.tensor(d2mp_config.bbox_std, device=device, dtype=contexts.dtype).view(1, 1, 4)
#             contexts = (contexts - mean) / std
#             contexts = contexts.flatten(start_dim=1)

#             if getattr(d2mp_config, "use_fp16", False) and device.type == "cuda":
#                 contexts = contexts.half()

#             with torch.inference_mode():
#                 pred = model.sample(
#                     context=contexts,
#                     sample=1,
#                     bestof=False,
#                     point_dim=d2mp_config.point_dim,
#                     ret_traj=False
#                 )

#             # [INSERT POINT 1] ---- 形状诊断 + 统一到 (B,4) ----
#             if not hasattr(self, "_pred_shape_log"):
#                 self._pred_shape_log = 0
#             if self._pred_shape_log < 5:
#                 try:
#                     print("[D2MP][sample] raw pred type:", type(pred), 
#                         "shape:", tuple(pred.shape) if hasattr(pred, "shape") else None)
#                 except Exception as e:
#                     print("[D2MP][sample] shape print err:", e)
#                 self._pred_shape_log += 1

#             # 统一 pred 形状为 (B,4)
#             if isinstance(pred, (list, tuple)):
#                 pred = pred[0]
#             if pred.ndim == 3:
#                 S, B_ = pred.shape[:2]
#                 if S == 1 and B_ == len(trackers):
#                     pred = pred[0]        # (B,4)
#                 elif B_ == 1 and pred.shape[0] == len(trackers):
#                     pred = pred[:, 0]     # (B,4)
#                 else:
#                     print(f"[D2MP][warn] unexpected pred shape {tuple(pred.shape)}; expect (1,B,4) or (B,1,4), B={len(trackers)}")
#                     if pred.shape[-1] == 4:
#                         pred = pred.reshape(-1, 4)  # 粗暴兜底
#                     else:
#                         raise RuntimeError(f"Unrecognized pred shape: {tuple(pred.shape)}")

#             if pred.dtype != contexts.dtype:
#                 pred = pred.to(contexts.dtype)
#             pred = pred.view(len(trackers), 4)  # 现在应为 (B,4)
#             # [INSERT POINT 1 END]

#             # 2) 反归一化回“单位域”
#             pred = pred * std.view(1, 4) + mean.view(1, 4)  # ← 单位域

#             pred_np = pred.detach().float().cpu().numpy()

#             # 3) bbox_format 转换
#             if d2mp_config.bbox_format == 'cxcywh':
#                 pred_np = np.stack([cxcywh_to_xyxy(p) for p in pred_np], axis=0)
#             elif d2mp_config.bbox_format == 'xyxy':
#                 pass
#             else:
#                 print(f"[D2MP][warn] unexpected bbox_format: {d2mp_config.bbox_format}")

               
#             preds_px = np.stack([to_pixel_xyxy(p, (H, W)) for p in pred_np], axis=0)
#             preds_px = np.stack([clamp_xyxy(p, (H, W)) for p in preds_px], axis=0).astype(np.float32)

#         # 稳健性：门控 + 融合
#         bl = getattr(d2mp_config, "pred_blend", 0.25)
#         gate = getattr(d2mp_config, "pred_gate_iou", 0.0)
#         use_cv = getattr(d2mp_config, "use_const_v_fallback", True)

#         for i, (trk, px) in enumerate(zip(metas, preds_px)):
#             fused = px
#             if trk.last_observation.sum() > 0:
#                 last = trk.last_observation[:4]
#                 if gate > 0:
#                     iou = self._iou_single(px, last)
#                     if iou < gate:
#                         if use_cv and trk.velocity is not None:
#                             w = last[2] - last[0]; h = last[3] - last[1]
#                             cx = (last[0] + last[2]) * 0.5 + float(trk.velocity[1])
#                             cy = (last[1] + last[3]) * 0.5 + float(trk.velocity[0])
#                             fused = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)
#                         else:
#                             fused = last.astype(np.float32)
#                 if bl < 1.0:
#                     fused = (bl * fused + (1.0 - bl) * last).astype(np.float32)
#             preds_px[i] = clamp_xyxy(fused, (H, W))

#         # 诊断：预测 vs 最近观测的 IoU
#         ious = []
#         for trk, px in zip(metas, preds_px):
#             if trk.last_observation.sum() > 0:
#                 ious.append(self._iou_single(px, trk.last_observation[:4]))
#         if len(ious) > 0 and self._dbg_pred_cnt < 200:
#             self._dbg_pred_cnt += 1
#             self._dbg_iou_running.append(float(np.median(ious)))
#             if self._dbg_pred_cnt % 20 == 0:
#                 print(f"[D2MP][diag] median IoU(pred vs obs) over last {len(self._dbg_iou_running)} steps: "
#                       f"{np.median(self._dbg_iou_running):.3f}")

#         # 同步计数器
#         for trk, px in zip(metas, preds_px):
#             trk.age += 1
#             if trk.time_since_update > 0:
#                 trk.hit_streak = 0
#             trk.time_since_update += 1
#             trk.last_bbox = px.astype(np.float32)

#         return preds_px

#     def update(self, output_results, img_info, img_size):
#         """
#         Params:
#           output_results: numpy array of detections [[x1,y1,x2,y2,score], ...]
#           img_info: (img_h, img_w, ...)
#           img_size: (test_input_h, test_input_w) - for scaling back
#         """
#         if output_results is None:
#             return np.empty((0, 5))

#         self.frame_count += 1

#         if output_results.shape[1] == 5:
#             scores = output_results[:, 4]
#             bboxes = output_results[:, :4]
#         else:
#             output_results = output_results.cpu().numpy()
#             scores = output_results[:, 4] * output_results[:, 5]
#             bboxes = output_results[:, :4]
#         img_h, img_w = _to_int(img_info[0]), _to_int(img_info[1])
#         self.last_img_size = (img_h, img_w)
#         scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))

#         bboxes = bboxes / scale
#         dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)

#         inds_low = scores > 0.1
#         inds_high = scores < self.det_thresh
#         inds_second = np.logical_and(inds_low, inds_high)
#         dets_second = dets[inds_second]
#         remain_inds = scores > self.det_thresh
#         dets = dets[remain_inds]

#         # 批量扩散预测
#         trks = np.zeros((len(self.trackers), 5), dtype=np.float32)
#         to_del = []
#         ret = []
#         if len(self.trackers) > 0:
#             preds_px = self._batched_predict(self.trackers, img_size=(img_h, img_w))  # (N,4)
#             for t in range(len(self.trackers)):
#                 pos = preds_px[t]
#                 trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
#                 if np.any(np.isnan(pos)):
#                     to_del.append(t)

#         trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#         for t in reversed(to_del):
#             self.trackers.pop(t)

#         velocities = np.array(
#             [trk.velocity if trk.velocity is not None else np.array((0, 0), dtype=np.float32) for trk in self.trackers])
#         last_boxes = np.array([trk.last_observation for trk in self.trackers])
#         k_observations = np.array(
#             [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

#         # First round of association
#         matched, unmatched_dets, unmatched_trks = associate(
#             dets, trks, self.iou_threshold, velocities, k_observations, self.inertia)
#         for m in matched:
#             self.trackers[m[1]].update(dets[m[0], :], img_size=(img_h, img_w))

#         # Second round of association by OCR (BYTE optional)
#         if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
#             u_trks = trks[unmatched_trks]
#             iou_left = self.asso_func(dets_second, u_trks)
#             iou_left = np.array(iou_left)
#             if iou_left.max() > self.iou_threshold:
#                 matched_indices = linear_assignment(-iou_left)
#                 to_remove_trk_indices = []
#                 for m in matched_indices:
#                     det_ind, trk_ind = m[0], unmatched_trks[m[1]]
#                     if iou_left[m[0], m[1]] < self.iou_threshold:
#                         continue
#                     self.trackers[trk_ind].update(dets_second[det_ind, :], img_size=(img_h, img_w))
#                     to_remove_trk_indices.append(trk_ind)
#                 unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

#         if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
#             left_dets = dets[unmatched_dets]
#             left_trks = last_boxes[unmatched_trks]
#             iou_left = self.asso_func(left_dets, left_trks)
#             iou_left = np.array(iou_left)
#             if iou_left.max() > self.iou_threshold:
#                 rematched_indices = linear_assignment(-iou_left)
#                 to_remove_det_indices = []
#                 to_remove_trk_indices = []
#                 for m in rematched_indices:
#                     det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
#                     if iou_left[m[0], m[1]] < self.iou_threshold:
#                         continue
#                     self.trackers[trk_ind].update(dets[det_ind, :], img_size=(img_h, img_w))
#                     to_remove_det_indices.append(det_ind)
#                     to_remove_trk_indices.append(trk_ind)
#                 unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
#                 unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

#         for m in unmatched_trks:
#             self.trackers[m].update(None)

#         # create and initialise new trackers
#         for i in unmatched_dets:
#             trk = DiffusionBoxTracker(dets[i, :], img_size=(img_h, img_w), delta_t=self.delta_t)
#             self.trackers.append(trk)

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
#         Public detections variant.
#         Note: will use last known image size from update(); if None, falls back to (1080, 1920).
#         """
#         self.frame_count += 1

#         det_scores = np.ones((dets.shape[0], 1), dtype=np.float32)
#         dets = np.concatenate((dets, det_scores), axis=1)

#         remain_inds = scores > self.det_thresh
#         cates = cates[remain_inds]
#         dets = dets[remain_inds]

#         trks = np.zeros((len(self.trackers), 5), dtype=np.float32)
#         to_del = []
#         ret = []
#         for t, trk in enumerate(self.trackers):
#             pos = trk.predict()[0]
#             cat = getattr(trk, 'cate', 0)
#             trks[t, :] = [pos[0], pos[1], pos[2], pos[3], cat]
#             if np.any(np.isnan(pos)):
#                 to_del.append(t)
#         trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#         for t in reversed(to_del):
#             self.trackers.pop(t)

#         velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0, 0), dtype=np.float32) for trk in self.trackers])
#         last_boxes = np.array([trk.last_observation for trk in self.trackers])
#         k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

#         matched, unmatched_dets, unmatched_trks = associate_kitti(
#             dets, trks, cates, self.iou_threshold, velocities, k_observations, self.inertia
#         )

#         for m in matched:
#             img_size = self.last_img_size if self.last_img_size is not None else (1080, 1920)
#             self.trackers[m[1]].update(dets[m[0], :], img_size=img_size)

#         if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
#             left_dets = dets[unmatched_dets]
#             left_trks = last_boxes[unmatched_trks]
#             left_dets_c = left_dets.copy()
#             left_trks_c = left_trks.copy()

#             iou_left = self.asso_func(left_dets_c, left_trks_c)
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
#                     img_size = self.last_img_size if self.last_img_size is not None else (1080, 1920)
#                     self.trackers[trk_ind].update(dets[det_ind, :], img_size=img_size)
#                     to_remove_det_indices.append(det_ind)
#                     to_remove_trk_indices.append(trk_ind)
#                 unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
#                 unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

#         for i in unmatched_dets:
#             img_size = self.last_img_size if self.last_img_size is not None else (1080, 1920)
#             trk = DiffusionBoxTracker(dets[i, :], img_size=img_size, delta_t=self.delta_t)
#             trk.cate = cates[i]
#             self.trackers.append(trk)
#         i = len(self.trackers)

#         for trk in reversed(self.trackers):
#             if trk.last_observation.sum() > 0:
#                 d = trk.last_observation[:4]
#             else:
#                 d = trk.get_state()[0]
#             if trk.time_since_update < 1:
#                 if (self.frame_count <= self.min_hits) or (trk.hit_streak >= self.min_hits):
#                     ret.append(np.concatenate((d, [trk.id + 1], [getattr(trk, 'cate', 0)], [0])).reshape(1, -1))
#                 if trk.hit_streak == self.min_hits:
#                     for prev_i in range(self.min_hits - 1):
#                         prev_observation = trk.history_observations[-(prev_i + 2)]
#                         ret.append((np.concatenate((prev_observation[:4], [trk.id + 1], [trk.cate], [-(prev_i + 1)]))).reshape(1, -1))
#             i -= 1
#             if trk.time_since_update > self.max_age:
#                 self.trackers.pop(i)

#         if len(ret) > 0:
#             return np.concatenate(ret)
#         return np.empty((0, 7))

# from __future__ import print_function

# import os
# import collections
# import numpy as np
# import torch

# from .association import *
# from .diffusion import VarianceSchedule, HMINet, D2MP_OB
# from .gmc import GMC

# # 全局推理模式，避免误触梯度
# torch.set_grad_enabled(False)

# # Ampere 及以上建议开启 TF32（提升矩阵乘性能，几乎不影响精度）
# if torch.cuda.is_available():
#     try:
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True
#         if hasattr(torch, "set_float32_matmul_precision"):
#             torch.set_float32_matmul_precision('high')
#     except Exception:
#         pass


# def _to_int(x):
#     if isinstance(x, torch.Tensor):
#         return int(x.reshape(-1)[0].item())
#     return int(x)

# def warp_box_affine_xyxy(H_2x3, box_xyxy):
#     """
#     将上一帧的 xyxy 框用 2x3 仿射矩阵 H 映射到当前帧坐标系。
#     H 是 prev -> curr 的仿射：p_curr = H * [x_prev, y_prev, 1]^T
#     """
#     if H_2x3 is None:
#         return box_xyxy.astype(np.float32)
#     H = np.asarray(H_2x3, dtype=np.float32)
#     x1, y1, x2, y2 = map(float, box_xyxy)
#     pts = np.array([[x1, y1, 1.0],
#                     [x2, y1, 1.0],
#                     [x2, y2, 1.0],
#                     [x1, y2, 1.0]], dtype=np.float32)  # (4,3)
#     xy = (H @ pts.T).T  # (4,2)
#     x_min = float(np.min(xy[:, 0])); y_min = float(np.min(xy[:, 1]))
#     x_max = float(np.max(xy[:, 0])); y_max = float(np.max(xy[:, 1]))
#     return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

# # -----------------------------
# # Helpers kept from original OC-SORT
# # -----------------------------
# def k_previous_obs(observations, cur_age, k):
#     if len(observations) == 0:
#         return np.array([-1, -1, -1, -1, -1], dtype=np.float32)
#     for i in range(k):
#         dt = k - i
#         if cur_age - dt in observations:
#             return observations[cur_age - dt]
#     max_age = max(observations.keys())
#     return observations[max_age]


# def convert_bbox_to_z(bbox):
#     w = bbox[2] - bbox[0]
#     h = bbox[3] - bbox[1]
#     x = bbox[0] + w / 2.0
#     y = bbox[1] + h / 2.0
#     s = w * h
#     r = w / float(h + 1e-6)
#     return np.array([x, y, s, r]).reshape((4, 1))


# def convert_x_to_bbox(x, score=None):
#     w = np.sqrt(x[2] * x[3])
#     h = x[2] / w
#     if score is None:
#         return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
#     else:
#         return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


# def speed_direction(bbox1, bbox2):
#     cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
#     cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
#     speed = np.array([cy2 - cy1, cx2 - cx1], dtype=np.float32)
#     norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
#     return speed / norm


# ASSO_FUNCS = {
#     "iou": iou_batch,
#     "giou": giou_batch,
#     "ciou": ciou_batch,
#     "diou": diou_batch,
#     "ct_dist": ct_dist,
# }


# # -----------------------------
# # D2MP config + utilities
# # -----------------------------
# class D2MPConfig:
#     def __init__(self):
#         # 数值稳定项（D2MP_OB 需要）
#         self.eps = 1e-5

#         # bbox 表达，Diffmot 训练默认 cxcywh
#         self.bbox_format = 'cxcywh'

#         # 对 generate 分支无额外归一化，sample 分支仍可用
#         self.bbox_mean = [0.0, 0.0, 0.0, 0.0]
#         self.bbox_std  = [1.0, 1.0, 1.0, 1.0]
        
#         # self.bbox_mean = [0.5, 0.5, 0.5, 0.5]
#         # self.bbox_std  = [0.5, 0.5, 0.5, 0.5]

#         self.context_dim = None
#         self.hist_len = 5

#         # 扩散调度与步数（尽量对齐训练）
#         self.num_steps = 1
#         self.beta_1 = 1e-4
#         self.beta_T = 5e-2
#         self.var_mode = 'linear'  # 若你确认支持 'cosine' 再改

#         # 模型结构
#         self.point_dim = 4
#         self.tf_layer  = 3

#         # 其他开关
#         self.seed = 12345
#         self.use_fp16 = False

#         # 稳健性
#         self.pred_blend = 0.5
#         self.pred_gate_iou = 0.05
#         self.use_const_v_fallback = True
        
#         self.prefer_generate = True
#         self.gen_residual_scale = 0.6  # 0.4~0.8 之间调，越小越稳，越大越跟随


# d2mp_config = D2MPConfig()


# def xyxy_to_cxcywh(xyxy):
#     x1, y1, x2, y2 = xyxy
#     w = max(x2 - x1, 1e-6)
#     h = max(y2 - y1, 1e-6)
#     cx = x1 + w / 2.0
#     cy = y1 + h / 2.0
#     return np.array([cx, cy, w, h], dtype=np.float32)


# def cxcywh_to_xyxy(cxcywh):
#     cx, cy, w, h = cxcywh
#     x1 = cx - w / 2.0
#     y1 = cy - h / 2.0
#     x2 = cx + w / 2.0
#     y2 = cy + h / 2.0
#     return np.array([x1, y1, x2, y2], dtype=np.float32)


# def to_unit_xyxy(xyxy, img_size):
#     H, W = img_size
#     H, W = _to_int(H), _to_int(W)
#     x1, y1, x2, y2 = xyxy
#     return np.array([x1 / W, y1 / H, x2 / W, y2 / H], dtype=np.float32)


# def to_pixel_xyxy(unit_xyxy, img_size):
#     H, W = img_size
#     H, W = _to_int(H), _to_int(W)
#     x1, y1, x2, y2 = unit_xyxy
#     return np.array([x1 * W, y1 * H, x2 * W, y2 * H], dtype=np.float32)


# def clamp_xyxy(xyxy, img_size):
#     H, W = img_size
#     H, W = _to_int(H), _to_int(W)
#     x1, y1, x2, y2 = xyxy
#     x1 = max(0, min(W - 1, x1))
#     y1 = max(0, min(H - 1, y1))
#     x2 = max(0, min(W - 1, x2))
#     y2 = max(0, min(H - 1, y2))
#     if x2 < x1:
#         x2 = x1
#     if y2 < y1:
#         y2 = y1
#     return np.array([x1, y1, x2, y2], dtype=np.float32)


# # -----------------------------
# # Global D2MP model init (lazy)
# # -----------------------------
# _global_d2mp = None


# def _infer_context_dim_and_hist_len_from_state(state):
#     # Try to infer context_dim from concat1._hyper_bias.weight in HMINet
#     ctx_dim = None
#     for k, v in state.items():
#         if isinstance(k, str) and k.endswith('concat1._hyper_bias.weight') and hasattr(v, "ndim") and v.ndim == 2:
#             ctx_dim = int(v.shape[1]) - 3  # - time_emb(3)
#             break
#     if ctx_dim is not None and ctx_dim > 0:
#         d2mp_config.context_dim = ctx_dim
#         if ctx_dim % 4 == 0:
#             d2mp_config.hist_len = ctx_dim // 4


# def init_d2mp_model():
#     """
#     Lazy init global D2MP model with robust checkpoint parsing.
#     """
#     global _global_d2mp
#     if _global_d2mp is not None:
#         return _global_d2mp

#     if d2mp_config.seed is not None:
#         torch.manual_seed(d2mp_config.seed)
#         np.random.seed(d2mp_config.seed)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Use your absolute path
#     weight_path = "/public/home/cookie/anaconda3/envs/ocsort_env/OC_SORT/trackers/diffmot_tracker/weights/MOT_epoch800.pt"

#     # 1) Load raw checkpoint
#     try:
#         state_raw = torch.load(weight_path, map_location=device, weights_only=True)
#     except TypeError:
#         state_raw = torch.load(weight_path, map_location=device)

#     # 2) Unwrap common containers
#     if isinstance(state_raw, dict):
#         if 'state_dict' in state_raw and isinstance(state_raw['state_dict'], dict):
#             state_raw = state_raw['state_dict']
#         elif 'model' in state_raw and isinstance(state_raw['model'], dict):
#             state_raw = state_raw['model']
#         elif 'ddpm' in state_raw and isinstance(state_raw['ddpm'], dict):
#             inner = state_raw['ddpm']
#             if isinstance(inner, dict) and 'state_dict' in inner and isinstance(inner['state_dict'], dict):
#                 state_raw = inner['state_dict']
#             else:
#                 state_raw = inner

#     # 3) Strip 'module.' prefix if any
#     if isinstance(state_raw, dict) and any(isinstance(k, str) and k.startswith('module.') for k in state_raw.keys()):
#         state_raw = {k.replace('module.', '', 1): v for k, v in state_raw.items()}

#     # 4) Build flat state dict
#     state = {}
#     if isinstance(state_raw, dict):
#         if 'net' in state_raw and isinstance(state_raw['net'], dict):
#             for k, v in state_raw['net'].items():
#                 key = f'net.{k}' if not k.startswith('net.') else k
#                 state[key] = v
#         elif any(isinstance(k, str) and k.startswith('net.') for k in state_raw.keys()):
#             for k, v in state_raw.items():
#                 if k.startswith('net.'):
#                     state[k] = v
#         else:
#             for k, v in state_raw.items():
#                 if any(k.startswith(p) for p in ['optimizer', 'epoch', 'ema', 'amp', 'scaler', 'global_step']):
#                     continue
#                 if k.startswith('var_sched.'):
#                     state[k] = v
#                 else:
#                     state[f'net.{k}'] = v

#         if 'var_sched' in state_raw and isinstance(state_raw['var_sched'], dict):
#             for k, v in state_raw['var_sched'].items():
#                 key = f'var_sched.{k}' if not k.startswith('var_sched.') else k
#                 state[key] = v
#         elif any(isinstance(k, str) and k.startswith('var_sched.') for k in state_raw.keys()):
#             for k, v in state_raw.items():
#                 if k.startswith('var_sched.'):
#                     state[k] = v

#     # 5) Logging
#     num_net = sum(1 for k in state.keys() if isinstance(k, str) and k.startswith('net.'))
#     num_sched = sum(1 for k in state.keys() if isinstance(k, str) and k.startswith('var_sched.'))
#     if num_net == 0:
#         print('[D2MP] WARNING: Could not find net weights in checkpoint.')
#     else:
#         print(f'[D2MP] Parsed weights: net keys = {num_net}, var_sched keys = {num_sched}')

#     # 6) Infer context_dim / hist_len
#     _infer_context_dim_and_hist_len_from_state(state)
#     context_dim = d2mp_config.context_dim if d2mp_config.context_dim is not None else (d2mp_config.hist_len * 4)

#     # 7) Build model
#     var_sched = VarianceSchedule(
#         num_steps=d2mp_config.num_steps,
#         mode=d2mp_config.var_mode,
#         beta_1=d2mp_config.beta_1,
#         beta_T=d2mp_config.beta_T
#     )
#     hminet = HMINet(
#         point_dim=d2mp_config.point_dim,
#         context_dim=context_dim,
#         tf_layer=d2mp_config.tf_layer
#     )
    
#     if not hasattr(d2mp_config, "eps"):
#         d2mp_config.eps = 1e-5
    
#     model = D2MP_OB(
#         net=hminet,
#         var_sched=var_sched,
#         config=d2mp_config
#     )

#     # 8) Load weights
#     try:
#         model.load_state_dict(state, strict=False)
#     except Exception as e:
#         print(f"[D2MP] load_state_dict failed: {e}")

#     model.eval().to(device)
#     if getattr(d2mp_config, "use_fp16", False) and device.type == "cuda":
#         model.half()

#     print(f"[D2MP] cfg => format={d2mp_config.bbox_format}, hist_len={d2mp_config.hist_len}, "
#           f"num_steps={d2mp_config.num_steps}, var_mode={d2mp_config.var_mode}, "
#           f"use_fp16={d2mp_config.use_fp16}")
#     _global_d2mp = model
#     return _global_d2mp


# # -----------------------------
# # Diffusion-based tracker
# # -----------------------------
# class DiffusionBoxTracker(object):
#     """
#     D2MP-based tracker replacing Kalman filter.
#     Maintains same external interface for OC-SORT association.
#     """
#     count = 0

#     def __init__(self, bbox, img_size, delta_t=3):
#         """
#         bbox: [x1, y1, x2, y2, score] in pixel coords
#         img_size: (H, W)
#         """
#         self.img_size = img_size  # (H, W)
#         # OC-SORT counters
#         self.time_since_update = 0
#         self.id = DiffusionBoxTracker.count
#         DiffusionBoxTracker.count += 1
#         self.hits = 0
#         self.hit_streak = 0
#         self.age = 0

#         # histories
#         self.history_bboxes = [bbox[:4].astype(np.float32)]
#         self.last_bbox = bbox[:4].astype(np.float32)
#         self.last_observation = np.array([-1, -1, -1, -1, -1], dtype=np.float32)
#         self.observations = dict()
#         self.history_observations = []
#         self.velocity = None
#         self.delta_t = delta_t

#         # Diffmot 兼容：维护 conds（像素域）
#         self.prev_accept_xywh = None
#         self.diff_conds = collections.deque([], maxlen=5)

#         # 初始化立即作为一次观测
#         self.update(bbox, img_size=img_size)

#     def set_img_size(self, img_size):
#         H, W = img_size
#         H, W = _to_int(H), _to_int(W)
#         self.img_size = (H, W)

#     # normalization helpers for sample fallback
#     def _normalize_tensor(self, bbox_tensor):
#         mean = torch.tensor(d2mp_config.bbox_mean, device=bbox_tensor.device, dtype=bbox_tensor.dtype)
#         std = torch.tensor(d2mp_config.bbox_std, device=bbox_tensor.device, dtype=bbox_tensor.dtype)
#         return (bbox_tensor - mean) / std

#     def _denormalize_tensor(self, norm_tensor):
#         mean = torch.tensor(d2mp_config.bbox_mean, device=norm_tensor.device, dtype=norm_tensor.dtype)
#         std = torch.tensor(d2mp_config.bbox_std, device=norm_tensor.device, dtype=norm_tensor.dtype)
#         return norm_tensor * std + mean

#     def update(self, bbox, img_size=None):
#         """
#         Update with detection bbox (x1,y1,x2,y2,score) or None.
#         """
#         if img_size is not None:
#             self.set_img_size(img_size)

#         if bbox is not None:
#             bbox = bbox.astype(np.float32)
#             # velocity (direction) using observation delta_t steps away
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

#             # maintain obs history like original OC-SORT
#             self.last_observation = bbox.copy()
#             self.observations[self.age] = bbox.copy()
#             self.history_observations.append(bbox.copy())

#             # maintain bbox history for D2MP (store xyxy)
#             self.last_bbox = bbox[:4].copy()
#             self.history_bboxes.append(self.last_bbox.copy())
#             if len(self.history_bboxes) > d2mp_config.hist_len:
#                 self.history_bboxes = self.history_bboxes[-d2mp_config.hist_len:]

#             # Diffmot conds: [xywh, delta_xywh] in pixels
#             x1, y1, x2, y2 = self.last_bbox
#             cx = 0.5 * (x1 + x2)
#             cy = 0.5 * (y1 + y2)
#             w = max(x2 - x1, 1.0)
#             h = max(y2 - y1, 1.0)
#             xywh = np.array([cx, cy, w, h], dtype=np.float32)
#             if self.prev_accept_xywh is None:
#                 delta = np.zeros(4, dtype=np.float32)
#             else:
#                 delta = (xywh - self.prev_accept_xywh).astype(np.float32)
#             tmp_conds = np.concatenate([xywh, delta]).astype(np.float32)
#             self.diff_conds.append(tmp_conds)
#             self.prev_accept_xywh = xywh.copy()

#             # OC-SORT counters
#             self.time_since_update = 0
#             self.hits += 1
#             self.hit_streak += 1
#         else:
#             # no observation this frame
#             self.time_since_update += 1

#     def predict(self):
#         """
#         Predict next bbox with D2MP (single-track fallback).
#         主流程使用 batched 版本；public 模式等仍会用到该方法。
#         Returns shape (1,4) in pixel coords.
#         """
#         model = init_d2mp_model()
#         device = next(model.parameters()).device

#         # sample 分支（保持原逻辑）
#         H, W = self.img_size
#         L = d2mp_config.hist_len
#         hist = self.history_bboxes[-L:]
#         if len(hist) < L:
#             hist = hist + [hist[-1]] * (L - len(hist))

#         hist_unit = [to_unit_xyxy(b, (H, W)) for b in hist]
#         hist_unit = np.stack(hist_unit, axis=0)  # (L,4)

#         if d2mp_config.bbox_format == 'cxcywh':  # 原cxcywh
#             hist_unit = np.stack([xyxy_to_cxcywh(b) for b in hist_unit], axis=0)

#         inp = torch.tensor(hist_unit, dtype=torch.float32, device=device)  # (L,4)
#         inp = self._normalize_tensor(inp)
#         context = inp.flatten().unsqueeze(0)  # (1, L*4)

#         with torch.inference_mode():
#             pred = model.sample(
#                 context=context,
#                 sample=1,
#                 bestof=False,
#                 point_dim=d2mp_config.point_dim,
#                 ret_traj=False
#             )

#         if isinstance(pred, (list, tuple)):
#             pred = pred[0]
#         pred = pred[0, 0] if pred.ndim == 3 else pred.squeeze(0)  # (4,)
#         pred = self._denormalize_tensor(pred)

#         pred_unit = pred.detach().cpu().numpy()
#         if d2mp_config.bbox_format == 'cxcywh': #cxcywh
#             pred_unit = cxcywh_to_xyxy(pred_unit)

#         pred_px = to_pixel_xyxy(pred_unit, (H, W))
#         pred_px = clamp_xyxy(pred_px, (H, W))

#         # advance counters
#         self.age += 1
#         if self.time_since_update > 0:
#             self.hit_streak = 0
#         self.time_since_update += 1

#         self.last_bbox = pred_px.astype(np.float32)
#         return self.last_bbox.reshape(1, 4)

#     def get_state(self):
#         return self.last_bbox.reshape(1, 4)


# # -----------------------------
# # OCSort using DiffusionBoxTracker
# # -----------------------------
# class Diff_OCSort(object):
#     def __init__(self, det_thresh, max_age=30, min_hits=3,
#                  iou_threshold=0.3, delta_t=3, asso_func="iou", inertia=0.0, use_byte=False,use_gmc=True, gmc_method='sparseOptFlow', gmc_downscale=2):
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

#         DiffusionBoxTracker.count = 0
#         self.last_img_size = None  # (H,W)

#         # 诊断用
#         self._dbg_pred_cnt = 0
#         self._dbg_iou_running = []

#         # 跨序列检测状态
#         self._prev_frame_id = -1  # 如果外层传 frame_id，可用于回绕检测
        
#         self.use_gmc = use_gmc
#         self.gmc = GMC(method=gmc_method, downscale=gmc_downscale) if use_gmc else None
#         self._last_H_prev2curr = None  # 仅用于调试或备用

#     # --- 新增：跨序列 reset ---
#     def reset(self):
#         """
#         清空所有与单序列相关的状态，准备下一个视频序列。
#         """
#         # 清空跟踪器与计数
#         self.trackers = []
#         self.frame_count = 0
#         self.last_img_size = None

#         # 重置 Track ID 计数器
#         DiffusionBoxTracker.count = 0

#         # 清空诊断
#         self._dbg_pred_cnt = 0
#         self._dbg_iou_running = []
#         if hasattr(self, "_pred_shape_log"):
#             self._pred_shape_log = 0

#         # 跨序列检测状态
#         self._prev_frame_id = -1
        
#         if self.gmc is not None:
#             self.gmc.initializedFirstFrame = False
#         self._last_H_prev2curr = None

#     def _maybe_reset_on_boundary(self, frame_id=None, img_size=None):
#         """
#         如果检测到“frame_id 回绕”或“分辨率变化”，自动 reset。
#         - frame_id: 外层若能传，优先用它判断回绕（如从 300 -> 1）。
#         - img_size: (H,W)，不同序列常有不同分辨率，也作为触发条件。
#         """
#         fid_back = (self._prev_frame_id != -1 and frame_id is not None and frame_id <= self._prev_frame_id)
#         size_changed = (img_size is not None and self.last_img_size is not None
#                         and tuple(img_size) != tuple(self.last_img_size))
#         if fid_back or size_changed:
#             print(f"[D2MP] new sequence detected (fid_back={fid_back}, size_changed={size_changed}), resetting tracker")
#             self.reset()
#         # 更新最近状态（即使 reset 了也要记录当前信息）
#         if frame_id is not None:
#             self._prev_frame_id = int(frame_id)
#         if img_size is not None:
#             self.last_img_size = tuple(img_size)

#     def _iou_single(self, a, b):
#         x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
#         x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
#         iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
#         inter = iw * ih
#         area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
#         area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
#         union = area_a + area_b - inter + 1e-6
#         return inter / union

#     # def _batched_predict(self, trackers, img_size):
#     def _batched_predict(self, trackers, img_size, H_prev2curr=None):
#         """
#         - 如模型具备 generate(): 走 Diffmot 语义（conds 像素域，输出偏移）
#         - 否则：回退到 sample(context)（unit 域绝对框）
#         返回 (N,4) xyxy 像素坐标，并同步计数器和 last_bbox。
#         """
#         if len(trackers) == 0:
#             return np.zeros((0, 4), dtype=np.float32)

#         model = init_d2mp_model()
#         device = next(model.parameters()).device
#         H, W = img_size

#         # has_gen = callable(getattr(model, "generate", None))
#         # print(f"[D2MP] Predict branch: {'generate' if has_gen else 'sample'} | num_steps={d2mp_config.num_steps}, hist_len={d2mp_config.hist_len}")
        
#         has_gen = getattr(d2mp_config, "prefer_generate", True) and callable(getattr(model, "generate", None))
        
#         # 优先使用 Diffmot 风格的 generate 接口
#         # if hasattr(model, "generate"):
#         #     dets_norm = []
#         #     conds_list = []
#         #     metas = []
#         #     for trk in trackers:
#         #         trk.set_img_size((H, W))
#         #         # 基线使用“上次接受的观测”
#         #         if getattr(trk, "prev_accept_xywh", None) is not None:
#         #             cx, cy, w, h = trk.prev_accept_xywh
#         #         else:
#         #             x1, y1, x2, y2 = trk.last_observation[:4] if trk.last_observation.sum() > 0 else trk.last_bbox
#         #             cx = 0.5 * (x1 + x2)
#         #             cy = 0.5 * (y1 + y2)
#         #             w = max(1.0, x2 - x1)
#         #             h = max(1.0, y2 - y1)
#         #         dets_norm.append([cx / W, cy / H, w / W, h / H])

#         #         # conds（像素域，含 delta）
#         #         if not hasattr(trk, "diff_conds") or len(trk.diff_conds) == 0:
#         #             delta = np.zeros(4, dtype=np.float32)
#         #             cond = np.array([cx, cy, w, h, *delta], dtype=np.float32)
#         #             conds = [cond]
#         #         else:
#         #             conds = list(trk.diff_conds)
#         #         conds_list.append(conds)
#         #         metas.append(trk)

#         #     dets_norm = np.asarray(dets_norm, dtype=np.float32)

#         #     # with torch.inference_mode():
#         #     #     multi_track_pred = model.generate(
#         #     #         conds_list, sample=1, bestof=True, img_w=W, img_h=H
#         #     #     )

#         #     # if isinstance(multi_track_pred, (list, tuple)):
#         #     #     multi_track_pred = multi_track_pred[0]
#         #     # pred = multi_track_pred
#         #     # pred = np.asarray(pred)
#         #     # if pred.ndim == 3:
#         #     #     pred = pred.mean(0)  # (N,4) 归一化域 residual cxcywh

#         #     with torch.inference_mode():
#         #         multi_track_pred = model.generate(
#         #             conds_list, sample=1, bestof=True, img_w=W, img_h=H
#         #         )
#         #     if isinstance(multi_track_pred, (list, tuple)):
#         #         multi_track_pred = multi_track_pred[0]
#         #     pred = np.asarray(multi_track_pred)
#         #     if pred.ndim == 3:  # (S,N,4) -> (N,4)
#         #         pred = pred.mean(0)

#         #     # 关键：对残差做温和缩放，避免过冲
#         #     res_scale = float(getattr(d2mp_config, "gen_residual_scale", 0.6))
#         #     pred = pred * res_scale  # 仍是归一化域的残差 cxcywh

#         #     track_pred = pred + dets_norm  # 归一化 cxcywh -> 绝对 cxcywh
#         #     # 后续像素化与 xyxy 转换保持不变

#         #     track_pred = pred + dets_norm  # 归一化域 cxcywh

#         #     # 归一化 -> 像素，并转 xyxy
#         #     track_pred_px = track_pred.copy()
#         #     track_pred_px[:, 0::2] *= float(W)  # cx,w
#         #     track_pred_px[:, 1::2] *= float(H)  # cy,h
#         #     x_tl = track_pred_px[:, 0] - track_pred_px[:, 2] / 2.0
#         #     y_tl = track_pred_px[:, 1] - track_pred_px[:, 3] / 2.0
#         #     w_px = track_pred_px[:, 2]; h_px = track_pred_px[:, 3]
#         #     x1 = x_tl; y1 = y_tl; x2 = x_tl + w_px; y2 = y_tl + h_px
#         #     preds_px = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
#         #     preds_px[:, 0] = np.clip(preds_px[:, 0], 0, W - 1)
#         #     preds_px[:, 1] = np.clip(preds_px[:, 1], 0, H - 1)
#         #     preds_px[:, 2] = np.clip(preds_px[:, 2], 0, W - 1)
#         #     preds_px[:, 3] = np.clip(preds_px[:, 3], 0, H - 1)
        
#         if hasattr(model, "generate") and getattr(d2mp_config, "prefer_generate", True):
#             dets_norm = []
#             conds_list = []
#             metas = []
#             gmc_anchors_px = []

#             for trk in trackers:
#                 trk.set_img_size((H, W))

#                 # 取上一帧的参考框（优先观测，否则last_bbox）
#                 prev_box = trk.last_observation[:4] if trk.last_observation.sum() > 0 else trk.last_bbox
#                 # 用 GMC 仿射映射到当前帧，得到稳定锚
#                 gmc_box = warp_box_affine_xyxy(H_prev2curr, prev_box)
#                 gmc_box = clamp_xyxy(gmc_box, (H, W))
#                 gmc_anchors_px.append(gmc_box)

#                 # 转 cxcywh（像素）
#                 x1, y1, x2, y2 = gmc_box
#                 cx = 0.5 * (x1 + x2); cy = 0.5 * (y1 + y2)
#                 w = max(1.0, x2 - x1); h = max(1.0, y2 - y1)

#                 # 归一化锚，作为 generate 的基线
#                 dets_norm.append([cx / W, cy / H, w / W, h / H])

#                 # 条件（像素域），你的 diff_conds 原样提供即可
#                 if not hasattr(trk, "diff_conds") or len(trk.diff_conds) == 0:
#                     delta = np.zeros(4, dtype=np.float32)
#                     cond = np.array([cx, cy, w, h, *delta], dtype=np.float32)
#                     conds = [cond]
#                 else:
#                     conds = list(trk.diff_conds)
#                 conds_list.append(conds)
#                 metas.append(trk)

#             dets_norm = np.asarray(dets_norm, dtype=np.float32)

#             with torch.inference_mode():
#                 multi_track_pred = model.generate(conds_list, sample=1, bestof=True, img_w=W, img_h=H)
#             if isinstance(multi_track_pred, (list, tuple)):
#                 multi_track_pred = multi_track_pred[0]
#             pred = np.asarray(multi_track_pred)
#             if pred.ndim == 3:
#                 pred = pred.mean(0)  # (N,4) 残差，归一化 cxcywh

#             # 动态残差缩放 + 尺度夹紧（沿用你当前逻辑）
#             rs_min = float(getattr(d2mp_config, "gen_residual_scale_min", 0.3))
#             rs_max = float(getattr(d2mp_config, "gen_residual_scale_max", 0.8))
#             scales = np.array([rs_min + (rs_max - rs_min) * float(getattr(trk, "pred_iou_ema", 0.6))
#                                 for trk in metas], dtype=np.float32).reshape(-1, 1)
#             pred = pred * scales

#             track_pred = pred + dets_norm  # 合成：归一化 cxcywh
#             ratio = float(getattr(d2mp_config, "size_change_limit", 0.4))
#             if ratio > 0:
#                 w0 = dets_norm[:, 2:3]; h0 = dets_norm[:, 3:4]
#                 track_pred[:, 2:3] = np.clip(track_pred[:, 2:3], w0 * (1 - ratio), w0 * (1 + ratio))
#                 track_pred[:, 3:4] = np.clip(track_pred[:, 3:4], h0 * (1 - ratio), h0 * (1 + ratio))

#             # 还原到像素 xyxy
#             tp = track_pred.copy()
#             tp[:, 0::2] *= float(W); tp[:, 1::2] *= float(H)
#             x_tl = tp[:, 0] - tp[:, 2] / 2.0; y_tl = tp[:, 1] - tp[:, 3] / 2.0
#             preds_px = np.stack([x_tl, y_tl, x_tl + tp[:, 2], y_tl + tp[:, 3]], axis=1).astype(np.float32)
#             preds_px[:, 0] = np.clip(preds_px[:, 0], 0, W - 1)
#             preds_px[:, 1] = np.clip(preds_px[:, 1], 0, H - 1)
#             preds_px[:, 2] = np.clip(preds_px[:, 2], 0, W - 1)
#             preds_px[:, 3] = np.clip(preds_px[:, 3], 0, H - 1)

#             # 保存“原始预测”供 EMA 更新
#             for trk, px in zip(metas, preds_px):
#                 trk.pending_pred = px.copy()

#             # 门控 + 与 GMC 锚的微融合（更连续）
#             gate_iou = float(getattr(d2mp_config, "pred_gate_iou", 0.30))
#             cv_mix_alpha = float(getattr(d2mp_config, "cv_mix_alpha", 0.2))
#             for i, (trk, px_pred, gmc_px) in enumerate(zip(metas, preds_px, gmc_anchors_px)):
#                 fused = px_pred
#                 if trk.last_observation.sum() > 0:
#                     last = trk.last_observation[:4]
#                     if self._iou_single(px_pred, last) < gate_iou:
#                         fused = gmc_px.astype(np.float32)  # 退回 GMC
#                     else:
#                         fused = (1.0 - cv_mix_alpha) * gmc_px + cv_mix_alpha * px_pred
#                 preds_px[i] = clamp_xyxy(fused.astype(np.float32), (H, W))
        
#         else:
#             # 回退到 sample(context) 路径
#             L = d2mp_config.hist_len
#             contexts = []
#             metas = []
#             for trk in trackers:
#                 trk.set_img_size((H, W))
#                 hist = trk.history_bboxes[-L:]
#                 if len(hist) < L:
#                     hist = hist + [hist[-1]] * (L - len(hist))
#                 hist_unit = [to_unit_xyxy(b, (H, W)) for b in hist]
#                 hist_unit = np.stack(hist_unit, axis=0)  # (L,4)
#                 if d2mp_config.bbox_format == 'cxcywh': # cxcywh
#                     hist_unit = np.stack([xyxy_to_cxcywh(b) for b in hist_unit], axis=0)
#                 contexts.append(hist_unit)
#                 metas.append(trk)

#             contexts = torch.tensor(np.stack(contexts, axis=0), dtype=torch.float32, device=device)  # (B,L,4)
#             mean = torch.tensor(d2mp_config.bbox_mean, device=device, dtype=contexts.dtype).view(1, 1, 4)
#             std = torch.tensor(d2mp_config.bbox_std, device=device, dtype=contexts.dtype).view(1, 1, 4)
#             contexts = (contexts - mean) / std
#             contexts = contexts.flatten(start_dim=1)

#             if getattr(d2mp_config, "use_fp16", False) and device.type == "cuda":
#                 contexts = contexts.half()

#             with torch.inference_mode():
#                 pred = model.sample(
#                     context=contexts,
#                     sample=1,
#                     bestof=False,
#                     point_dim=d2mp_config.point_dim,
#                     ret_traj=False
#                 )

#             # [INSERT POINT 1] ---- 形状诊断 + 统一到 (B,4) ----
#             if not hasattr(self, "_pred_shape_log"):
#                 self._pred_shape_log = 0
#             if self._pred_shape_log < 5:
#                 try:
#                     print("[D2MP][sample] raw pred type:", type(pred), 
#                         "shape:", tuple(pred.shape) if hasattr(pred, "shape") else None)
#                 except Exception as e:
#                     print("[D2MP][sample] shape print err:", e)
#                 self._pred_shape_log += 1

#             # 统一 pred 形状为 (B,4)
#             if isinstance(pred, (list, tuple)):
#                 pred = pred[0]
#             if pred.ndim == 3:
#                 S, B_ = pred.shape[:2]
#                 if S == 1 and B_ == len(trackers):
#                     pred = pred[0]        # (B,4)
#                 elif B_ == 1 and pred.shape[0] == len(trackers):
#                     pred = pred[:, 0]     # (B,4)
#                 else:
#                     print(f"[D2MP][warn] unexpected pred shape {tuple(pred.shape)}; expect (1,B,4) or (B,1,4), B={len(trackers)}")
#                     if pred.shape[-1] == 4:
#                         pred = pred.reshape(-1, 4)  # 粗暴兜底
#                     else:
#                         raise RuntimeError(f"Unrecognized pred shape: {tuple(pred.shape)}")

#             if pred.dtype != contexts.dtype:
#                 pred = pred.to(contexts.dtype)
#             pred = pred.view(len(trackers), 4)  # 现在应为 (B,4)
#             # [INSERT POINT 1 END]

#             # 2) 反归一化回“单位域”
#             pred = pred * std.view(1, 4) + mean.view(1, 4)  # ← 单位域

#             pred_np = pred.detach().float().cpu().numpy()

#             # 3) bbox_format 转换
#             if d2mp_config.bbox_format == 'cxcywh':
#                 pred_np = np.stack([cxcywh_to_xyxy(p) for p in pred_np], axis=0)
#             elif d2mp_config.bbox_format == 'xyxy':
#                 pass
#             else:
#                 print(f"[D2MP][warn] unexpected bbox_format: {d2mp_config.bbox_format}")

               
#             preds_px = np.stack([to_pixel_xyxy(p, (H, W)) for p in pred_np], axis=0)
#             preds_px = np.stack([clamp_xyxy(p, (H, W)) for p in preds_px], axis=0).astype(np.float32)

#         # # 稳健性：门控 + 融合
#         # bl = getattr(d2mp_config, "pred_blend", 0.25)
#         # gate = getattr(d2mp_config, "pred_gate_iou", 0.0)
#         # use_cv = getattr(d2mp_config, "use_const_v_fallback", True)

#         # for i, (trk, px) in enumerate(zip(metas, preds_px)):
#         #     fused = px
#         #     if trk.last_observation.sum() > 0:
#         #         last = trk.last_observation[:4]
#         #         if gate > 0:
#         #             iou = self._iou_single(px, last)
#         #             if iou < gate:
#         #                 if use_cv and trk.velocity is not None:
#         #                     w = last[2] - last[0]; h = last[3] - last[1]
#         #                     cx = (last[0] + last[2]) * 0.5 + float(trk.velocity[1])
#         #                     cy = (last[1] + last[3]) * 0.5 + float(trk.velocity[0])
#         #                     fused = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)
#         #                 else:
#         #                     fused = last.astype(np.float32)
#         #         if bl < 1.0:
#         #             fused = (bl * fused + (1.0 - bl) * last).astype(np.float32)
#         #     preds_px[i] = clamp_xyxy(fused, (H, W))
        
#         # # 稳健性：门控 + 融合（加入“前几帧更保守”的策略）
#         # bl_cfg = getattr(d2mp_config, "pred_blend", 0.5)       # 原 0.5
#         # gate_cfg = getattr(d2mp_config, "pred_gate_iou", 0.05) # 原 0.05，太低

#         # for i, (trk, px) in enumerate(zip(metas, preds_px)):
#         #     fused = px
#         #     age = getattr(trk, "age", 0)

#         #     # 前5帧更保守：更强的观测融合、更高的门控
#         #     bl_eff = bl_cfg if age >= 5 else max(bl_cfg, 0.8)    # 前5帧至少 0.8
#         #     gate_eff = gate_cfg if age >= 5 else max(gate_cfg, 0.30)  # 前5帧至少 0.30

#         #     if trk.last_observation.sum() > 0:
#         #         last = trk.last_observation[:4]
#         #         iou = self._iou_single(px, last)

#         #         # 门控：若预测偏离过大，直接回退（或用匀速外推）
#         #         if gate_eff > 0 and iou < gate_eff:
#         #             if getattr(d2mp_config, "use_const_v_fallback", True) and trk.velocity is not None:
#         #                 w = last[2] - last[0]; h = last[3] - last[1]
#         #                 cx = (last[0] + last[2]) * 0.5 + float(trk.velocity[1])
#         #                 cy = (last[1] + last[3]) * 0.5 + float(trk.velocity[0])
#         #                 fused = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)
#         #             else:
#             #             fused = last.astype(np.float32)
#             #     # 融合：向观测靠拢
#             #     if bl_eff < 1.0:
#             #         fused = (bl_eff * fused + (1.0 - bl_eff) * last).astype(np.float32)

#             # preds_px[i] = clamp_xyxy(fused, (H, W))
        
#                 # === 用 GMC 锚做微融合/回退 ===
#         gate_iou = 0.30
#         cv_mix_alpha = 0.2
#         for i, (trk, px_pred) in enumerate(zip(metas, preds_px)):
#             prev_box = trk.last_observation[:4] if trk.last_observation.sum() > 0 else trk.last_bbox
#             gmc_px = warp_box_affine_xyxy(H_prev2curr, prev_box)
#             gmc_px = clamp_xyxy(gmc_px, (H, W))

#             fused = px_pred
#             if trk.last_observation.sum() > 0:
#                 if self._iou_single(px_pred, trk.last_observation[:4]) < gate_iou:
#                     fused = gmc_px
#                 else:
#                     fused = (1.0 - cv_mix_alpha) * gmc_px + cv_mix_alpha * px_pred
#             preds_px[i] = clamp_xyxy(fused.astype(np.float32), (H, W))
    

#         # 诊断：预测 vs 最近观测的 IoU
#         ious = []
#         for trk, px in zip(metas, preds_px):
#             if trk.last_observation.sum() > 0:
#                 ious.append(self._iou_single(px, trk.last_observation[:4]))
#         if len(ious) > 0 and self._dbg_pred_cnt < 200:
#             self._dbg_pred_cnt += 1
#             self._dbg_iou_running.append(float(np.median(ious)))
#             if self._dbg_pred_cnt % 20 == 0:
#                 print(f"[D2MP][diag] median IoU(pred vs obs) over last {len(self._dbg_iou_running)} steps: "
#                       f"{np.median(self._dbg_iou_running):.3f}")

#         # 同步计数器
#         for trk, px in zip(metas, preds_px):
#             trk.age += 1
#             if trk.time_since_update > 0:
#                 trk.hit_streak = 0
#             trk.time_since_update += 1
#             trk.last_bbox = px.astype(np.float32)

#         return preds_px

#     def update(self, output_results, img_info, img_size, frame_id=None, raw_frame=None):
#         """
#         Params:
#           output_results: numpy array of detections [[x1,y1,x2,y2,score], ...]
#           img_info: (img_h, img_w, ...)
#           img_size: (test_input_h, test_input_w) - for scaling back
#           frame_id: Optional[int] - 若可传入当前帧号，可用于检测跨序列（从大到小回绕，例如跳回 1）
#         """
#         if output_results is None:
#             return np.empty((0, 5))

#         # --- 自动跨序列检测（务必在构造 contexts/预测前调用）---
#         img_h, img_w = _to_int(img_info[0]), _to_int(img_info[1])
#         self._maybe_reset_on_boundary(frame_id=frame_id, img_size=(img_h, img_w))

#         # 正常帧计数
#         self.frame_count += 1

#         if output_results.shape[1] == 5:
#             scores = output_results[:, 4]
#             bboxes = output_results[:, :4]
#         else:
#             output_results = output_results.cpu().numpy()
#             scores = output_results[:, 4] * output_results[:, 5]
#             bboxes = output_results[:, :4]

#         # 记录当前分辨率（_maybe_reset_on_boundary 已经记录，这里覆盖为同值无碍）
#         self.last_img_size = (img_h, img_w)
#         scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))

#         bboxes = bboxes / scale
#         dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)

#         inds_low = scores > 0.1
#         inds_high = scores < self.det_thresh
#         inds_second = np.logical_and(inds_low, inds_high)
#         dets_second = dets[inds_second]
#         remain_inds = scores > self.det_thresh
#         dets = dets[remain_inds]
        
#         # === GMC: 估计 prev->curr 的 2x3 仿射 H，并保存 ===
#         H_prev2curr = None
#         if self.use_gmc and raw_frame is not None and self.gmc is not None:
#             try:
#                 # 传入当前帧与“当前帧的检测”作为 mask（与 Diffmot 做法一致）
#                 H_prev2curr = self.gmc.apply(raw_frame=raw_frame, detections=dets[:, :4].astype(np.int32))
#             except Exception as e:
#                 print("[GMC] failed:", e)
#                 H_prev2curr = None
#         self._last_H_prev2curr = H_prev2curr

#         # 批量扩散预测
#         trks = np.zeros((len(self.trackers), 5), dtype=np.float32)
#         to_del = []
#         ret = []
#         # if len(self.trackers) > 0:
#         #     preds_px = self._batched_predict(self.trackers, img_size=(img_h, img_w))  # (N,4)
#         if len(self.trackers) > 0:
#             preds_px = self._batched_predict(self.trackers, img_size=(img_h, img_w), H_prev2curr=H_prev2curr)
#             for t in range(len(self.trackers)):
#                 pos = preds_px[t]
#                 trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
#                 if np.any(np.isnan(pos)):
#                     to_del.append(t)

#         trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#         for t in reversed(to_del):
#             self.trackers.pop(t)

#         velocities = np.array(
#             [trk.velocity if trk.velocity is not None else np.array((0, 0), dtype=np.float32) for trk in self.trackers])
#         last_boxes = np.array([trk.last_observation for trk in self.trackers])
#         k_observations = np.array(
#             [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

#         # First round of association
#         matched, unmatched_dets, unmatched_trks = associate(
#             dets, trks, self.iou_threshold, velocities, k_observations, self.inertia)
#         for m in matched:
#             self.trackers[m[1]].update(dets[m[0], :], img_size=(img_h, img_w))

#         # Second round of association by OCR (BYTE optional)
#         if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
#             u_trks = trks[unmatched_trks]
#             iou_left = self.asso_func(dets_second, u_trks)
#             iou_left = np.array(iou_left)
#             if iou_left.max() > self.iou_threshold:
#                 matched_indices = linear_assignment(-iou_left)
#                 to_remove_trk_indices = []
#                 for m in matched_indices:
#                     det_ind, trk_ind = m[0], unmatched_trks[m[1]]
#                     if iou_left[m[0], m[1]] < self.iou_threshold:
#                         continue
#                     self.trackers[trk_ind].update(dets_second[det_ind, :], img_size=(img_h, img_w))
#                     to_remove_trk_indices.append(trk_ind)
#                 unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

#         if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
#             left_dets = dets[unmatched_dets]
#             left_trks = last_boxes[unmatched_trks]
#             iou_left = self.asso_func(left_dets, left_trks)
#             iou_left = np.array(iou_left)
#             if iou_left.max() > self.iou_threshold:
#                 rematched_indices = linear_assignment(-iou_left)
#                 to_remove_det_indices = []
#                 to_remove_trk_indices = []
#                 for m in rematched_indices:
#                     det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
#                     if iou_left[m[0], m[1]] < self.iou_threshold:
#                         continue
#                     self.trackers[trk_ind].update(dets[det_ind, :], img_size=(img_h, img_w))
#                     to_remove_det_indices.append(det_ind)
#                     to_remove_trk_indices.append(trk_ind)
#                 unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
#                 unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

#         for m in unmatched_trks:
#             self.trackers[m].update(None)

#         # create and initialise new trackers
#         for i in unmatched_dets:
#             trk = DiffusionBoxTracker(dets[i, :], img_size=(img_h, img_w), delta_t=self.delta_t)
#             self.trackers.append(trk)

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

#     def update_public(self, dets, cates, scores, frame_id=None):
#         """
#         Public detections variant.
#         Note: will use last known image size from update(); if None, falls back to (1080, 1920).
#         frame_id: Optional[int] 仅用于外部想触发 reset 的场景（通常不需要）
#         """
#         # 若外层能传 frame_id 并希望在 public 模式也自动 reset，可启用如下检测；
#         # 这里没有当前 img_size，只能依赖 frame_id 回绕（可选）
#         if frame_id is not None:
#             self._maybe_reset_on_boundary(frame_id=frame_id, img_size=self.last_img_size)

#         self.frame_count += 1

#         det_scores = np.ones((dets.shape[0], 1), dtype=np.float32)
#         dets = np.concatenate((dets, det_scores), axis=1)

#         remain_inds = scores > self.det_thresh
#         cates = cates[remain_inds]
#         dets = dets[remain_inds]

#         trks = np.zeros((len(self.trackers), 5), dtype=np.float32)
#         to_del = []
#         ret = []
#         for t, trk in enumerate(self.trackers):
#             pos = trk.predict()[0]
#             cat = getattr(trk, 'cate', 0)
#             trks[t, :] = [pos[0], pos[1], pos[2], pos[3], cat]
#             if np.any(np.isnan(pos)):
#                 to_del.append(t)
#         trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#         for t in reversed(to_del):
#             self.trackers.pop(t)

#         velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0, 0), dtype=np.float32) for trk in self.trackers])
#         last_boxes = np.array([trk.last_observation for trk in self.trackers])
#         k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

#         matched, unmatched_dets, unmatched_trks = associate_kitti(
#             dets, trks, cates, self.iou_threshold, velocities, k_observations, self.inertia
#         )

#         for m in matched:
#             img_size = self.last_img_size if self.last_img_size is not None else (1080, 1920)
#             self.trackers[m[1]].update(dets[m[0], :], img_size=img_size)

#         if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
#             left_dets = dets[unmatched_dets]
#             left_trks = last_boxes[unmatched_trks]
#             left_dets_c = left_dets.copy()
#             left_trks_c = left_trks.copy()

#             iou_left = self.asso_func(left_dets_c, left_trks_c)
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
#                     img_size = self.last_img_size if self.last_img_size is not None else (1080, 1920)
#                     self.trackers[trk_ind].update(dets[det_ind, :], img_size=img_size)
#                     to_remove_det_indices.append(det_ind)
#                     to_remove_trk_indices.append(trk_ind)
#                 unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
#                 unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

#         for i in unmatched_dets:
#             img_size = self.last_img_size if self.last_img_size is not None else (1080, 1920)
#             trk = DiffusionBoxTracker(dets[i, :], img_size=img_size, delta_t=self.delta_t)
#             trk.cate = cates[i]
#             self.trackers.append(trk)
#         i = len(self.trackers)

#         for trk in reversed(self.trackers):
#             if trk.last_observation.sum() > 0:
#                 d = trk.last_observation[:4]
#             else:
#                 d = trk.get_state()[0]
#             if trk.time_since_update < 1:
#                 if (self.frame_count <= self.min_hits) or (trk.hit_streak >= self.min_hits):
#                     ret.append(np.concatenate((d, [trk.id + 1], [getattr(trk, 'cate', 0)], [0])).reshape(1, -1))
#                 if trk.hit_streak == self.min_hits:
#                     for prev_i in range(self.min_hits - 1):
#                         prev_observation = trk.history_observations[-(prev_i + 2)]
#                         ret.append((np.concatenate((prev_observation[:4], [trk.id + 1], [trk.cate], [-(prev_i + 1)]))).reshape(1, -1))
#             i -= 1
#             if trk.time_since_update > self.max_age:
#                 self.trackers.pop(i)

#         if len(ret) > 0:
#             return np.concatenate(ret)
#         return np.empty((0, 7))

# from __future__ import print_function  

# import os  
# import collections  
# import numpy as np  
# import torch  

# from . import denoising_diffusion_pytorch as ddpm
# _DDPM_AVAILABLE = True

# from .association import *  
# from .diffusion import VarianceSchedule, HMINet, D2MP_OB  
# from .gmc import GMC  

# # ===== helper: Hungarian (SciPy if available; otherwise greedy fallback) =====
# try:
#     from scipy.optimize import linear_sum_assignment
#     _HAS_SCIPY = True
# except Exception:
#     _HAS_SCIPY = False

# def _hungarian_assign(cost_mat: np.ndarray) -> np.ndarray:
#     """
#     cost_mat: shape (T, D) where T = number of trackers (rows), D = number of detections (cols)
#     returns: array of pairs (row_i, col_j)
#     """
#     if cost_mat.size == 0:
#         return np.empty((0, 2), dtype=int)
#     if _HAS_SCIPY:
#         r, c = linear_sum_assignment(cost_mat)
#         if r.size == 0:
#             return np.empty((0, 2), dtype=int)
#         return np.stack([r, c], axis=1).astype(int)

#     # Greedy fallback (not optimal but works without extra deps)
#     T, D = cost_mat.shape
#     flat = [(cost_mat[i, j], i, j) for i in range(T) for j in range(D)]
#     flat.sort(key=lambda x: x[0])  # lower cost is better
#     used_r, used_c, pairs = set(), set(), []
#     for cost, i, j in flat:
#         if i in used_r or j in used_c:
#             continue
#         pairs.append((i, j))
#         used_r.add(i); used_c.add(j)
#     return np.array(pairs, dtype=int) if len(pairs) else np.empty((0, 2), dtype=int)
# # ===== end helper =====

# # 全局推理模式，避免误触梯度  
# torch.set_grad_enabled(False)  

# # Ampere 及以上建议开启 TF32（提升矩阵乘性能，几乎不影响精度）  
# if torch.cuda.is_available():  
#     try:  
#         torch.backends.cuda.matmul.allow_tf32 = True  
#         torch.backends.cudnn.allow_tf32 = True  
#         if hasattr(torch, "set_float32_matmul_precision"):  
#             torch.set_float32_matmul_precision('high')  
#     except Exception:  
#         pass  


# def _to_int(x):  
#     if isinstance(x, torch.Tensor):  
#         return int(x.reshape(-1)[0].item())  
#     return int(x)  


# def warp_box_affine_xyxy(H_2x3, box_xyxy):  
#     """  
#     将上一帧的 xyxy 框用 2x3 仿射矩阵 H 映射到当前帧坐标系。  
#     H 是 prev -> curr 的仿射：p_curr = H * [x_prev, y_prev, 1]^T  
#     """  
#     if H_2x3 is None:  
#         return box_xyxy.astype(np.float32)  
#     H = np.asarray(H_2x3, dtype=np.float32)  
#     x1, y1, x2, y2 = map(float, box_xyxy)  
#     pts = np.array([[x1, y1, 1.0],  
#                     [x2, y1, 1.0],  
#                     [x2, y2, 1.0],  
#                     [x1, y2, 1.0]], dtype=np.float32)  # (4,3)  
#     xy = (H @ pts.T).T  # (4,2)  
#     x_min = float(np.min(xy[:, 0])); y_min = float(np.min(xy[:, 1]))  
#     x_max = float(np.max(xy[:, 0])); y_max = float(np.max(xy[:, 1]))  
#     return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)  


# # -----------------------------  
# # Helpers kept from original OC-SORT  
# # -----------------------------  
# def k_previous_obs(observations, cur_age, k):  
#     if len(observations) == 0:  
#         return np.array([-1, -1, -1, -1, -1], dtype=np.float32)  
#     for i in range(k):  
#         dt = k - i  
#         if cur_age - dt in observations:  
#             return observations[cur_age - dt]  
#     max_age = max(observations.keys())  
#     return observations[max_age]  


# def convert_bbox_to_z(bbox):  
#     w = bbox[2] - bbox[0]  
#     h = bbox[3] - bbox[1]  
#     x = bbox[0] + w / 2.0  
#     y = bbox[1] + h / 2.0  
#     s = w * h  
#     r = w / float(h + 1e-6)  
#     return np.array([x, y, s, r]).reshape((4, 1))  


# def convert_x_to_bbox(x, score=None):  
#     w = np.sqrt(x[2] * x[3])  
#     h = x[2] / w  
#     if score is None:  
#         return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))  
#     else:  
#         return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))  


# def speed_direction(bbox1, bbox2):  
#     cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0  
#     cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0  
#     speed = np.array([cy2 - cy1, cx2 - cx1], dtype=np.float32)  
#     norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6  
#     return speed / norm  


# ASSO_FUNCS = {  
#     "iou": iou_batch,  
#     "giou": giou_batch,  
#     "ciou": ciou_batch,  
#     "diou": diou_batch,  
#     "ct_dist": ct_dist,  
# }  


# class D2MPConfig:  
#     def __init__(self):  
#         # 数值稳定项（D2MP_OB 需要）  
#         self.eps = 1e-5  

#         # bbox 表达，Diffmot 训练默认 cxcywh  
#         self.bbox_format = 'cxcywh'  

#         # 对 generate 分支无额外归一化，sample 分支仍可用  
#         self.bbox_mean = [0.0, 0.0, 0.0, 0.0]  
#         self.bbox_std = [1.0, 1.0, 1.0, 1.0]
        
#         # self.bbox_mean = [0.5, 0.5, 0.5, 0.5]  
#         # self.bbox_std = [0.5, 0.5, 0.5, 0.5]      

#         self.context_dim = None  
#         self.hist_len = 5  

#         # 扩散调度与步数（尽量对齐训练）  
#         self.num_steps = 1  
#         self.beta_1 = 1e-4  
#         self.beta_T = 5e-2  
#         self.var_mode = 'linear'  # 若你确认支持 'cosine' 再改  

#         # 模型结构  
#         self.point_dim = 4  
#         self.tf_layer = 3  

#         # 其他开关  
#         self.seed = 12345  
#         self.use_fp16 = False  

#         # 稳健性与融合  
#         self.pred_blend = 0.5  
#         self.pred_gate_iou = 0.25   #0.05
#         self.use_const_v_fallback = True  

#         # generate 路径相关  
#         self.prefer_generate = True  
#         self.gen_residual_scale = 0.6  # 旧字段，仍保留  
#         # 新增：代码里用到的 min/max/限制系数  
#         self.gen_residual_scale_min = 0.35 #0.3  
#         self.gen_residual_scale_max = 1.00  #0.8
#         self.size_change_limit = 0.45  # 0.4 
#         self.cv_mix_alpha = 0.2  

#         # 预留：DDPM 配置（后续你让我启用时再走这条）  
#         self.use_ddpm = True         # True 时尝试走 denoising_dffusion_pytorch  
#         self.ddpm_steps = 10  
#         self.ddpm_eta = 0.0  
#         self.ddpm_objective = 'pred_noise'  # 或 'pred_v'，视你的实现  
#         self.ddpm_beta_schedule = 'cosine'   # 'linear'/'cosine'  
#         self.ddpm_ckpt = None               # 若要加载权重，这里给路径  


# d2mp_config = D2MPConfig()  


# def xyxy_to_cxcywh(xyxy):  
#     x1, y1, x2, y2 = xyxy  
#     w = max(x2 - x1, 1e-6)  
#     h = max(y2 - y1, 1e-6)  
#     cx = x1 + w / 2.0  
#     cy = y1 + h / 2.0  
#     return np.array([cx, cy, w, h], dtype=np.float32)  


# def cxcywh_to_xyxy(cxcywh):  
#     cx, cy, w, h = cxcywh  
#     x1 = cx - w / 2.0  
#     y1 = cy - h / 2.0  
#     x2 = cx + w / 2.0  
#     y2 = cy + h / 2.0  
#     return np.array([x1, y1, x2, y2], dtype=np.float32)  


# def to_unit_xyxy(xyxy, img_size):  
#     H, W = img_size  
#     H, W = _to_int(H), _to_int(W)  
#     x1, y1, x2, y2 = xyxy  
#     return np.array([x1 / W, y1 / H, x2 / W, y2 / H], dtype=np.float32)  


# def to_pixel_xyxy(unit_xyxy, img_size):  
#     H, W = img_size  
#     H, W = _to_int(H), _to_int(W)  
#     x1, y1, x2, y2 = unit_xyxy  
#     return np.array([x1 * W, y1 * H, x2 * W, y2 * H], dtype=np.float32)  


# def clamp_xyxy(xyxy, img_size):  
#     H, W = img_size  
#     H, W = _to_int(H), _to_int(W)  
#     x1, y1, x2, y2 = xyxy  
#     x1 = max(0, min(W - 1, x1))  
#     y1 = max(0, min(H - 1, y1))  
#     x2 = max(0, min(W - 1, x2))  
#     y2 = max(0, min(H - 1, y2))  
#     if x2 < x1:  
#         x2 = x1  
#     if y2 < y1:  
#         y2 = y1  
#     return np.array([x1, y1, x2, y2], dtype=np.float32)  


# _global_d2mp = None  


# def _infer_context_dim_and_hist_len_from_state(state):  
#     # Try to infer context_dim from concat1._hyper_bias.weight in HMINet  
#     ctx_dim = None  
#     for k, v in state.items():  
#         if isinstance(k, str) and k.endswith('concat1._hyper_bias.weight') and hasattr(v, "ndim") and v.ndim == 2:  
#             ctx_dim = int(v.shape[1]) - 3  # - time_emb(3)  
#             break  
#     if ctx_dim is not None and ctx_dim > 0:  
#         d2mp_config.context_dim = ctx_dim  
#         if ctx_dim % 4 == 0:  
#             d2mp_config.hist_len = ctx_dim // 4  


# # 2) 新增：DDPM 包装器（占位，保持接口兼容）  
# class DDPMWrapper(torch.nn.Module):  
#     """  
#     兼容当前调用接口的占位包装器：  
#     - generate(conds_list, img_w, img_h): 返回 N×4 归一化残差（当前占位：全 0）  
#     - sample(context, ...): 返回 B×4 预测（当前占位：回退为 context 的末帧 or 零）  
#     后续你给出明确指令（输入/目标/权重）后，我把这里接到 denoising_dffusion_pytorch 的 GaussianDiffusion/Unet 上。  
#     """  
#     def __init__(self):  
#         super().__init__()  
#         self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
#         # 预留：构建 ddpm.Unet / ddpm.GaussianDiffusion 的代码入口  
#         self._built = False  
#         self._model = None  
#         self._diffusion = None  

#     def build_if_needed(self):  
#         # 预留真实构建逻辑：根据 d2mp_config 构建 unet 与 diffusion，并可加载权重  
#         if self._built:  
#             return  
#         if not _DDPM_AVAILABLE:  
#             print("[D2MP][DDPM] ddpm module not available; using stub outputs.")  
#             self._built = True  
#             return  
#         # TODO: 后续根据你的指令具体实现，例如：  
#         # self._model = ddpm.Unet(dim=64, channels=4, ...)  # 视你的 1D/2D 设计  
#         # self._diffusion = ddpm.GaussianDiffusion(model=self._model, timesteps=..., objective=...)  
#         # if d2mp_config.ddpm_ckpt: load_state_dict(...)  
#         self._built = True  

#     # @torch.inference_mode()  
#     # def generate(self, conds_list, sample=1, bestof=True, img_w=None, img_h=None):  
#     #     """  
#     #     输入：每个 track 的 conds 历史（像素域 [cx,cy,w,h, dx,dy,dw,dh]）  
#     #     输出：N×4 残差（归一化 cxcywh），当前占位为全 0，等价于使用锚点。  
#     #     """  
#     #     self.build_if_needed()  
#     #     N = len(conds_list)  
#     #     zero = np.zeros((N, 4), dtype=np.float32)  
#     #     return zero  # 占位：残差为 0  
    
#     # @torch.inference_mode()
#     # def generate(self, conds_list, sample=1, bestof=True, img_w=None, img_h=None):
#     #     """
#     #     conds_list: 每个track的历史条件，形如 [ [cx,cy,w,h, dx,dy,dw,dh], ... ]（像素域）
#     #     返回: 归一化 residual（cxcywh），与 dets_norm 做加法得到最终 cxcywh 预测
#     #     """
#     #     N = len(conds_list)
#     #     out = np.zeros((N, 4), dtype=np.float32)
#     #     W = float(max(img_w or 1, 1.0))
#     #     H = float(max(img_h or 1, 1.0))

#     #     # 可调：动量/平滑
#     #     alpha = float(getattr(d2mp_config, "cv_alpha", 1.0))  # 1.0=纯用最近一次增量
#     #     scale = float(getattr(d2mp_config, "gen_residual_scale", 0.6))  # 缩放，避免过冲

#     #     for i, conds in enumerate(conds_list):
#     #         arr = np.asarray(conds, dtype=np.float32)
#     #         if arr.ndim == 1:
#     #             arr = arr[None, :]
#     #         # 取最近的位移增量（像素）
#     #         dx, dy, dw, dh = 0.0, 0.0, 0.0, 0.0
#     #         if arr.shape[0] >= 1 and arr.shape[1] >= 8:
#     #             last = arr[-1]
#     #             dx, dy, dw, dh = last[4], last[5], last[6], last[7]
#     #             if arr.shape[0] >= 2:
#     #                 prev = arr[-2]
#     #                 pdx, pdy, pdw, pdh = prev[4], prev[5], prev[6], prev[7]
#     #                 dx = alpha * dx + (1.0 - alpha) * pdx
#     #                 dy = alpha * dy + (1.0 - alpha) * pdy
#     #                 dw = alpha * dw + (1.0 - alpha) * pdw
#     #                 dh = alpha * dh + (1.0 - alpha) * pdh

#     #         # 像素域增量 -> 归一化残差（与 dets_norm 同尺度）
#     #         out[i, 0] = (dx / W) * scale
#     #         out[i, 1] = (dy / H) * scale
#     #         out[i, 2] = (dw / W) * scale
#     #         out[i, 3] = (dh / H) * scale

#     #     return out
    
#     #@torch.inference_mode()
#     def generate(self, conds_list, sample=1, bestof=True, img_w=None, img_h=None):
#         N = len(conds_list)
#         out = np.zeros((N, 4), dtype=np.float32)
#         W = float(max(img_w or 1, 1.0))
#         H = float(max(img_h or 1, 1.0))

#         # 超参（可挂到 d2mp_config）
#         alpha = float(getattr(d2mp_config, "cv_alpha", 0.8))     # 速度 EMA
#         beta  = float(getattr(d2mp_config, "cv_acc_beta", 0.5))  # 加速度权重
#         k_med = int(getattr(d2mp_config, "cv_median_k", 3))      # 鲁棒中位数窗口
#         scale = float(getattr(d2mp_config, "gen_residual_scale", 0.6))

#         for i, conds in enumerate(conds_list):
#             arr = np.asarray(conds, dtype=np.float32)
#             if arr.ndim == 1:
#                 arr = arr[None, :]

#             # 收集最近 k 个增量用于鲁棒估计（像素域）
#             dxs, dys, dws, dhs = [], [], [], []
#             if arr.shape[0] >= 1 and arr.shape[1] >= 8:
#                 for j in range(max(0, arr.shape[0]-k_med), arr.shape[0]):
#                     dxs.append(float(arr[j,4])); dys.append(float(arr[j,5]))
#                     dws.append(float(arr[j,6])); dhs.append(float(arr[j,7]))

#             def robust(x):
#                 if len(x) == 0: return 0.0
#                 return float(np.median(x))

#             vx = robust(dxs); vy = robust(dys); vw = robust(dws); vh = robust(dhs)

#             # 上一时刻增量（用于加速度）
#             pvx = pvy = pvw = pvh = 0.0
#             if arr.shape[0] >= 2 and arr.shape[1] >= 8:
#                 pvx = float(arr[-2,4]); pvy = float(arr[-2,5])
#                 pvw = float(arr[-2,6]); pvh = float(arr[-2,7])

#             # 速度 EMA
#             vx = alpha * vx + (1.0 - alpha) * pvx
#             vy = alpha * vy + (1.0 - alpha) * pvy
#             vw = alpha * vw + (1.0 - alpha) * pvw
#             vh = alpha * vh + (1.0 - alpha) * pvh

#             # 加速度项（a = v_t - v_{t-1}）
#             ax = (vx - pvx); ay = (vy - pvy); aw = (vw - pvw); ah = (vh - pvh)

#             # 二阶外推：v + beta * a
#             dx_hat = vx + beta * ax
#             dy_hat = vy + beta * ay
#             dw_hat = vw + beta * aw
#             dh_hat = vh + beta * ah

#             out[i, 0] = (dx_hat / W) * scale
#             out[i, 1] = (dy_hat / H) * scale
#             out[i, 2] = (dw_hat / W) * scale
#             out[i, 3] = (dh_hat / H) * scale

#         return out

#     @torch.inference_mode()  
#     def sample(self, context, sample=1, bestof=False, point_dim=4, ret_traj=False):  
#         """  
#         输入：flatten 后的 (B, L*4) 上下文（单位域/归一化）  
#         输出：B×4 预测（单位域），当前占位为 context 最后一帧（或 0）  
#         """  
#         self.build_if_needed()  
#         if context is None or context.numel() == 0:  
#             return torch.zeros((1, 4), dtype=torch.float32, device=self._device)  
#         B = context.shape[0]  
#         L = d2mp_config.hist_len  
#         ctx = context.view(B, L, 4)  
#         last = ctx[:, -1, :]  # 取最后一帧作为占位预测  
#         return last  


# def init_d2mp_model():  
#     """  
#     Lazy init global D2MP model with robust checkpoint parsing.  
#     """  
#     global _global_d2mp  
#     if _global_d2mp is not None:  
#         return _global_d2mp  

#     if d2mp_config.seed is not None:  
#         torch.manual_seed(d2mp_config.seed)  
#         np.random.seed(d2mp_config.seed)  

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

#     # 若启用 ddpm，优先返回 DDPMWrapper（占位版本可直接运行）  
#     if getattr(d2mp_config, "use_ddpm", False):  
#         model = DDPMWrapper().to(device)  
#         model.eval()  
#         if getattr(d2mp_config, "use_fp16", False) and device.type == "cuda":  
#             model.half()  
#         print("[D2MP] Using DDPMWrapper (denoising_dffusion_pytorch).")  
#         _global_d2mp = model  
#         return _global_d2mp  

#     # 否则按原来的 D2MP_OB 路径  
#     # Use your absolute path  
#     weight_path = "/public/home/cookie/anaconda3/envs/ocsort_env/OC_SORT/trackers/diffmot_tracker/weights/MOT_epoch800.pt"  

#     # 1) Load raw checkpoint  
#     try:  
#         state_raw = torch.load(weight_path, map_location=device, weights_only=True)  
#     except TypeError:  
#         state_raw = torch.load(weight_path, map_location=device)  

#     # 2) Unwrap common containers  
#     if isinstance(state_raw, dict):  
#         if 'state_dict' in state_raw and isinstance(state_raw['state_dict'], dict):  
#             state_raw = state_raw['state_dict']  
#         elif 'model' in state_raw and isinstance(state_raw['model'], dict):  
#             state_raw = state_raw['model']  
#         elif 'ddpm' in state_raw and isinstance(state_raw['ddpm'], dict):  
#             inner = state_raw['ddpm']  
#             if isinstance(inner, dict) and 'state_dict' in inner and isinstance(inner['state_dict'], dict):  
#                 state_raw = inner['state_dict']  
#             else:  
#                 state_raw = inner  

#     # 3) Strip 'module.' prefix if any  
#     if isinstance(state_raw, dict) and any(isinstance(k, str) and k.startswith('module.') for k in state_raw.keys()):  
#         state_raw = {k.replace('module.', '', 1): v for k, v in state_raw.items()}  

#     # 4) Build flat state dict  
#     state = {}  
#     if isinstance(state_raw, dict):  
#         if 'net' in state_raw and isinstance(state_raw['net'], dict):  
#             for k, v in state_raw['net'].items():  
#                 key = f'net.{k}' if not k.startswith('net.') else k  
#                 state[key] = v  
#         elif any(isinstance(k, str) and k.startswith('net.') for k in state_raw.keys()):  
#             for k, v in state_raw.items():  
#                 if k.startswith('net.'):  
#                     state[k] = v  
#         else:  
#             for k, v in state_raw.items():  
#                 if any(k.startswith(p) for p in ['optimizer', 'epoch', 'ema', 'amp', 'scaler', 'global_step']):  
#                     continue  
#                 if k.startswith('var_sched.'):  
#                     state[k] = v  
#                 else:  
#                     state[f'net.{k}'] = v  

#         if 'var_sched' in state_raw and isinstance(state_raw['var_sched'], dict):  
#             for k, v in state_raw['var_sched'].items():  
#                 key = f'var_sched.{k}' if not k.startswith('var_sched.') else k  
#                 state[key] = v  
#         elif any(isinstance(k, str) and k.startswith('var_sched.') for k in state_raw.keys()):  
#             for k, v in state_raw.items():  
#                 if k.startswith('var_sched.'):  
#                     state[k] = v  

#     # 5) Logging  
#     num_net = sum(1 for k in state.keys() if isinstance(k, str) and k.startswith('net.'))  
#     num_sched = sum(1 for k in state.keys() if isinstance(k, str) and k.startswith('var_sched.'))  
#     if num_net == 0:  
#         print('[D2MP] WARNING: Could not find net weights in checkpoint.')  
#     else:  
#         print(f'[D2MP] Parsed weights: net keys = {num_net}, var_sched keys = {num_sched}')  

#     # 6) Infer context_dim / hist_len  
#     _infer_context_dim_and_hist_len_from_state(state)  
#     context_dim = d2mp_config.context_dim if d2mp_config.context_dim is not None else (d2mp_config.hist_len * 4)  

#     # 7) Build model  
#     var_sched = VarianceSchedule(  
#         num_steps=d2mp_config.num_steps,  
#         mode=d2mp_config.var_mode,  
#         beta_1=d2mp_config.beta_1,  
#         beta_T=d2mp_config.beta_T  
#     )  
#     hminet = HMINet(  
#         point_dim=d2mp_config.point_dim,  
#         context_dim=context_dim,  
#         tf_layer=d2mp_config.tf_layer  
#     )  

#     if not hasattr(d2mp_config, "eps"):  
#         d2mp_config.eps = 1e-5  

#     model = D2MP_OB(  
#         net=hminet,  
#         var_sched=var_sched,  
#         config=d2mp_config  
#     )  

#     # 8) Load weights  
#     try:  
#         model.load_state_dict(state, strict=False)  
#     except Exception as e:  
#         print(f"[D2MP] load_state_dict failed: {e}")  

#     model.eval().to(device)  
#     if getattr(d2mp_config, "use_fp16", False) and device.type == "cuda":  
#         model.half()  

#     print(f"[D2MP] cfg => format={d2mp_config.bbox_format}, hist_len={d2mp_config.hist_len}, "  
#           f"num_steps={d2mp_config.num_steps}, var_mode={d2mp_config.var_mode}, "  
#           f"use_fp16={d2mp_config.use_fp16}")  
#     _global_d2mp = model  
#     return _global_d2mp  


# # -----------------------------  
# # Diffusion-based tracker  
# # -----------------------------  
# class DiffusionBoxTracker(object):  
#     """  
#     D2MP-based tracker replacing Kalman filter.  
#     Maintains same external interface for OC-SORT association.  
#     """  
#     count = 0  

#     def __init__(self, bbox, img_size, delta_t=3):  
#         """  
#         bbox: [x1, y1, x2, y2, score] in pixel coords  
#         img_size: (H, W)  
#         """  
#         self.img_size = img_size  # (H, W)  
#         # OC-SORT counters  
#         self.time_since_update = 0  
#         self.id = DiffusionBoxTracker.count  
#         DiffusionBoxTracker.count += 1  
#         self.hits = 0  
#         self.hit_streak = 0  
#         self.age = 0  

#         # histories  
#         self.history_bboxes = [bbox[:4].astype(np.float32)]  
#         self.last_bbox = bbox[:4].astype(np.float32)  
#         self.last_observation = np.array([-1, -1, -1, -1, -1], dtype=np.float32)  
#         self.observations = dict()  
#         self.history_observations = []  
#         self.velocity = None  
#         self.delta_t = delta_t  

#         # Diffmot 兼容：维护 conds（像素域）  
#         self.prev_accept_xywh = None  
#         self.diff_conds = collections.deque([], maxlen=5)  

#         # 初始化立即作为一次观测  
#         self.update(bbox, img_size=img_size)
        
#         # 用于“预测质量”的 EMA 与本帧预测缓存
#         self.pred_iou_ema = 0.5
#         self.pending_pred = None  # or pending_pred_tlbr，统一用 pending_pred 保存像素 xyxy  

#     def set_img_size(self, img_size):  
#         H, W = img_size  
#         H, W = _to_int(H), _to_int(W)  
#         self.img_size = (H, W)  

#     # normalization helpers for sample fallback  
#     def _normalize_tensor(self, bbox_tensor):  
#         mean = torch.tensor(d2mp_config.bbox_mean, device=bbox_tensor.device, dtype=bbox_tensor.dtype)  
#         std = torch.tensor(d2mp_config.bbox_std, device=bbox_tensor.device, dtype=bbox_tensor.dtype)  
#         return (bbox_tensor - mean) / std  

#     def _denormalize_tensor(self, norm_tensor):  
#         mean = torch.tensor(d2mp_config.bbox_mean, device=norm_tensor.device, dtype=norm_tensor.dtype)  
#         std = torch.tensor(d2mp_config.bbox_std, device=norm_tensor.device, dtype=norm_tensor.dtype)  
#         return norm_tensor * std + mean  

#     def update(self, bbox, img_size=None):  
#         """  
#         Update with detection bbox (x1,y1,x2,y2,score) or None.  
#         """  
#         if img_size is not None:  
#             self.set_img_size(img_size)  

#         if bbox is not None:  
#             bbox = bbox.astype(np.float32)  
#             # velocity (direction) using observation delta_t steps away  
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

#             # maintain obs history like original OC-SORT  
#             self.last_observation = bbox.copy()  
#             self.observations[self.age] = bbox.copy()  
#             self.history_observations.append(bbox.copy())  

#             # maintain bbox history for D2MP (store xyxy)  
#             self.last_bbox = bbox[:4].copy()  
#             self.history_bboxes.append(self.last_bbox.copy())  
#             if len(self.history_bboxes) > d2mp_config.hist_len:  
#                 self.history_bboxes = self.history_bboxes[-d2mp_config.hist_len:]  

#             # Diffmot conds: [xywh, delta_xywh] in pixels  
#             x1, y1, x2, y2 = self.last_bbox  
#             cx = 0.5 * (x1 + x2)  
#             cy = 0.5 * (y1 + y2)  
#             w = max(x2 - x1, 1.0)  
#             h = max(y2 - y1, 1.0)  
#             xywh = np.array([cx, cy, w, h], dtype=np.float32)  
#             if self.prev_accept_xywh is None:  
#                 delta = np.zeros(4, dtype=np.float32)  
#             else:  
#                 delta = (xywh - self.prev_accept_xywh).astype(np.float32)  
#             tmp_conds = np.concatenate([xywh, delta]).astype(np.float32)  
#             self.diff_conds.append(tmp_conds)  
#             self.prev_accept_xywh = xywh.copy()  

#             # OC-SORT counters  
#             self.time_since_update = 0  
#             self.hits += 1  
#             self.hit_streak += 1  
#         else:  
#             # no observation this frame  
#             self.time_since_update += 1  

#     def predict(self):  
#         """  
#         Predict next bbox with D2MP (single-track fallback).  
#         主流程使用 batched 版本；public 模式等仍会用到该方法。  
#         Returns shape (1,4) in pixel coords.  
#         """  
#         model = init_d2mp_model()  
#         device = next(model.parameters()).device  

#         # sample 分支（保持原逻辑）  
#         H, W = self.img_size  
#         L = d2mp_config.hist_len  
#         hist = self.history_bboxes[-L:]  
#         if len(hist) < L:  
#             hist = hist + [hist[-1]] * (L - len(hist))  

#         hist_unit = [to_unit_xyxy(b, (H, W)) for b in hist]  
#         hist_unit = np.stack(hist_unit, axis=0)  # (L,4)  

#         if d2mp_config.bbox_format == 'cxcywh':  # 原cxcywh  
#             hist_unit = np.stack([xyxy_to_cxcywh(b) for b in hist_unit], axis=0)  

#         inp = torch.tensor(hist_unit, dtype=torch.float32, device=device)  # (L,4)  
#         inp = self._normalize_tensor(inp)  
#         context = inp.flatten().unsqueeze(0)  # (1, L*4)  

#         with torch.inference_mode():  
#             pred = model.sample(  
#                 context=context,  
#                 sample=1,  
#                 bestof=False,  
#                 point_dim=d2mp_config.point_dim,  
#                 ret_traj=False  
#             )  

#         if isinstance(pred, (list, tuple)):  
#             pred = pred[0]  
#         pred = pred[0, 0] if hasattr(pred, "ndim") and pred.ndim == 3 else pred.squeeze(0)  # (4,)  
#         pred = self._denormalize_tensor(pred)  

#         pred_unit = pred.detach().cpu().numpy()  
#         if d2mp_config.bbox_format == 'cxcywh':  # cxcywh  
#             pred_unit = cxcywh_to_xyxy(pred_unit)  

#         pred_px = to_pixel_xyxy(pred_unit, (H, W))  
#         pred_px = clamp_xyxy(pred_px, (H, W))  

#         # advance counters  
#         self.age += 1  
#         if self.time_since_update > 0:  
#             self.hit_streak = 0  
#         self.time_since_update += 1  

#         self.last_bbox = pred_px.astype(np.float32)  
#         return self.last_bbox.reshape(1, 4)  

#     def get_state(self):  
#         return self.last_bbox.reshape(1, 4)  


# # -----------------------------  
# # OCSort using DiffusionBoxTracker  
# # -----------------------------  
# class Diff_OCSort(object):  
#     def __init__(self, det_thresh, max_age=30, min_hits=3,  
#                  iou_threshold=0.3, delta_t=3, asso_func="iou",  
#                  inertia=0.2, use_byte=False, use_gmc=True,  
#                  gmc_method='sparseOptFlow', gmc_downscale=2):  
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

#         DiffusionBoxTracker.count = 0  
#         self.last_img_size = None  # (H,W)  

#         # 诊断用  
#         self._dbg_pred_cnt = 0  
#         self._dbg_iou_running = []  

#         # 跨序列检测状态  
#         self._prev_frame_id = -1  # 如果外层传 frame_id，可用于回绕检测  

#         self.use_gmc = use_gmc  
#         self.gmc = GMC(method=gmc_method, downscale=gmc_downscale) if use_gmc else None  
#         self._last_H_prev2curr = None  # 仅用于调试或备用  

#     # --- 新增：跨序列 reset ---  
#     def reset(self):  
#         """  
#         清空所有与单序列相关的状态，准备下一个视频序列。  
#         """  
#         # 清空跟踪器与计数  
#         self.trackers = []  
#         self.frame_count = 0  
#         self.last_img_size = None  

#         # 重置 Track ID 计数器  
#         DiffusionBoxTracker.count = 0  

#         # 清空诊断  
#         self._dbg_pred_cnt = 0  
#         self._dbg_iou_running = []  
#         if hasattr(self, "_pred_shape_log"):  
#             self._pred_shape_log = 0  

#         # 跨序列检测状态  
#         self._prev_frame_id = -1  

#         if self.gmc is not None:  
#             self.gmc.initializedFirstFrame = False  
#         self._last_H_prev2curr = None  

#     def _maybe_reset_on_boundary(self, frame_id=None, img_size=None):  
#         """  
#         如果检测到“frame_id 回绕”或“分辨率变化”，自动 reset。  
#         - frame_id: 外层若能传，优先用它判断回绕（如从 300 -> 1）。  
#         - img_size: (H,W)，不同序列常有不同分辨率，也作为触发条件。  
#         """  
#         fid_back = (self._prev_frame_id != -1 and frame_id is not None and frame_id <= self._prev_frame_id)  
#         size_changed = (img_size is not None and self.last_img_size is not None  
#                         and tuple(img_size) != tuple(self.last_img_size))  
#         if fid_back or size_changed:  
#             print(f"[D2MP] new sequence detected (fid_back={fid_back}, size_changed={size_changed}), resetting tracker")  
#             self.reset()  
#         # 更新最近状态（即使 reset 了也要记录当前信息）  
#         if frame_id is not None:  
#             self._prev_frame_id = int(frame_id)  
#         if img_size is not None:  
#             self.last_img_size = tuple(img_size)  

#     def _iou_single(self, a, b):  
#         x1 = max(a[0], b[0]); y1 = max(a[1], b[1])  
#         x2 = min(a[2], b[2]); y2 = min(a[3], b[3])  
#         iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)  
#         inter = iw * ih  
#         area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))  
#         area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))  
#         union = area_a + area_b - inter + 1e-6  
#         return inter / union  

#     # def _batched_predict(self, trackers, img_size):  
#     def _batched_predict(self, trackers, img_size, H_prev2curr=None):  
#         """  
#         - 如模型具备 generate(): 走 Diffmot 语义（conds 像素域，输出偏移）  
#         - 否则：回退到 sample(context)（unit 域绝对框）  
#         返回 (N,4) xyxy 像素坐标，并同步计数器和 last_bbox。  
#         """  
#         if len(trackers) == 0:  
#             return np.zeros((0, 4), dtype=np.float32)  

#         model = init_d2mp_model()  
#         # --- fix: safe get device ---
#         try:
#             device = next(model.parameters()).device
#         except StopIteration:
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # ---
#         H, W = img_size  
#         if hasattr(model, "generate") and getattr(d2mp_config, "prefer_generate", True):  
#             dets_norm = []  
#             conds_list = []  
#             metas = []  
#             gmc_anchors_px = []  

#             for trk in trackers:  
#                 trk.set_img_size((H, W))  

#                 # 取上一帧的参考框（优先观测，否则last_bbox）  
#                 prev_box = trk.last_observation[:4] if trk.last_observation.sum() > 0 else trk.last_bbox  
#                 # 用 GMC 仿射映射到当前帧，得到稳定锚  
#                 gmc_box = warp_box_affine_xyxy(H_prev2curr, prev_box)  
#                 gmc_box = clamp_xyxy(gmc_box, (H, W))  
#                 gmc_anchors_px.append(gmc_box)  

#                 # 转 cxcywh（像素）  
#                 x1, y1, x2, y2 = gmc_box  
#                 cx = 0.5 * (x1 + x2); cy = 0.5 * (y1 + y2)  
#                 w = max(1.0, x2 - x1); h = max(1.0, y2 - y1)  

#                 # 归一化锚，作为 generate 的基线  
#                 dets_norm.append([cx / W, cy / H, w / W, h / H])  

#                 # 条件（像素域）  
#                 if not hasattr(trk, "diff_conds") or len(trk.diff_conds) == 0:  
#                     delta = np.zeros(4, dtype=np.float32)  
#                     cond = np.array([cx, cy, w, h, *delta], dtype=np.float32)  
#                     conds = [cond]  
#                 else:  
#                     # 对齐长度到 hist_len（右对齐，左侧 pad）  
#                     L = int(getattr(d2mp_config, "hist_len", 5))  
#                     tmp = list(trk.diff_conds)  
#                     if len(tmp) < L:  
#                         tmp = [tmp[0]] * (L - len(tmp)) + tmp  
#                     elif len(tmp) > L:  
#                         tmp = tmp[-L:]  
#                     conds = tmp  
#                 conds_list.append(conds)  
#                 metas.append(trk)  

#             dets_norm = np.asarray(dets_norm, dtype=np.float32)  

#             with torch.inference_mode():  
#                 multi_track_pred = model.generate(conds_list, sample=1, bestof=True, img_w=W, img_h=H)  
#             if isinstance(multi_track_pred, (list, tuple)):  
#                 multi_track_pred = multi_track_pred[0]  
#             pred = np.asarray(multi_track_pred)  
#             if pred.ndim == 3:  
#                 pred = pred.mean(0)  # (N,4) 残差，归一化 cxcywh  

#             # 动态残差缩放 + 尺度夹紧  
#             rs_min = float(getattr(d2mp_config, "gen_residual_scale_min", 0.3))  
#             rs_max = float(getattr(d2mp_config, "gen_residual_scale_max", 0.8))  
#             scales = np.array([rs_min + (rs_max - rs_min) * float(getattr(trk, "pred_iou_ema", 0.6))  
#                                for trk in metas], dtype=np.float32).reshape(-1, 1)  
#             pred = pred * scales  

#             track_pred = pred + dets_norm  # 合成：归一化 cxcywh  
#             ratio = float(getattr(d2mp_config, "size_change_limit", 0.4))  
#             if ratio > 0:  
#                 w0 = dets_norm[:, 2:3]; h0 = dets_norm[:, 3:4]  
#                 track_pred[:, 2:3] = np.clip(track_pred[:, 2:3], w0 * (1 - ratio), w0 * (1 + ratio))  
#                 track_pred[:, 3:4] = np.clip(track_pred[:, 3:4], h0 * (1 - ratio), h0 * (1 + ratio))  

#             # 还原到像素 xyxy  
#             tp = track_pred.copy()  
#             tp[:, 0::2] *= float(W); tp[:, 1::2] *= float(H)  
#             x_tl = tp[:, 0] - tp[:, 2] / 2.0; y_tl = tp[:, 1] - tp[:, 3] / 2.0  
#             preds_px = np.stack([x_tl, y_tl, x_tl + tp[:, 2], y_tl + tp[:, 3]], axis=1).astype(np.float32)  
#             preds_px[:, 0] = np.clip(preds_px[:, 0], 0, W - 1)  
#             preds_px[:, 1] = np.clip(preds_px[:, 1], 0, H - 1)  
#             preds_px[:, 2] = np.clip(preds_px[:, 2], 0, W - 1)  
#             preds_px[:, 3] = np.clip(preds_px[:, 3], 0, H - 1)  

#             # 保存“原始预测”供 EMA 更新  
#             for trk, px in zip(metas, preds_px):  
#                 trk.pending_pred = px.copy()  

#             # # 门控 + 与 GMC 锚的微融合（更连续）  
#             # gate_iou = float(getattr(d2mp_config, "pred_gate_iou", 0.30))  
#             # cv_mix_alpha = float(getattr(d2mp_config, "cv_mix_alpha", 0.2))  
#             # for i, (trk, px_pred, gmc_px) in enumerate(zip(metas, preds_px, gmc_anchors_px)):  
#             #     fused = px_pred  
#             #     if trk.last_observation.sum() > 0:  
#             #         last = trk.last_observation[:4]  
#             #         if self._iou_single(px_pred, last) < gate_iou:  
#             #             fused = gmc_px.astype(np.float32)  # 退回 GMC  
#             #         else:  
#             #             fused = (1.0 - cv_mix_alpha) * gmc_px + cv_mix_alpha * px_pred  
#             #     preds_px[i] = clamp_xyxy(fused.astype(np.float32), (H, W))  

#         else:  
#             # 回退到 sample(context) 路径  
#             L = d2mp_config.hist_len  
#             contexts = []  
#             metas = []  
#             for trk in trackers:  
#                 trk.set_img_size((H, W))  
#                 hist = trk.history_bboxes[-L:]  
#                 if len(hist) < L:  
#                     hist = hist + [hist[-1]] * (L - len(hist))  
#                 hist_unit = [to_unit_xyxy(b, (H, W)) for b in hist]  
#                 hist_unit = np.stack(hist_unit, axis=0)  # (L,4)  
#                 if d2mp_config.bbox_format == 'cxcywh':  # cxcywh  
#                     hist_unit = np.stack([xyxy_to_cxcywh(b) for b in hist_unit], axis=0)  
#                 contexts.append(hist_unit)  
#                 metas.append(trk)  

#             contexts = torch.tensor(np.stack(contexts, axis=0), dtype=torch.float32, device=device)  # (B,L,4)  
#             mean = torch.tensor(d2mp_config.bbox_mean, device=device, dtype=contexts.dtype).view(1, 1, 4)  
#             std = torch.tensor(d2mp_config.bbox_std, device=device, dtype=contexts.dtype).view(1, 1, 4)  
#             contexts = (contexts - mean) / std  
#             contexts = contexts.flatten(start_dim=1)  

#             if getattr(d2mp_config, "use_fp16", False) and device.type == "cuda":  
#                 contexts = contexts.half()  

#             with torch.inference_mode():  
#                 pred = model.sample(  
#                     context=contexts,  
#                     sample=1,  
#                     bestof=False,  
#                     point_dim=d2mp_config.point_dim,  
#                     ret_traj=False  
#                 )  

#             # [INSERT POINT 1] ---- 形状诊断 + 统一到 (B,4) ----  
#             if not hasattr(self, "_pred_shape_log"):  
#                 self._pred_shape_log = 0  
#             if self._pred_shape_log < 5:  
#                 try:  
#                     print("[D2MP][sample] raw pred type:", type(pred),  
#                           "shape:", tuple(pred.shape) if hasattr(pred, "shape") else None)  
#                 except Exception as e:  
#                     print("[D2MP][sample] shape print err:", e)  
#                 self._pred_shape_log += 1  

#             # 统一 pred 形状为 (B,4)  
#             if isinstance(pred, (list, tuple)):  
#                 pred = pred[0]  
#             if hasattr(pred, "ndim") and pred.ndim == 3:  
#                 S, B_ = pred.shape[:2]  
#                 if S == 1 and B_ == len(trackers):  
#                     pred = pred[0]        # (B,4)  
#                 elif B_ == 1 and pred.shape[0] == len(trackers):  
#                     pred = pred[:, 0]     # (B,4)  
#                 else:  
#                     print(f"[D2MP][warn] unexpected pred shape {tuple(pred.shape)}; expect (1,B,4) or (B,1,4), B={len(trackers)}")  
#                     if pred.shape[-1] == 4:  
#                         pred = pred.reshape(-1, 4)  # 粗暴兜底  
#                     else:  
#                         raise RuntimeError(f"Unrecognized pred shape: {tuple(pred.shape)}")  

#             if hasattr(pred, "dtype") and pred.dtype != contexts.dtype:  
#                 pred = pred.to(contexts.dtype)  
#             pred = pred.view(len(trackers), 4)  # 现在应为 (B,4)  
#             # [INSERT POINT 1 END]  

#             # 2) 反归一化回“单位域”  
#             pred = pred * std.view(1, 4) + mean.view(1, 4)  # ← 单位域  

#             pred_np = pred.detach().float().cpu().numpy()  

#             # 3) bbox_format 转换  
#             if d2mp_config.bbox_format == 'cxcywh':  
#                 pred_np = np.stack([cxcywh_to_xyxy(p) for p in pred_np], axis=0)  
#             elif d2mp_config.bbox_format == 'xyxy':  
#                 pass  
#             else:  
#                 print(f"[D2MP][warn] unexpected bbox_format: {d2mp_config.bbox_format}")  

#             preds_px = np.stack([to_pixel_xyxy(p, (H, W)) for p in pred_np], axis=0)  
#             preds_px = np.stack([clamp_xyxy(p, (H, W)) for p in preds_px], axis=0).astype(np.float32)  
            
#             # 在 sample 路径 preds_px 计算完后，补充：
#             for trk, px in zip(metas, preds_px):
#                 trk.pending_pred = px.copy()

#         # # === 用 GMC 锚做微融合/回退（两条路径统一处理） ===  
#         # gate_iou = float(getattr(d2mp_config, "pred_gate_iou", 0.30))  
#         # cv_mix_alpha = float(getattr(d2mp_config, "cv_mix_alpha", 0.2))  
#         # metas = trackers if 'metas' not in locals() else metas  
#         # for i, (trk, px_pred) in enumerate(zip(metas, preds_px)):  
#         #     prev_box = trk.last_observation[:4] if trk.last_observation.sum() > 0 else trk.last_bbox  
#         #     gmc_px = warp_box_affine_xyxy(H_prev2curr, prev_box)  
#         #     gmc_px = clamp_xyxy(gmc_px, (H, W))  
#         #     fused = px_pred  
#         #     if trk.last_observation.sum() > 0:  
#         #         if self._iou_single(px_pred, trk.last_observation[:4]) < gate_iou:  
#         #             fused = gmc_px  
#         #         else:  
#         #             fused = (1.0 - cv_mix_alpha) * gmc_px + cv_mix_alpha * px_pred  
#         #     preds_px[i] = clamp_xyxy(fused.astype(np.float32), (H, W))  
        
#                 # === 统一候选选择 + 自适应融合（替换原来的统一融合块） ===
#         gate_iou = float(getattr(d2mp_config, "pred_gate_iou", 0.30))
#         base = 0.05   # 融合对预测的最低权重 0.1
#         gain = 0.9   # 可被 pred_iou_ema 提升的权重幅度 0.7

#         def _build_cv_candidate_from_gmc(gmc_px, trk):
#             # 从 GMC 锚出发，加上最近一次 conds 的像素域增量，构成“恒速外推”候选
#             dx=dy=dw=dh=0.0
#             if hasattr(trk, "diff_conds") and len(trk.diff_conds) > 0:
#                 v = trk.diff_conds[-1]
#                 if len(v) >= 8:
#                     dx, dy, dw, dh = float(v[4]), float(v[5]), float(v[6]), float(v[7])
#             x1, y1, x2, y2 = gmc_px
#             cx = 0.5*(x1+x2) + dx
#             cy = 0.5*(y1+y2) + dy
#             w  = max(1.0, (x2-x1) + dw)
#             h  = max(1.0, (y2-y1) + dh)
#             return clamp_xyxy(np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dtype=np.float32), (H,W))

#         # def _score_candidate(cand_xyxy, gmc_xyxy, last_obs_xyxy, trk):
#         #     # 评分：与 GMC 一致性 + 速度方向一致性 - 尺寸突变
#         #     iou_gmc = self._iou_single(cand_xyxy, gmc_xyxy)

#         #     c_prev = np.array([(last_obs_xyxy[0]+last_obs_xyxy[2])/2.0, (last_obs_xyxy[1]+last_obs_xyxy[3])/2.0], dtype=np.float32)
#         #     c_cand = np.array([(cand_xyxy[0]+cand_xyxy[2])/2.0, (cand_xyxy[1]+cand_xyxy[3])/2.0], dtype=np.float32)
#         #     disp = c_cand - c_prev
#         #     v = trk.velocity if trk.velocity is not None else np.array([0.0, 0.0], dtype=np.float32)
#         #     align = 0.0
#         #     if np.linalg.norm(disp) > 1e-6 and np.linalg.norm(v) > 1e-6:
#         #         align = float(np.dot(disp/np.linalg.norm(disp), v/np.linalg.norm(v)))

#         #     w0 = max(1.0, last_obs_xyxy[2]-last_obs_xyxy[0]); h0 = max(1.0, last_obs_xyxy[3]-last_obs_xyxy[1])
#         #     w1 = max(1.0, cand_xyxy[2]-cand_xyxy[0]);       h1 = max(1.0, cand_xyxy[3]-cand_xyxy[1])
#         #     size_change = abs(w1-w0)/w0 + abs(h1-h0)/h0

#         #     # 线性组合（可微调权重）
#         #     return (0.6 * iou_gmc) + (0.4 * align) - (0.2 * size_change) 
        
#                 # 动态门控：EMA 越高，gate 越低（越敢偏离上一帧）
#         gate_base = float(getattr(d2mp_config, "gate_base", 0.35))
#         gate_gain = float(getattr(d2mp_config, "gate_gain", 0.25))

#         def _score_candidate(cand_xyxy, gmc_xyxy, last_obs_xyxy, trk):
#             # 评分：与 GMC 一致性(较低权重) + 速度方向一致性(更高权重) - 尺寸突变 - 方向反转惩罚
#             iou_gmc = self._iou_single(cand_xyxy, gmc_xyxy)

#             c_prev = np.array([(last_obs_xyxy[0]+last_obs_xyxy[2])/2.0, (last_obs_xyxy[1]+last_obs_xyxy[3])/2.0], dtype=np.float32)
#             c_cand = np.array([(cand_xyxy[0]+cand_xyxy[2])/2.0, (cand_xyxy[1]+cand_xyxy[3])/2.0], dtype=np.float32)
#             disp = c_cand - c_prev
#             v = trk.velocity if trk.velocity is not None else np.array([0.0, 0.0], dtype=np.float32)

#             align = 0.0
#             rev_penalty = 0.0
#             if np.linalg.norm(disp) > 1e-6 and np.linalg.norm(v) > 1e-6:
#                 vd = disp/np.linalg.norm(disp); vv = v/np.linalg.norm(v)
#                 align = float(np.dot(vd, vv))  # [-1,1]
#                 # 方向反转强惩罚（当 align < -0.5）
#                 if align < -0.5:
#                     rev_penalty = 0.3 * ( -align - 0.5 )  # 最大 ~0.45

#             w0 = max(1.0, last_obs_xyxy[2]-last_obs_xyxy[0]); h0 = max(1.0, last_obs_xyxy[3]-last_obs_xyxy[1])
#             w1 = max(1.0, cand_xyxy[2]-cand_xyxy[0]);       h1 = max(1.0, cand_xyxy[3]-cand_xyxy[1])
#             size_change = abs(w1-w0)/w0 + abs(h1-h0)/h0

#             # 更信速度一致性：0.5（align） + 0.35（GMC） - 0.2（尺寸） - 反转惩罚
#             return (0.35 * iou_gmc) + (0.50 * align) - (0.20 * size_change) - rev_penalty

#         metas = trackers if 'metas' not in locals() else metas
#         for i, (trk, px_pred) in enumerate(zip(metas, preds_px)):
#             prev_box = trk.last_observation[:4] if trk.last_observation.sum() > 0 else trk.last_bbox
#             gmc_px = warp_box_affine_xyxy(H_prev2curr, prev_box)
#             gmc_px = clamp_xyxy(gmc_px, (H, W))
#             cv_px  = _build_cv_candidate_from_gmc(gmc_px, trk)

#             # 3 候选：模型预测、GMC 锚、恒速外推
#             candidates = [px_pred, gmc_px, cv_px]
#             scores = [ _score_candidate(c, gmc_px, prev_box, trk) for c in candidates ]
#             best_idx = int(np.argmax(scores))
#             best = candidates[best_idx]

#             # # 自适应融合：预测质量越高（pred_iou_ema 越高），越信“best”
#             # alpha = float(np.clip(base + gain * float(getattr(trk, "pred_iou_ema", 0.5)), base, base + gain))

#             # # 保护性门控：如果 best 跟上一帧观测差太多，先回退 GMC
#             # if trk.last_observation.sum() > 0 and self._iou_single(best, trk.last_observation[:4]) < gate_iou:
#             #     fused = gmc_px
#             # else:
#             #     fused = (1.0 - alpha) * gmc_px + alpha * best
            
#             ema = float(getattr(trk, "pred_iou_ema", 0.5))
#             alpha = float(np.clip(base + gain * ema, base, base + gain))
#             gate_iou = max(0.05, gate_base - gate_gain * ema)  # EMA 高 -> gate 更低

#             if trk.last_observation.sum() > 0 and self._iou_single(best, trk.last_observation[:4]) < gate_iou:
#                 fused = gmc_px
#             else:
#                 fused = (1.0 - alpha) * gmc_px + alpha * best            

#             preds_px[i] = clamp_xyxy(fused.astype(np.float32), (H, W))

#         # 诊断：预测 vs 最近观测的 IoU（注意：这是 pred vs obs，并非 vs last）  
#         ious = []  
#         for trk, px in zip(metas, preds_px):  
#             if trk.last_observation.sum() > 0:  
#                 ious.append(self._iou_single(px, trk.last_observation[:4]))  
#         if len(ious) > 0 and self._dbg_pred_cnt < 200:  
#             self._dbg_pred_cnt += 1  
#             self._dbg_iou_running.append(float(np.median(ious)))  
#             if self._dbg_pred_cnt % 20 == 0:  
#                 print(f"[D2MP][diag] median IoU(pred vs obs) over last {len(self._dbg_iou_running)} steps: "  
#                       f"{np.median(self._dbg_iou_running):.3f}")  

#         # 同步计数器  
#         for trk, px in zip(metas, preds_px):  
#             trk.age += 1  
#             if trk.time_since_update > 0:  
#                 trk.hit_streak = 0  
#             trk.time_since_update += 1  
#             trk.last_bbox = px.astype(np.float32)  

#         return preds_px  



#     def update(self, output_results, img_info, img_size, frame_id=None, raw_frame=None):  
#         """  
#         Params:  
#           output_results: numpy array of detections [[x1,y1,x2,y2,score], ...]  
#           img_info: (img_h, img_w, ...)  
#           img_size: (test_input_h, test_input_w) - for scaling back  
#           frame_id: Optional[int] - 若可传入当前帧号，可用于检测跨序列（从大到小回绕，例如跳回 1）  
#         """  
#         if output_results is None:  
#             return np.empty((0, 5))  

#         # --- 自动跨序列检测（务必在构造 contexts/预测前调用）---  
#         img_h, img_w = _to_int(img_info[0]), _to_int(img_info[1])  
#         self._maybe_reset_on_boundary(frame_id=frame_id, img_size=(img_h, img_w))  

#         # 正常帧计数  
#         self.frame_count += 1  

#         if output_results.shape[1] == 5:  
#             scores = output_results[:, 4]  
#             bboxes = output_results[:, :4]  
#         else:  
#             output_results = output_results.cpu().numpy()  
#             scores = output_results[:, 4] * output_results[:, 5]  
#             bboxes = output_results[:, :4]  

#         # 记录当前分辨率（_maybe_reset_on_boundary 已经记录，这里覆盖为同值无碍）  
#         self.last_img_size = (img_h, img_w)  
#         scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))  

#         bboxes = bboxes / scale  
#         dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)  

#         inds_low = scores > 0.1  
#         inds_high = scores < self.det_thresh  
#         inds_second = np.logical_and(inds_low, inds_high)  
#         dets_second = dets[inds_second]  
#         remain_inds = scores > self.det_thresh  
#         dets = dets[remain_inds]  

#         # === GMC: 估计 prev->curr 的 2x3 仿射 H，并保存 ===  
#         H_prev2curr = None  
#         if self.use_gmc and raw_frame is not None and self.gmc is not None:  
#             try:  
#                 # 可先尝试 detections=None，避免特征点不足；当前保持与你代码一致  
#                 H_prev2curr = self.gmc.apply(raw_frame=raw_frame, detections=dets[:, :4].astype(np.int32))  
#             except Exception as e:  
#                 print("[GMC] failed:", e)  
#                 H_prev2curr = None  
#         self._last_H_prev2curr = H_prev2curr  

#         # 批量扩散预测  
#         trks = np.zeros((len(self.trackers), 5), dtype=np.float32)  
#         to_del = []  
#         ret = []  
#         if len(self.trackers) > 0:  
#             preds_px = self._batched_predict(self.trackers, img_size=(img_h, img_w), H_prev2curr=H_prev2curr)  
#             for t in range(len(self.trackers)):  
#                 pos = preds_px[t]  
#                 trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]  
#                 if np.any(np.isnan(pos)):  
#                     to_del.append(t)  

#         trks = np.ma.compress_rows(np.ma.masked_invalid(trks))  
#         for t in reversed(to_del):  
#             self.trackers.pop(t)  

#         velocities = np.array(  
#             [trk.velocity if trk.velocity is not None else np.array((0, 0), dtype=np.float32) for trk in self.trackers])  
#         last_boxes = np.array([trk.last_observation for trk in self.trackers])  
#         k_observations = np.array(  
#             [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])  

#         # # First round of association  
#         # matched, unmatched_dets, unmatched_trks = associate(  
#         #     dets, trks, self.iou_threshold, velocities, k_observations, self.inertia)  
#         # for m in matched:  
#         #     self.trackers[m[1]].update(dets[m[0], :], img_size=(img_h, img_w))  
        
        
#         # # ious_pred_vs_det = []
#         # # for m in matched:
#         # #     trk = self.trackers[m[1]]
#         # #     det_box = dets[m[0], :4]
#         # #     if hasattr(trk, "pending_pred"):
#         # #         iou_pd = self._iou_single(trk.pending_pred, det_box)
#         # #         ious_pred_vs_det.append(float(iou_pd))
#         # #     trk.update(dets[m[0], :], img_size=(img_h, img_w))

#         # # if len(ious_pred_vs_det) > 0:
#         # #     # 打印或滑动统计
#         # #     print(f"[D2MP][diag] IoU(pred vs det) median: {np.median(ious_pred_vs_det):.3f}")

#         # First round of association
#         matched, unmatched_dets, unmatched_trks = associate(
#             dets, trks, self.iou_threshold, velocities, k_observations, self.inertia
#         )

#         # ious_pred_vs_det = []
#         # for m in matched:
#         #     trk = self.trackers[m[1]]
#         #     det_box = dets[m[0], :4]

#         #     # 用 pending_pred（在 _batched_predict 两条路径都已写入）评估质量
#         #     if hasattr(trk, "pending_pred") and trk.pending_pred is not None:
#         #         pred_box = trk.pending_pred
#         #         # IoU
#         #         x1 = max(pred_box[0], det_box[0]); y1 = max(pred_box[1], det_box[1])
#         #         x2 = min(pred_box[2], det_box[2]); y2 = min(pred_box[3], det_box[3])
#         #         iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
#         #         inter = iw * ih
#         #         ap = max(0.0, (pred_box[2]-pred_box[0])*(pred_box[3]-pred_box[1]))
#         #         ad = max(0.0, (det_box[2]-det_box[0])*(det_box[3]-det_box[1]))
#         #         union = ap + ad - inter + 1e-6
#         #         iou_pd = inter / union

#         #         # EMA 更新（平滑因子可微调 0.9）
#         #         trk.pred_iou_ema = 0.9 * float(getattr(trk, "pred_iou_ema", 0.5)) + 0.1 * float(iou_pd)
#         #         ious_pred_vs_det.append(float(iou_pd))

#         #     # 原本更新轨迹
#         #     self.trackers[m[1]].update(dets[m[0], :], img_size=(img_h, img_w))
        
        
#         # ===== [PATCH-STEP3] PREDICT-AWARE ASSOCIATION (replace the original associate call) =====
#         # 1) 收集每条轨的预测框（pending_pred）；没有则回退到上一观测/最后框
#         active_ids, trk_boxes, row_ema = [], [], []
#         for gidx, trk in enumerate(self.trackers):
#             # 如果你有状态过滤，比如只取活跃轨迹，这里加判断
#             # if trk.state == TrackState.Deleted: continue
#             if hasattr(trk, "pending_pred") and trk.pending_pred is not None:
#                 trk_boxes.append(trk.pending_pred)
#             else:
#                 prev_box = trk.last_observation[:4] if getattr(trk, "last_observation", None) is not None and trk.last_observation.sum() > 0 else trk.last_bbox
#                 trk_boxes.append(prev_box)
#             active_ids.append(gidx)
#             row_ema.append(float(getattr(trk, "pred_iou_ema", 0.5)))

#         if len(trk_boxes) == 0 or dets.shape[0] == 0:
#             matched = np.empty((0, 2), dtype=int)
#             unmatched_trks = np.arange(len(active_ids), dtype=int)
#             unmatched_dets = np.arange(dets.shape[0], dtype=int)
#         else:
#             trk_boxes = np.asarray(trk_boxes, dtype=np.float32)   # (T,4) xyxy
#             det_boxes = dets[:, :4].astype(np.float32)            # (D,4) xyxy

#             # 2) IoU 矩阵（预测 vs 检测）
#             def iou_mat(A, B):
#                 N, M = A.shape[0], B.shape[0]
#                 ious = np.zeros((N, M), dtype=np.float32)
#                 for i in range(N):
#                     x1 = np.maximum(A[i,0], B[:,0]); y1 = np.maximum(A[i,1], B[:,1])
#                     x2 = np.minimum(A[i,2], B[:,2]); y2 = np.minimum(A[i,3], B[:,3])
#                     iw = np.clip(x2 - x1, 0.0, None); ih = np.clip(y2 - y1, 0.0, None)
#                     inter = iw * ih
#                     a = max(1e-6, (A[i,2]-A[i,0])*(A[i,3]-A[i,1]))
#                     b = np.maximum(1e-6, (B[:,2]-B[:,0])*(B[:,3]-B[:,1]))
#                     union = a + b - inter + 1e-6
#                     ious[i] = inter / union
#                 return ious

#             iou_pred = iou_mat(trk_boxes, det_boxes)  # (T,D)

#             # 3) 行加权（pred_iou_ema 越高，IoU 越大 -> 代价更低）
#             row_ema = np.asarray(row_ema, dtype=np.float32).reshape(-1, 1)
#             iou_scaled = iou_pred * (0.5 + 0.5 * row_ema)  # scale to [0.25, 1.0]

#             # 4) 可选：速度方向一致性小幅加分（与 det 方向一致奖励）
#             if getattr(self, "use_velocity_align_bonus", True):
#                 bonus_w = float(getattr(d2mp_config, "assoc_align_bonus", 0.05))
#                 for r, gidx in enumerate(active_ids):
#                     trk = self.trackers[gidx]
#                     v = trk.velocity if getattr(trk, "velocity", None) is not None else np.array([0.0,0.0], dtype=np.float32)
#                     nrmv = np.linalg.norm(v)
#                     if nrmv < 1e-6:
#                         continue
#                     v = v / (nrmv + 1e-6)
#                     c_prev = np.array([(trk_boxes[r,0] + trk_boxes[r,2]) * 0.5,
#                                     (trk_boxes[r,1] + trk_boxes[r,3]) * 0.5], dtype=np.float32)
#                     centers_det = np.stack([(det_boxes[:,0]+det_boxes[:,2]) * 0.5,
#                                             (det_boxes[:,1]+det_boxes[:,3]) * 0.5], axis=1)
#                     disp = centers_det - c_prev
#                     nrm = np.linalg.norm(disp, axis=1) + 1e-6
#                     align = (disp / nrm) @ v
#                     align = np.clip(align, -1.0, 1.0)
#                     iou_scaled[r] += bonus_w * np.maximum(0.0, align)  # reward only same direction

#             # 5) 转为代价，并屏蔽低 IoU 对
#             final_cost = -iou_scaled
#             # 屏蔽 IoU 低于阈值的对（设置很大代价，防止被选中）
#             thr = float(self.iou_threshold)
#             mask_low = (iou_pred < thr)
#             final_cost = final_cost + (mask_low.astype(np.float32) * 1e6)

#             # 6) 匈牙利求解
#             # 返回的是 (row_t, col_d)
#             pairs = _hungarian_assign(final_cost)  # shape (K, 2)

#             # 7) 组装 matched/unmatched；按你后续使用的格式输出 (det_index, trk_index)
#             matched_list = []
#             used_trk_rows = set()
#             used_det_cols = set()
#             for r, c in pairs:
#                 if r < 0 or c < 0: 
#                     continue
#                 if iou_pred[r, c] < thr:
#                     continue
#                 trk_global_idx = active_ids[r]
#                 det_idx = int(c)
#                 matched_list.append([det_idx, trk_global_idx])  # 注意：输出 (det_index, trk_index)
#                 used_trk_rows.add(r); used_det_cols.add(c)

#             matched = np.asarray(matched_list, dtype=int) if len(matched_list) else np.empty((0, 2), dtype=int)

#             # unmatched（保持是全局 trk 索引和 det 索引）
#             all_trk_rows = set(range(len(active_ids)))
#             all_det_cols = set(range(det_boxes.shape[0]))
#             unmatched_trk_rows = sorted(list(all_trk_rows - used_trk_rows))
#             unmatched_det_cols = sorted(list(all_det_cols - used_det_cols))
#             unmatched_trks = np.asarray([active_ids[r] for r in unmatched_trk_rows], dtype=int)
#             unmatched_dets = np.asarray(unmatched_det_cols, dtype=int)
#         # ===== [PATCH-STEP3-END] =====
        
#         # # 可选：打印更有意义的 IoU 诊断（pred vs det）
#         # if len(ious_pred_vs_det) > 0:
#         #     print(f"[D2MP][diag] IoU(pred vs det) median: {np.median(ious_pred_vs_det):.3f}")

#         # Second round of association by OCR (BYTE optional)  
#         if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:  
#             u_trks = trks[unmatched_trks]  
#             iou_left = self.asso_func(dets_second, u_trks)  
#             iou_left = np.array(iou_left)  
#             if iou_left.max() > self.iou_threshold:  
#                 matched_indices = linear_assignment(-iou_left)  
#                 to_remove_trk_indices = []  
#                 for m in matched_indices:  
#                     det_ind, trk_ind = m[0], unmatched_trks[m[1]]  
#                     if iou_left[m[0], m[1]] < self.iou_threshold:  
#                         continue  
#                     self.trackers[trk_ind].update(dets_second[det_ind, :], img_size=(img_h, img_w))  
#                     to_remove_trk_indices.append(trk_ind)  
#                 unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))  

#         if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:  
#             left_dets = dets[unmatched_dets]  
#             left_trks = last_boxes[unmatched_trks]  
#             iou_left = self.asso_func(left_dets, left_trks)  
#             iou_left = np.array(iou_left)  
#             if iou_left.max() > self.iou_threshold:  
#                 rematched_indices = linear_assignment(-iou_left)  
#                 to_remove_det_indices = []  
#                 to_remove_trk_indices = []  
#                 for m in rematched_indices:  
#                     det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]  
#                     if iou_left[m[0], m[1]] < self.iou_threshold:  
#                         continue  
#                     self.trackers[trk_ind].update(dets[det_ind, :], img_size=(img_h, img_w))  
#                     to_remove_det_indices.append(det_ind)  
#                     to_remove_trk_indices.append(trk_ind)  
#                 unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))  
#                 unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))  

#         for m in unmatched_trks:  
#             self.trackers[m].update(None)  

#         # create and initialise new trackers  
#         for i in unmatched_dets:  
#             trk = DiffusionBoxTracker(dets[i, :], img_size=(img_h, img_w), delta_t=self.delta_t)  
#             self.trackers.append(trk)  

#         i = len(self.trackers)  
#         ret = []  
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

#     def update_public(self, dets, cates, scores, frame_id=None):  
#         """  
#         Public detections variant.  
#         Note: will use last known image size from update(); if None, falls back to (1080, 1920).  
#         frame_id: Optional[int] 仅用于外部想触发 reset 的场景（通常不需要）  
#         """  
#         # 若外层能传 frame_id 并希望在 public 模式也自动 reset，可启用如下检测；  
#         # 这里没有当前 img_size，只能依赖 frame_id 回绕（可选）  
#         if frame_id is not None:  
#             self._maybe_reset_on_boundary(frame_id=frame_id, img_size=self.last_img_size)  

#         self.frame_count += 1  

#         det_scores = np.ones((dets.shape[0], 1), dtype=np.float32)  
#         dets = np.concatenate((dets, det_scores), axis=1)  

#         remain_inds = scores > self.det_thresh  
#         cates = cates[remain_inds]  
#         dets = dets[remain_inds]  

#         trks = np.zeros((len(self.trackers), 5), dtype=np.float32)  
#         to_del = []  
#         ret = []  
#         for t, trk in enumerate(self.trackers):  
#             pos = trk.predict()[0]  
#             cat = getattr(trk, 'cate', 0)  
#             trks[t, :] = [pos[0], pos[1], pos[2], pos[3], cat]  
#             if np.any(np.isnan(pos)):  
#                 to_del.append(t)  
#         trks = np.ma.compress_rows(np.ma.masked_invalid(trks))  
#         for t in reversed(to_del):  
#             self.trackers.pop(t)  

#         velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0, 0), dtype=np.float32) for trk in self.trackers])  
#         last_boxes = np.array([trk.last_observation for trk in self.trackers])  
#         k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])  

#         matched, unmatched_dets, unmatched_trks = associate_kitti(  
#             dets, trks, cates, self.iou_threshold, velocities, k_observations, self.inertia  
#         )  

#         for m in matched:  
#             img_size = self.last_img_size if self.last_img_size is not None else (1080, 1920)  
#             self.trackers[m[1]].update(dets[m[0], :], img_size=img_size)  

#         if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:  
#             left_dets = dets[unmatched_dets]  
#             left_trks = last_boxes[unmatched_trks]  
#             left_dets_c = left_dets.copy()  
#             left_trks_c = left_trks.copy()  

#             iou_left = self.asso_func(left_dets_c, left_trks_c)  
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
#                     img_size = self.last_img_size if self.last_img_size is not None else (1080, 1920)  
#                     self.trackers[trk_ind].update(dets[det_ind, :], img_size=img_size)  
#                     to_remove_det_indices.append(det_ind)  
#                     to_remove_trk_indices.append(trk_ind)  
#                 unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))  
#                 unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))  

#         for i in unmatched_dets:  
#             img_size = self.last_img_size if self.last_img_size is not None else (1080, 1920)  
#             trk = DiffusionBoxTracker(dets[i, :], img_size=img_size, delta_t=self.delta_t)  
#             trk.cate = cates[i]  


# """
# OC-SORT with Soft Depth Regularization + D2MP (Diffusion Motion Prediction)

# Changes:
# - Class renamed to Diff_OCSort.
# - D2MP diffusion prediction replaces Kalman predict for association (switchable).
# - Keep Kalman predict for state evolution (for ablation).
# - Includes reset() and _maybe_reset_on_boundary() for cross-sequence handling.
# """

# from __future__ import print_function
# import warnings
# import math
# import numpy as np
# import torch
# import torch.nn as nn

# from .association import *
# from .diffusion import VarianceSchedule, HMINet, D2MP_OB


# # ---------------- Helper functions ---------------- #

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
#     w = bbox[2] - bbox[0]
#     h = bbox[3] - bbox[1]
#     x = bbox[0] + w / 2.0
#     y = bbox[1] + h / 2.0
#     s = w * h
#     r = w / float(h + 1e-6)
#     return np.array([x, y, s, r]).reshape((4, 1))


# def convert_x_to_bbox(x, score=None):
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


# def _xyxy_to_norm(xyxy, W, H):
#     x1, y1, x2, y2 = xyxy
#     return np.array([2*x1/W - 1, 2*y1/H - 1, 2*x2/W - 1, 2*y2/H - 1], dtype=np.float32)


# def _norm_to_xyxy(norm, W, H):
#     nx1, ny1, nx2, ny2 = norm
#     x1 = (nx1 + 1) * 0.5 * W
#     y1 = (ny1 + 1) * 0.5 * H
#     x2 = (nx2 + 1) * 0.5 * W
#     y2 = (ny2 + 1) * 0.5 * H
#     return np.array([x1, y1, x2, y2], dtype=np.float32)


# # ---------------- Kalman tracker ---------------- #

# class KalmanBoxTracker:
#     count = 0
#     def __init__(self, bbox, delta_t=3, orig=False, ema_obs=0.5, ema_pred=0.1):
#         if not orig:
#             from .kalmanfilter import KalmanFilterNew as KalmanFilter
#             self.kf = KalmanFilter(dim_x=7, dim_z=4)
#         else:
#             from filterpy.kalman import KalmanFilter
#             self.kf = KalmanFilter(dim_x=7, dim_z=4)

#         self.kf.F = np.array([
#             [1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],
#             [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]
#         ])
#         self.kf.H = np.array([
#             [1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]
#         ])
#         self.kf.R[2:,2:] *= 10.0
#         self.kf.P[4:,4:] *= 1000.0
#         self.kf.P *= 10.0
#         self.kf.Q[-1,-1] *= 0.01
#         self.kf.Q[4:,4:] *= 0.01

#         self.kf.x[:4] = convert_bbox_to_z(bbox)
#         self.time_since_update = 0
#         self.id = KalmanBoxTracker.count
#         KalmanBoxTracker.count += 1
#         self.history = []
#         self.hits = 0
#         self.hit_streak = 0
#         self.age = 0
#         self.last_observation = np.array([-1,-1,-1,-1,-1])
#         self.observations = {}
#         self.velocity = None
#         self.delta_t = delta_t
#         self.ema_obs = ema_obs
#         self.ema_pred = ema_pred
#         self.depth_ema = float(bbox[3])

#     def update(self, bbox):
#         if bbox is not None:
#             if self.last_observation.sum() >= 0:
#                 prev_box = self.last_observation
#                 self.velocity = speed_direction(prev_box, bbox)
#             y2 = float(bbox[3])
#             self.depth_ema = (1. - self.ema_obs) * self.depth_ema + self.ema_obs * y2
#             self.last_observation = bbox
#             self.observations[self.age] = bbox
#             self.time_since_update = 0
#             self.history = []
#             self.hits += 1
#             self.hit_streak += 1
#             self.kf.update(convert_bbox_to_z(bbox))
#         self.kf.update(bbox)

#     def predict(self):
#         if (self.kf.x[6] + self.kf.x[2]) <= 0: self.kf.x[6] *= 0.0
#         self.kf.predict()
#         self.age += 1
#         if self.time_since_update > 0: self.hit_streak = 0
#         self.time_since_update += 1
#         pred = convert_x_to_bbox(self.kf.x)
#         self.history.append(pred)
#         pred_y2 = float(pred[0][3])
#         self.depth_ema = (1. - self.ema_pred) * self.depth_ema + self.ema_pred * pred_y2
#         return self.history[-1]

#     def get_state(self): return convert_x_to_bbox(self.kf.x)


# DiffusionBoxTracker = KalmanBoxTracker  # alias


# # ---------------- small support for D2MP ---------------- #

# class _SimpleCfg:
#     def __init__(self, eps=1e-3): self.eps = eps


# class D2MPAdapter:
#     def __init__(self, enabled, context_dim=256, tf_layers=3, eps=1e-3,
#                  device='cuda',
#                  weight_path="/public/home/cookie/anaconda3/envs/ocsort_env/OC_SORT/trackers/diffmot_tracker/weights/MOT_epoch800.pt"):
#         self.enabled = bool(enabled)
#         self.ok = False
#         self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
#         self.context_dim = context_dim
#         if not self.enabled: return
#         try:
#             self.net = HMINet(point_dim=4, context_dim=context_dim, tf_layer=tf_layers).to(self.device)
#             self.var_sched = VarianceSchedule(num_steps=1, mode='linear')
#             self.wrapper = D2MP_OB(self.net, self.var_sched, _SimpleCfg(eps)).to(self.device)
#             self.wrapper.eval()
#             ckpt = torch.load(weight_path, map_location=self.device)
#             if isinstance(ckpt, dict):
#                 for key in ['net','model','state_dict']:
#                     if key in ckpt and isinstance(ckpt[key], dict):
#                         self.net.load_state_dict(ckpt[key], strict=False); self.ok=True; break
#             else:
#                 self.net.load_state_dict(ckpt, strict=False); self.ok=True
#         except Exception as e:
#             warnings.warn(f"D2MP load failed: {e}"); self.ok=False

#     def predict_boxes(self, last_obs, kalman_pred, kprev, vels, depths, ages, H,W,K=1,bestof=True):
#         if not (self.enabled and self.ok): return None
#         T=last_obs.shape[0]
#         ctx=[]
#         maxwh=max(float(W),float(H))
#         for i in range(T):
#             lo_n=_xyxy_to_norm(last_obs[i,:4],W,H)
#             pr_n=_xyxy_to_norm(kalman_pred[i,:4],W,H)
#             kp_n=_xyxy_to_norm(kprev[i,:4],W,H)
#             vel=vels[i] if vels is not None else np.zeros(2)
#             vel_n=vel/maxwh
#             y2=float(depths[i]); d_n=2*y2/H-1
#             a=float(ages[i]); a_n=2*min(a/30,1)-1
#             feat=np.concatenate([lo_n,pr_n,kp_n,vel_n,[d_n,a_n]]).astype(np.float32)
#             pad_w=max(0,self.context_dim-len(feat))
#             feat=np.pad(feat,(0,pad_w))
#             ctx.append(feat[:self.context_dim])
#         ctx=torch.tensor(np.stack(ctx)).float().to(self.device)
#         smp=self.wrapper.sample(ctx, sample=1, bestof=bestof, point_dim=4)
#         arr=smp.squeeze(0).cpu().numpy()  # (T,4)
#         out=np.zeros_like(arr)
#         for i in range(T): out[i]=_norm_to_xyxy(arr[i],W,H)
#         return out


# # ---------------- association mapping ---------------- #

# ASSO_FUNCS={"iou":iou_batch,"giou":giou_batch,"ciou":ciou_batch,"diou":diou_batch,"ct_dist":ct_dist}


# # ---------------- main tracker (renamed) ---------------- #

# class Diff_OCSort:
#     def __init__(self, det_thresh, max_age=30, min_hits=3, iou_threshold=0.3,
#                  delta_t=3, asso_func="iou", use_byte=False,
#                  ema_obs=0.5, ema_pred=0.1,
#                  use_d2mp=True, d2mp_context_dim=256, d2mp_tf_layers=3,
#                  d2mp_eps=1e-3, d2mp_device='cuda', d2mp_bestof=True):

#         self.max_age=max_age; self.min_hits=min_hits; self.iou_threshold=iou_threshold
#         self.det_thresh=det_thresh; self.delta_t=delta_t; self.asso_func=ASSO_FUNCS[asso_func]
#         self.use_byte=use_byte; self.ema_obs=ema_obs; self.ema_pred=ema_pred

#         self.use_d2mp=use_d2mp; self.d2mp_bestof=d2mp_bestof
#         self.d2mp=D2MPAdapter(enabled=use_d2mp, context_dim=d2mp_context_dim,
#                                tf_layers=d2mp_tf_layers, eps=d2mp_eps, device=d2mp_device)
#         if use_d2mp and not self.d2mp.ok:
#             print("[WARN] D2MP load failed, fallback to Kalman.")

#         self.trackers=[]; self.frame_count=0
#         self.last_img_size=None; self._prev_frame_id=-1

#     # --- Added Reset / boundary detection ---
#     def reset(self):
#         self.trackers=[]; self.frame_count=0; self.last_img_size=None
#         DiffusionBoxTracker.count=0; self._prev_frame_id=-1

#     def _maybe_reset_on_boundary(self, frame_id=None, img_size=None):
#         fid_back=(self._prev_frame_id!=-1 and frame_id is not None and frame_id<=self._prev_frame_id)
#         size_changed=(img_size and self.last_img_size and tuple(img_size)!=tuple(self.last_img_size))
#         if fid_back or size_changed:
#             print(f"[D2MP] new sequence detected, resetting tracker"); self.reset()
#         if frame_id is not None: self._prev_frame_id=int(frame_id)
#         if img_size is not None: self.last_img_size=tuple(img_size)

#     def _iou_single(self,a,b):
#         x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
#         inter=max(0,x2-x1)*max(0,y2-y1)
#         ua=max(0,(a[2]-a[0])*(a[3]-a[1])); ub=max(0,(b[2]-b[0])*(b[3]-b[1]))
#         return inter/(ua+ub-inter+1e-6)

# -*- coding: utf-8 -*-
"""  
Diff_OCSort: OC-SORT + Soft Depth + D2MP-OB (use last K observations) with robustness and GPU optimizations  

- D2MP-OB: uses last K (default=5) observed boxes per tracker as context  
- Batched per-frame D2MP inference, vectorized context, pinned H2D copy, non_blocking  
- Trust gating: soft fusion (default) between D2MP and Kalman; configurable thresholds  
- Stable numerics for Kalman→box conversion; robust association with valid kprev/velocity  
- Inference-only optimizations: global no-grad, eval, inference_mode, autocast, cudnn.benchmark  
"""  

from __future__ import print_function  
import warnings  
import numpy as np  
import torch  

# --------------------------- global inference mode and backends ---------------------------  
try:  
    torch.set_grad_enabled(False)  # inference global: drop autograd  
except Exception:  
    pass  

try:  
    torch.backends.cudnn.benchmark = True  
except Exception:  
    pass  
try:  
    torch.set_float32_matmul_precision("high")  
except Exception:  
    pass  

from .association import *  # iou_batch, giou_batch, ciou_batch, diou_batch, ct_dist, associate, associate_kitti, linear_assignment  
from .diffusion import VarianceSchedule, HMINet, D2MP_OB  


# --------------------------- helpers ---------------------------  

def k_previous_obs(observations, cur_age, k):  
    if len(observations) == 0:  
        return np.array([-1, -1, -1, -1, -1], dtype=np.float32)  
    for i in range(k):  
        dt = k - i  
        if cur_age - dt in observations:  
            return np.asarray(observations[cur_age - dt], dtype=np.float32)  
    max_age = max(observations.keys())  
    return np.asarray(observations[max_age], dtype=np.float32)  


def last_k_observations(observations, K):  
    """  
    Return array (K,5) of last K valid observations (xyxy[+score]), newest at the end by default.  
    If fewer than K exist, pad by repeating the most recent observation.  
    If none, return all -1.  
    """  
    if len(observations) == 0:  
        return np.tile(np.array([-1, -1, -1, -1, -1], dtype=np.float32), (K, 1))  
    keys = sorted(observations.keys())  
    vals = [np.asarray(observations[k], dtype=np.float32) for k in keys]  
    if len(vals) >= K:  
        tail = vals[-K:]  
    else:  
        pad = [vals[-1]] * (K - len(vals))  # repeat the most recent observation  
        tail = pad + vals  
    return np.stack(tail, axis=0).astype(np.float32)  


def speed_direction(bbox1, bbox2):  
    """  
    返回从 bbox1 到 bbox2 的中心点单位方向向量 [dy, dx]  
    """  
    b1 = np.asarray(bbox1, dtype=np.float32).reshape(-1)  
    b2 = np.asarray(bbox2, dtype=np.float32).reshape(-1)  
    cx1, cy1 = (b1[0] + b1[2]) * 0.5, (b1[1] + b1[3]) * 0.5  
    cx2, cy2 = (b2[0] + b2[2]) * 0.5, (b2[1] + b2[3]) * 0.5  
    dy, dx = float(cy2 - cy1), float(cx2 - cx1)  
    norm = np.sqrt(dy * dy + dx * dx) + 1e-6  
    return np.array([dy / norm, dx / norm], dtype=np.float32)  


def convert_bbox_to_z(bbox):  
    bbox = np.asarray(bbox, dtype=np.float32).reshape(-1)  
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])  
    w = x2 - x1  
    h = y2 - y1  
    x = x1 + w / 2.0  
    y = y1 + h / 2.0  
    s = w * h  
    r = w / float(h + 1e-6)  
    return np.array([x, y, s, r], dtype=np.float32).reshape((4, 1))  


def convert_x_to_bbox(x, score=None):  
    """  
    Robust: clamp s/r lower bound; protect sqrt/div; nan_to_num at end.  
    """  
    x = np.asarray(x, dtype=np.float32).reshape(-1)  
    s = float(x[2]); r = float(x[3])  
    if not np.isfinite(s): s = 1e-3  
    if not np.isfinite(r): r = 1.0  
    s = max(s, 1e-6)  
    r = max(r, 1e-6)  

    w = float(np.sqrt(s * r))  
    h = float(s / max(w, 1e-6))  
    x_c = float(x[0])  
    y_c = float(x[1])  

    x1 = x_c - w / 2.0  
    y1 = y_c - h / 2.0  
    x2 = x_c + w / 2.0  
    y2 = y_c + h / 2.0  

    if score is None:  
        out = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)  
    else:  
        out = np.array([x1, y1, x2, y2, float(score)], dtype=np.float32).reshape(1, 5)  
    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)  


def _xyxy_to_norm(xyxy, W, H):  
    v = np.asarray(xyxy, dtype=np.float32).reshape(-1)  
    W = float(np.asarray(W).reshape(()))  
    H = float(np.asarray(H).reshape(()))  
    x1, y1, x2, y2 = float(v[0]), float(v[1]), float(v[2]), float(v[3])  
    nx1 = 2.0 * x1 / W - 1.0  
    ny1 = 2.0 * y1 / H - 1.0  
    nx2 = 2.0 * x2 / W - 1.0  
    ny2 = 2.0 * y2 / H - 1.0  
    return np.array([nx1, ny1, nx2, ny2], dtype=np.float32)  


def _norm_to_xyxy(norm, W, H):  
    v = np.asarray(norm, dtype=np.float32).reshape(-1)  
    W = float(np.asarray(W).reshape(()))  
    H = float(np.asarray(H).reshape(()))  
    nx1, ny1, nx2, ny2 = float(v[0]), float(v[1]), float(v[2]), float(v[3])  
    x1 = float((nx1 + 1.0) * 0.5 * W)  
    y1 = float((ny1 + 1.0) * 0.5 * H)  
    x2 = float((nx2 + 1.0) * 0.5 * W)  
    y2 = float((ny2 + 1.0) * 0.5 * H)  
    return np.array([x1, y1, x2, y2], dtype=np.float32)  


# --------------------------- Kalman tracker ---------------------------  

class KalmanBoxTracker(object):  
    """  
    Kalman tracker + depth EMA smoothing.  
    """  
    count = 0  

    def __init__(self, bbox, delta_t=3, orig=False, ema_obs=0.5, ema_pred=0.1):  
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
        ], dtype=np.float32)  
        self.kf.H = np.array([  
            [1, 0, 0, 0, 0, 0, 0],  
            [0, 1, 0, 0, 0, 0, 0],  
            [0, 0, 1, 0, 0, 0, 0],  
            [0, 0, 0, 1, 0, 0, 0]  
        ], dtype=np.float32)  

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

        self.last_observation = np.array([-1, -1, -1, -1, -1], dtype=np.float32)  
        self.observations = dict()   # key: age(frame index), val: [x1,y1,x2,y2,score]  
        self.history_observations = []  
        self.velocity = None  
        self.delta_t = delta_t  

        self.ema_obs = float(ema_obs)  
        self.ema_pred = float(ema_pred)  
        self.depth_ema = float(np.asarray(bbox, dtype=np.float32).reshape(-1)[3])  

    def update(self, bbox):  
        if bbox is not None:  
            bbox = np.asarray(bbox, dtype=np.float32).reshape(-1)  
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

            y2 = float(bbox[3])  
            self.depth_ema = (1.0 - self.ema_obs) * self.depth_ema + self.ema_obs * y2  

            if bbox.shape[0] >= 5:  
                self.last_observation = np.array([bbox[0], bbox[1], bbox[2], bbox[3], float(bbox[4])], dtype=np.float32)  
            else:  
                self.last_observation = np.array([bbox[0], bbox[1], bbox[2], bbox[3], -1.0], dtype=np.float32)  
            self.observations[self.age] = self.last_observation  
            self.history_observations.append(self.last_observation)  

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
        # Clamp s/r to positive  
        self.kf.x[2] = np.array([max(float(self.kf.x[2]), 1e-6)], dtype=np.float32)  
        self.kf.x[3] = np.array([max(float(self.kf.x[3]), 1e-6)], dtype=np.float32)  

        self.age += 1  
        if self.time_since_update > 0:  
            self.hit_streak = 0  
        self.time_since_update += 1  

        pred = convert_x_to_bbox(self.kf.x)  # (1,4)  
        self.history.append(pred)  

        pred_y2 = float(pred[0][3])  
        self.depth_ema = (1.0 - self.ema_pred) * self.depth_ema + self.ema_pred * pred_y2  
        return self.history[-1]  

    def get_state(self):  
        return convert_x_to_bbox(self.kf.x)  


DiffusionBoxTracker = KalmanBoxTracker  


# --------------------------- D2MP Adapter ---------------------------  

class _SimpleConfig:  
    def __init__(self, eps=1e-3):  
        self.eps = float(eps)  


class D2MPAdapter:  
    """  
    Two modes:  
      - OB mode (recommended for D2MP-OB): context built from last K observations (K*4 dims, padded to context_dim)  
      - Legacy mode: (kept for compatibility) mixed features including last/pred/kprev/velocity/depth/age (16 dims), padded  
    """  
    def __init__(  
        self,  
        enabled: bool,  
        weight_path: str,  
        context_dim: int = 256,  
        tf_layers: int = 3,  
        eps: float = 1e-3,  
        device: str = 'cuda',  
        use_fp16: bool = False,  
        use_bf16: bool = False,  
        ob_win: int = 5,           # OB window length K  
        use_ob_mode: bool = True,  # True => use OB mode (last K frames)  
        steps: int = 1             # diffusion steps; try 10 for quality if weights trained multi-step  
    ):  
        self.enabled = bool(enabled)  
        self.ok = False  
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith('cuda') else 'cpu')  
        self.context_dim = int(context_dim)  
        self.tf_layers = int(tf_layers)  
        self.eps = float(eps)  
        self.use_fp16 = bool(use_fp16) and (self.device.type == 'cuda')  
        self.use_bf16 = bool(use_bf16) and (self.device.type == 'cuda')  
        if self.use_fp16 and self.use_bf16:  
            # prefer bf16  
            self.use_fp16 = False  

        self.ob_win = int(ob_win)  
        self.use_ob_mode = bool(use_ob_mode)  
        self.steps = int(steps)  

        if not self.enabled:  
            return  

        try:  
            self.net = HMINet(point_dim=4, context_dim=self.context_dim, tf_layer=self.tf_layers).to(self.device)  
            self.var_sched = VarianceSchedule(num_steps=self.steps, mode='linear')  
            self.wrapper = D2MP_OB(self.net, self.var_sched, _SimpleConfig(eps=self.eps)).to(self.device)  

            # freeze + eval  
            self.net.eval()  
            self.wrapper.eval()  
            for p in self.net.parameters():  
                p.requires_grad_(False)  
            for p in self.wrapper.parameters():  
                p.requires_grad_(False)  

            # dtype cast  
            if self.use_bf16:  
                self.net = self.net.to(torch.bfloat16)  
                self.wrapper = self.wrapper.to(torch.bfloat16)  
            elif self.use_fp16:  
                self.net = self.net.half()  
                self.wrapper = self.wrapper.half()  

            # Safe weight load  
            try:  
                ckpt = torch.load(weight_path, map_location=self.device, weights_only=True)  
            except TypeError:  
                ckpt = torch.load(weight_path, map_location=self.device)  
            loaded = False  
            if isinstance(ckpt, dict):  
                for key in ['net', 'model', 'state_dict']:  
                    if key in ckpt and isinstance(ckpt[key], dict):  
                        try:  
                            self.net.load_state_dict(ckpt[key], strict=True); loaded = True; break  
                        except Exception:  
                            try:  
                                self.wrapper.load_state_dict(ckpt[key], strict=False); loaded = True; break  
                            except Exception:  
                                pass  
            if not loaded:  
                try:  
                    self.net.load_state_dict(ckpt, strict=True); loaded = True  
                except Exception:  
                    try:  
                        self.wrapper.load_state_dict(ckpt, strict=False); loaded = True  
                    except Exception:  
                        pass  
            if not loaded:  
                warnings.warn("D2MP: cannot load state_dict; disabled.")  
                return  
            if isinstance(ckpt, dict) and 'config' in ckpt and isinstance(ckpt['config'], dict) and 'eps' in ckpt['config']:  
                self.eps = float(ckpt['config']['eps'])  

            self.ok = True  
            self._warmed_bs = 0  
        except Exception as e:  
            warnings.warn(f"D2MP: load failed, disabled. err={e}")  
            self.ok = False  

    def _amp_dtype(self):  
        if self.use_bf16:  
            return torch.bfloat16  
        if self.use_fp16:  
            return torch.float16  
        return None  

    def warmup(self, batch_size: int):  
        if (not self.enabled) or (not self.ok) or (self.device.type != 'cuda'):  
            return  
        if batch_size <= 0 or batch_size <= self._warmed_bs:  
            return  
        dummy = torch.zeros((batch_size, self.context_dim), device=self.device)  
        amp_dtype = self._amp_dtype()  
        with torch.inference_mode():  
            if amp_dtype is not None:  
                with torch.autocast(device_type="cuda", dtype=amp_dtype):  
                    _ = self.wrapper.sample(context=dummy, sample=1, bestof=True, point_dim=4, flexibility=0.0, ret_traj=False)  
            else:  
                _ = self.wrapper.sample(context=dummy, sample=1, bestof=True, point_dim=4, flexibility=0.0, ret_traj=False)  
        self._warmed_bs = batch_size  

    # --------- OB mode: vectorized context build (CPU) ---------  
    def _build_context_ob(  
        self,  
        obs_seq_5,  # (T, K, 5) last K observations (xyxy + score)  
        img_h: int,  
        img_w: int  
    ):  
        """  
        OB 模式上下文：将最近 K 个观测框的 xyxy 归一化后展平 (K*4)，再 pad 到 context_dim  
        采用 newest-first 顺序（默认训练常用），即序列按 [t, t-1, ..., t-K+1] 展平  
        """  
        obs_seq_5 = np.asarray(obs_seq_5, dtype=np.float32)  # (T,K,5)  
        T, K = obs_seq_5.shape[0], obs_seq_5.shape[1]  
        if T == 0:  
            return np.zeros((0, self.context_dim), dtype=np.float32)  

        # newest-first  
        obs_seq_5 = obs_seq_5[:, ::-1, :]  # flip along K  

        W = float(img_w); H = float(img_h)  
        xyxy = obs_seq_5[:, :, :4].reshape(T * K, 4)  
        # 归一化 [-1,1]  
        xyxy_n = np.empty_like(xyxy, dtype=np.float32)  
        xyxy_n[:, 0] = 2.0 * xyxy[:, 0] / W - 1.0  
        xyxy_n[:, 1] = 2.0 * xyxy[:, 1] / H - 1.0  
        xyxy_n[:, 2] = 2.0 * xyxy[:, 2] / W - 1.0  
        xyxy_n[:, 3] = 2.0 * xyxy[:, 3] / H - 1.0  
        xyxy_n = xyxy_n.reshape(T, K * 4)  

        # # 将 -1（无观测）转换为 0 特征（仅当整条轨迹从未观测时）  
        # mask_invalid = (obs_seq_5[:, :, 0] < 0) | (obs_seq_5[:, :, 1] < 0)  
        # if np.any(mask_invalid):  
        #     mask_invalid_flat = np.repeat(mask_invalid, 4, axis=1)  # (T, K*4)  
        #     xyxy_n[mask_invalid_flat] = 0.0  

        # pad 到 context_dim  
        if xyxy_n.shape[1] == self.context_dim:  
            return xyxy_n  
        elif xyxy_n.shape[1] < self.context_dim:  
            pad = np.zeros((T, self.context_dim - xyxy_n.shape[1]), dtype=np.float32)  
            return np.concatenate([xyxy_n, pad], axis=1)  
        else:  
            return xyxy_n[:, :self.context_dim]  

    # --------- Legacy (非 OB) 模式：兼容旧代码（保留） ---------  
    def _build_context_legacy(  
        self,  
        last_obs_xyxy, kalman_pred_xyxy, kprev_xyxy, velocities, depth_emas, ages, img_h: int, img_w: int  
    ):  
        last_obs_xyxy = np.asarray(last_obs_xyxy, dtype=np.float32)  
        kalman_pred_xyxy = np.asarray(kalman_pred_xyxy, dtype=np.float32)  
        kprev_xyxy = np.asarray(kprev_xyxy, dtype=np.float32)  
        velocities = np.asarray(velocities, dtype=np.float32)  
        depth_emas = np.asarray(depth_emas, dtype=np.float32)  
        ages = np.asarray(ages, dtype=np.float32)  

        T = last_obs_xyxy.shape[0]  
        if T == 0:  
            return np.zeros((0, self.context_dim), dtype=np.float32)  

        W = float(img_w); H = float(img_h)  
        # 回填无效的 kprev  
        kp_bad = (kprev_xyxy[:, :4] < 0).any(axis=1)  
        if kp_bad.any():  
            kprev_xyxy[kp_bad, :4] = last_obs_xyxy[kp_bad, :4]  
            if kprev_xyxy.shape[1] >= 5 and last_obs_xyxy.shape[1] >= 5:  
                kprev_xyxy[kp_bad, 4] = last_obs_xyxy[kp_bad, 4]  

        def norm_xyxy(x):  
            x = np.asarray(x, dtype=np.float32)  
            out = np.empty_like(x[:, :4], dtype=np.float32)  
            out[:, 0] = 2.0 * x[:, 0] / W - 1.0  
            out[:, 1] = 2.0 * x[:, 1] / H - 1.0  
            out[:, 2] = 2.0 * x[:, 2] / W - 1.0  
            out[:, 3] = 2.0 * x[:, 3] / H - 1.0  
            return out  

        lo_n = norm_xyxy(last_obs_xyxy)  
        pr_n = norm_xyxy(kalman_pred_xyxy)  
        kp_n = norm_xyxy(kprev_xyxy)  

        max_wh = max(W, H, 1.0)  
        vel_n = np.zeros_like(velocities, dtype=np.float32)  
        vel_n[:, 0] = velocities[:, 0] / max_wh  
        vel_n[:, 1] = velocities[:, 1] / max_wh  

        depth_n = 2.0 * (depth_emas / max(H, 1e-6)) - 1.0  
        age01 = np.clip(ages / 30.0, 0.0, 1.0)  
        age_n = 2.0 * age01 - 1.0  

        ctx16 = np.concatenate([lo_n, pr_n, kp_n, vel_n, depth_n[:, None], age_n[:, None]], axis=1).astype(np.float32)  
        if ctx16.shape[1] == self.context_dim:  
            return ctx16  
        elif ctx16.shape[1] < self.context_dim:  
            pad = np.zeros((T, self.context_dim - ctx16.shape[1]), dtype=np.float32)  
            return np.concatenate([ctx16, pad], axis=1)  
        else:  
            return ctx16[:, :self.context_dim]  

    @torch.no_grad()  
    def predict_boxes_ob(  
        self,  
        obs_seq_5,   # (T,K,5)  
        img_h: int, img_w: int,  
        K: int = 1,  
        bestof: bool = True  
    ):  
        if (not self.enabled) or (not self.ok):  
            return None  
        T = int(np.asarray(obs_seq_5).shape[0])  
        if T == 0:  
            return None  

        ctx_np = self._build_context_ob(obs_seq_5, img_h, img_w)  # (T, context_dim)  
        if ctx_np.size == 0:  
            return None  
        ctx = torch.from_numpy(ctx_np).pin_memory().to(self.device, non_blocking=True)  

        # warmup  
        self.warmup(T)  

        amp_dtype = self._amp_dtype()  
        with torch.inference_mode():  
            if amp_dtype is not None and self.device.type == "cuda":  
                with torch.autocast(device_type="cuda", dtype=amp_dtype):  
                    out = self.wrapper.sample(context=ctx, sample=K, bestof=bestof, point_dim=4, flexibility=0.0, ret_traj=False)  
            else:  
                out = self.wrapper.sample(context=ctx, sample=K, bestof=bestof, point_dim=4, flexibility=0.0, ret_traj=False)  

        if isinstance(out, torch.Tensor):  
            arr = out.squeeze(0).detach().to(torch.float32).cpu().numpy()  # (T,4) in [-1,1]  
        else:  
            arr = np.asarray(out).squeeze(0).astype(np.float32)  

        # 反归一化 + clamp  
        xyxy = np.empty((T, 1, 4), dtype=np.float32)  
        Wf = float(img_w); Hf = float(img_h)  
        x1 = (arr[:, 0] + 1.0) * 0.5 * Wf  
        y1 = (arr[:, 1] + 1.0) * 0.5 * Hf  
        x2 = (arr[:, 2] + 1.0) * 0.5 * Wf  
        y2 = (arr[:, 3] + 1.0) * 0.5 * Hf  
        xlo = np.minimum(x1, x2); xhi = np.maximum(x1, x2)  
        ylo = np.minimum(y1, y2); yhi = np.maximum(y1, y2)  
        xlo = np.clip(xlo, 0.0, Wf - 1.0); xhi = np.clip(xhi, 0.0, Wf - 1.0)  
        ylo = np.clip(ylo, 0.0, Hf - 1.0); yhi = np.clip(yhi, 0.0, Hf - 1.0)  
        xyxy[:, 0, 0] = xlo.astype(np.float32)  
        xyxy[:, 0, 1] = ylo.astype(np.float32)  
        xyxy[:, 0, 2] = xhi.astype(np.float32)  
        xyxy[:, 0, 3] = yhi.astype(np.float32)  
        return xyxy  # (T,1,4)  

    @torch.no_grad()  
    def predict_boxes_legacy(  
        self,  
        last_obs_xyxy, kalman_pred_xyxy, kprev_xyxy, velocities, depth_emas, ages,  
        img_h: int, img_w: int, K: int = 1, bestof: bool = True  
    ):  
        if (not self.enabled) or (not self.ok):  
            return None  
        T = int(np.asarray(last_obs_xyxy).shape[0])  
        if T == 0:  
            return None  

        ctx_np = self._build_context_legacy(last_obs_xyxy, kalman_pred_xyxy, kprev_xyxy, velocities, depth_emas, ages, img_h, img_w)  
        if ctx_np.size == 0:  
            return None  
        ctx = torch.from_numpy(ctx_np).pin_memory().to(self.device, non_blocking=True)  

        # warmup  
        self.warmup(T)  

        amp_dtype = self._amp_dtype()  
        with torch.inference_mode():  
            if amp_dtype is not None and self.device.type == "cuda":  
                with torch.autocast(device_type="cuda", dtype=amp_dtype):  
                    out = self.wrapper.sample(context=ctx, sample=K, bestof=bestof, point_dim=4, flexibility=0.0, ret_traj=False)  
            else:  
                out = self.wrapper.sample(context=ctx, sample=K, bestof=bestof, point_dim=4, flexibility=0.0, ret_traj=False)  

        if isinstance(out, torch.Tensor):  
            arr = out.squeeze(0).detach().to(torch.float32).cpu().numpy()  
        else:  
            arr = np.asarray(out).squeeze(0).astype(np.float32)  

        xyxy = np.empty((T, 1, 4), dtype=np.float32)  
        Wf = float(img_w); Hf = float(img_h)  
        x1 = (arr[:, 0] + 1.0) * 0.5 * Wf  
        y1 = (arr[:, 1] + 1.0) * 0.5 * Hf  
        x2 = (arr[:, 2] + 1.0) * 0.5 * Wf  
        y2 = (arr[:, 3] + 1.0) * 0.5 * Hf  
        xlo = np.minimum(x1, x2); xhi = np.maximum(x1, x2)  
        ylo = np.minimum(y1, y2); yhi = np.maximum(y1, y2)  
        xlo = np.clip(xlo, 0.0, Wf - 1.0); xhi = np.clip(xhi, 0.0, Wf - 1.0)  
        ylo = np.clip(ylo, 0.0, Hf - 1.0); yhi = np.clip(yhi, 0.0, Hf - 1.0)  
        xyxy[:, 0, 0] = xlo.astype(np.float32)  
        xyxy[:, 0, 1] = ylo.astype(np.float32)  
        xyxy[:, 0, 2] = xhi.astype(np.float32)  
        xyxy[:, 0, 3] = yhi.astype(np.float32)  
        return xyxy  # (T,1,4)  


# --------------------------- association funcs ---------------------------  

ASSO_FUNCS = {  
    "iou": iou_batch,  
    "giou": giou_batch,  
    "ciou": ciou_batch,  
    "diou": diou_batch,  
    "ct_dist": ct_dist  
}  


# --------------------------- main tracker with batched D2MP-OB ---------------------------  

class Diff_OCSort(object):  
    def __init__(  
        self,  
        det_thresh,  
        max_age=30,  
        min_hits=3,  
        iou_threshold=0.3,  
        delta_t=3,  
        asso_func="iou",  
        inertia=0.2,  
        use_byte=False,  
        # Soft-depth params  
        depth_alpha=0.5,  
        depth_beta=2.0,  
        depth_gate=0.25,  
        gate_floor=0.2,  
        ema_obs=0.5,  
        ema_pred=0.1,  

        # --- D2MP params ---  
        use_d2mp=True,  
        d2mp_weight_path="/public/home/cookie/anaconda3/envs/ocsort_env/OC_SORT/trackers/diffmot_tracker/weights/MOT_epoch800.pt",  
        d2mp_context_dim=256,  
        d2mp_tf_layers=3,  
        d2mp_eps=1e-3,  
        d2mp_device='cuda',  
        d2mp_bestof=True,  
        d2mp_fp16=False,  
        d2mp_bf16=False,  
        d2mp_steps=6,              # diffusion steps; try 10 if your weights trained multi-step  

        # --- OB 模式 ---  
        d2mp_use_ob=True,  
        ob_win=5,  # 使用前 5 帧（newest-first 顺序）  

        # D2MP 门控与融合
        d2mp_min_hist=5,  
        d2mp_trust_iou=0.35,       # D2MP vs Kalman IoU 起点（更稳）  
        d2mp_trust_last_iou=0.25,  # D2MP vs last observation IoU 起点  
        d2mp_use_soft_fusion=True, # 软融合（推荐开启）  
        d2mp_soft_alpha_kf=0.6,    # 融合时 IoU_kf 权重  
        d2mp_soft_alpha_last=0.4,  # 融合时 IoU_last 权重  
        d2mp_soft_start=0.2,       # alpha 起始阈值（越高越保守）  
        d2mp_soft_span=0.3,        # alpha 增长跨度  
        d2mp_area_ratio_min=0.6,   # 面积比约束下限  
        d2mp_dir_cos_min=0.0,      # 方向一致性最小 cos  
        d2mp_only_unstable=False   # 仅对不稳定轨迹使用 D2MP（遮挡或短龄）         
        
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
        self.ema_obs = float(ema_obs)  
        self.ema_pred = float(ema_pred)  

        # D2MP (OB enabled by default)  
        self.use_d2mp = bool(use_d2mp)  
        self.d2mp_bestof = bool(d2mp_bestof)  
        self.ob_win = int(ob_win)  
        self.d2mp_use_ob = bool(d2mp_use_ob) 
        self.d2mp_min_hist = int(d2mp_min_hist)

        self.d2mp = D2MPAdapter(  
            enabled=self.use_d2mp,  
            weight_path=d2mp_weight_path,  
            context_dim=d2mp_context_dim,  
            tf_layers=d2mp_tf_layers,  
            eps=d2mp_eps,  
            device=d2mp_device,  
            use_fp16=d2mp_fp16,  
            use_bf16=d2mp_bf16,  
            ob_win=self.ob_win,  
            use_ob_mode=self.d2mp_use_ob,  
            steps=int(d2mp_steps)  
            
        )  
        if self.use_d2mp and not self.d2mp.ok:  
            print("[WARN] D2MP enabled but failed to load; falling back to Kalman-only.")  

        # 门控与融合参数  
        self.d2mp_trust_iou = float(d2mp_trust_iou)  
        self.d2mp_trust_last_iou = float(d2mp_trust_last_iou)  
        self.d2mp_use_soft_fusion = bool(d2mp_use_soft_fusion)  
        self.d2mp_soft_alpha_kf = float(d2mp_soft_alpha_kf)  
        self.d2mp_soft_alpha_last = float(d2mp_soft_alpha_last)  
        self.d2mp_soft_start = float(d2mp_soft_start)  
        self.d2mp_soft_span = float(d2mp_soft_span)  
        self.d2mp_area_ratio_min = float(d2mp_area_ratio_min)  
        self.d2mp_dir_cos_min = float(d2mp_dir_cos_min)  
        self.d2mp_only_unstable = bool(d2mp_only_unstable)  
    
        # cross-sequence state  
        self.last_img_size = None  
        self._prev_frame_id = -1  

        KalmanBoxTracker.count = 0  
        
        

    # --- 跨序列 reset ---  
    def reset(self):  
        self.trackers = []  
        self.frame_count = 0  
        self.last_img_size = None  
        DiffusionBoxTracker.count = 0  
        self._prev_frame_id = -1  
        
        self._last_d2mp_pred = None

    def _maybe_reset_on_boundary(self, frame_id=None, img_size=None):  
        fid_back = (self._prev_frame_id != -1 and frame_id is not None and frame_id <= self._prev_frame_id)  
        size_changed = (img_size is not None and self.last_img_size is not None and tuple(img_size) != tuple(self.last_img_size))  
        if fid_back or size_changed:  
            print(f"[D2MP] new sequence detected (fid_back={fid_back}, size_changed={size_changed}), resetting tracker")  
            self.reset()  
        if frame_id is not None:  
            self._prev_frame_id = int(frame_id)  
        if img_size is not None:  
            self.last_img_size = tuple(img_size)  

    def _iou_single(self, a, b):  
        a = np.asarray(a, dtype=np.float32).reshape(-1)  
        b = np.asarray(b, dtype=np.float32).reshape(-1)  
        x1 = max(float(a[0]), float(b[0])); y1 = max(float(a[1]), float(b[1]))  
        x2 = min(float(a[2]), float(b[2])); y2 = min(float(a[3]), float(b[3]))  
        iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)  
        inter = iw * ih  
        area_a = max(0.0, (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1])))  
        area_b = max(0.0, (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1])))  
        union = area_a + area_b - inter + 1e-6  
        return inter / union  

    # ---------- Soft-depth ----------  
    @staticmethod  
    def _y2_from_boxes(boxes):  
        boxes = np.asarray(boxes, dtype=np.float32)  
        return boxes[:, 3].astype(float) if boxes.size > 0 else np.array([])  

    def _depth_affinity(self, dets_xyxy, trks_xyxy, trk_depth_ema, img_h):  
        dets_xyxy = np.asarray(dets_xyxy, dtype=np.float32)  
        trks_xyxy = np.asarray(trks_xyxy, dtype=np.float32)  
        trk_depth_ema = np.asarray(trk_depth_ema, dtype=np.float32)  
        if dets_xyxy.size == 0 or trks_xyxy.size == 0:  
            return np.zeros((dets_xyxy.shape[0], trks_xyxy.shape[0]), dtype=float)  
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

    def _blend_iou_with_depth(self, iou, affinity):  
        if self.depth_alpha <= 0:  
            return iou  
        return iou * ((1.0 - self.depth_alpha) + self.depth_alpha * affinity)  

    # ---------------------- Batched D2MP-OB predict with trust gating ----------------------  

    def _batched_predict(self, trackers, img_size, kf_preds):  
        """  
        批处理当前帧所有轨迹：  
          - 优先使用 D2MP-OB（若可用）：基于每条轨迹最近 K 次观测（newest-first）  
          - 门控与融合：默认软融合，衰减离群预测，稳健贴合 Kalman  
        返回 (N,4) numpy  
        """  
        T = len(trackers)  
        if T == 0:  
            return np.zeros((0, 4), dtype=np.float32)  
        img_h, img_w = int(img_size[0]), int(img_size[1])  
        
        # 加这一行，确保后续条件判断里 d2_preds 总是已定义
        d2_preds = None

        # Gather sequences: last K observations per tracker (newest-at-end -> flip later)  
        K = self.ob_win  
        obs_seq = np.zeros((T, K, 5), dtype=np.float32)  
        last_boxes = np.zeros((T, 5), dtype=np.float32)  
        for i, trk in enumerate(trackers):  
            seq_k = last_k_observations(trk.observations, K)  # (K,5), oldest..newest  
            obs_seq[i] = seq_k  
            last_boxes[i] = trk.last_observation  

        kf_preds = np.asarray(kf_preds, dtype=np.float32).reshape(T, 4)  

        # Track stability features
        ages = np.array([trk.age for trk in trackers], dtype=np.float32)
        hit_streaks = np.array([trk.hit_streak for trk in trackers], dtype=np.float32)
        tsu = np.array([trk.time_since_update for trk in trackers], dtype=np.float32)

        # 先做“最小连续匹配次数”门槛
        hist_ok = (hit_streaks >= self.d2mp_min_hist)

        # 再按需限制为仅不稳定（遮挡）轨迹
        if self.d2mp_only_unstable:
            apply_mask = hist_ok & (tsu > 0)         # 仅遮挡且有足够历史的轨迹
        else:
            apply_mask = hist_ok                     # 任何有足够历史的轨迹

        # 历史长度门控：只有 hit_streak >= d2mp_min_hist 的轨迹才允许使用 D2MP
        hist_ok = (hit_streaks >= self.d2mp_min_hist)
        apply_mask = apply_mask & hist_ok

        # 若没有任何轨迹满足门槛，直接返回 Kalman 预测
        if not np.any(apply_mask):
            return kf_preds.copy()

        # 软融合缩放：稳定/遮挡/新生的不同权重
        alpha_scale = np.ones(len(trackers), dtype=np.float32)
        alpha_scale[(tsu == 0) & (hit_streaks >= 5)] = 0.4
        alpha_scale[(tsu > 0)] = 1.2
        alpha_scale[(hit_streaks <= 1)] = 1.1
        alpha_scale = np.clip(alpha_scale, 0.3, 1.5)

        # 子集索引
        mask_idx = np.where(apply_mask)[0]

        
        use_d2 = (self.use_d2mp and self.d2mp.ok and self.d2mp_use_ob)
        d2_sub = None
        if use_d2 and mask_idx.size > 0:
            try:
                d2mp_props = self.d2mp.predict_boxes_ob(
                    obs_seq[mask_idx], img_h=img_h, img_w=img_w, K=1, bestof=self.d2mp_bestof
                )
                if d2mp_props is not None and d2mp_props.shape[0] == mask_idx.size:
                    d2_sub = d2mp_props[:, 0, :]
            except Exception as e:
                print(f"[WARN] D2MP-OB prediction failed (subset), fallback to Kalman. err={e}")
                d2_sub = None

        out = np.empty((T, 4), dtype=np.float32)
        out[:] = kf_preds
        if d2_sub is None or mask_idx.size == 0:
            return out
        
        

        kf_sub = kf_preds[mask_idx]
        last_sub = last_boxes[mask_idx, :4]
        fused_sub = self._fuse_d2mp_with_kf(d2_sub, kf_sub, last_sub, alpha_scale=alpha_scale[mask_idx])
        out[mask_idx] = fused_sub
        return out

    # def _fuse_d2mp_with_kf(self, d2_preds, kf_preds, last_boxes): 
    def _fuse_d2mp_with_kf(self, d2_preds, kf_preds, last_boxes, alpha_scale=None): 
        """  
        融合模块：  
          - 若 self.d2mp_use_soft_fusion=True：进行软融合（按 IoU/面积比/方向一致性计算 alpha）  
          - 否则：按硬门控（IoU 阈值）选择 D2 或 Kalman  
        输入与输出均为 (N,4) numpy  
        """  
        d2_preds = np.asarray(d2_preds, dtype=np.float32)  
        kf_preds = np.asarray(kf_preds, dtype=np.float32)  
        last_boxes = np.asarray(last_boxes, dtype=np.float32)  
        N = d2_preds.shape[0]  
        if N == 0:  
            return kf_preds  

        def iou_xyxy(a, b):  
            xa1 = np.maximum(a[:, 0], b[:, 0]); ya1 = np.maximum(a[:, 1], b[:, 1])  
            xa2 = np.minimum(a[:, 2], b[:, 2]); ya2 = np.minimum(a[:, 3], b[:, 3])  
            iw = np.clip(xa2 - xa1, 0.0, None); ih = np.clip(ya2 - ya1, 0.0, None)  
            inter = iw * ih  
            area_a = np.clip((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), 0.0, None)  
            area_b = np.clip((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 0.0, None)  
            union = area_a + area_b - inter + 1e-6  
            return inter / union  

        iou_kf = iou_xyxy(d2_preds, kf_preds)  
        iou_last = iou_xyxy(d2_preds, last_boxes)  

        if not self.d2mp_use_soft_fusion:  
            # 硬门控：更保守的 IoU 阈值  
            mask = (iou_kf >= self.d2mp_trust_iou) | (iou_last >= self.d2mp_trust_last_iou)  
            out = np.where(mask[:, None], d2_preds, kf_preds)  
            return out.astype(np.float32)  

        # 软融合：面积比 + 方向一致性 + IoU 形成 alpha  
        def area(x): return np.clip((x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1]), 1e-6, None)  
        aratio = np.minimum(area(d2_preds) / area(kf_preds), area(kf_preds) / area(d2_preds))  
        area_ok = (aratio > self.d2mp_area_ratio_min)  

        cx_lo = (last_boxes[:, 0] + last_boxes[:, 2]) * 0.5  
        cy_lo = (last_boxes[:, 1] + last_boxes[:, 3]) * 0.5  
        cx_kf = (kf_preds[:, 0] + kf_preds[:, 2]) * 0.5  
        cy_kf = (kf_preds[:, 1] + kf_preds[:, 3]) * 0.5  
        cx_d2 = (d2_preds[:, 0] + d2_preds[:, 2]) * 0.5  
        cy_d2 = (d2_preds[:, 1] + d2_preds[:, 3]) * 0.5  

        v_kf = np.stack([cy_kf - cy_lo, cx_kf - cx_lo], 1)  
        v_d2 = np.stack([cy_d2 - cy_lo, cx_d2 - cx_lo], 1)  
        def norm(v):  
            n = np.linalg.norm(v, axis=1, keepdims=True); n[n < 1e-6] = 1.0  
            return v / n  
        cos_sim = (norm(v_kf) * norm(v_d2)).sum(1)  
        dir_ok = (cos_sim > self.d2mp_dir_cos_min)  

        # alpha 合成：随 IoU 提升而增大，受 area_ok 和 dir_ok 抑制  
        alpha_raw = self.d2mp_soft_alpha_kf * iou_kf + self.d2mp_soft_alpha_last * iou_last  
        alpha = np.clip((alpha_raw - self.d2mp_soft_start) / max(self.d2mp_soft_span, 1e-6), 0.0, 1.0)  
        alpha *= (area_ok & dir_ok)  
        
                # 根据轨迹稳定度缩放 alpha（稳定轨迹更保守）
        if alpha_scale is not None:
            alpha = alpha * alpha_scale.astype(np.float32)

        # 异常位移抑制：若 Kalman 预计几乎不动而 D2MP 位移过大，抑制 D2MP 权重
        # 计算位移幅度
        d_kf = np.hypot(cy_kf - cy_lo, cx_kf - cx_lo)
        d_d2 = np.hypot(cy_d2 - cy_lo, cx_d2 - cx_lo)
        # 位移比（相对 Kalman）
        ratio = d_d2 / np.maximum(d_kf, 1.0)
        # 当 Kalman 位移很小但 D2MP 很大时，强抑制
        spike_mask = (d_kf < 3.0) & (d_d2 > 20.0)
        alpha[spike_mask] *= 0.1
        # 一般性抑制：位移比越大，alpha 越小（2/(1+ratio) 限幅到 [0.3, 1.0]）
        alpha *= np.clip(2.0 / (1.0 + ratio), 0.3, 1.0)
        

        out = (1.0 - alpha)[:, None] * kf_preds + alpha[:, None] * d2_preds  
        return out.astype(np.float32)  

    # ---------------------------------------  

    def update(self, output_results, img_info, img_size, frame_id=None):  
        """  
        Optional: supply frame_id for cross-sequence boundary detection.  
        img_info: (H, W, ...) in original image  
        img_size: (input_h, input_w) for detector scaling back  
        """  
        # auto reset on new sequence boundary  
        self._maybe_reset_on_boundary(frame_id=frame_id, img_size=(img_info[0], img_info[1]))  

        # if output_results is None:  
        #     return np.empty((0, 5))  
        
        if output_results is None:
            # 当作空检测继续流程，保持轨迹存活与时间前进
             output_results = np.zeros((0, 5), dtype=np.float32)

        self.frame_count += 1  

        # Post-process detections  
        if output_results.shape[1] == 5:  
            scores = output_results[:, 4]  
            bboxes = output_results[:, :4]  
        else:  
            output_results = output_results.cpu().numpy()  
            scores = output_results[:, 4] * output_results[:, 5]  
            bboxes = output_results[:, :4]  

        img_h, img_w = float(img_info[0]), float(img_info[1])  
        scale = min(float(img_size[0]) / img_h, float(img_size[1]) / img_w)  
        bboxes = bboxes / scale  
        dets = np.concatenate((bboxes.astype(np.float32), np.expand_dims(scores.astype(np.float32), axis=-1)), axis=1)  

        inds_low = scores > 0.1  
        inds_high = scores < self.det_thresh  
        inds_second = np.logical_and(inds_low, inds_high)  
        dets_second = dets[inds_second]  
        remain_inds = scores > self.det_thresh  
        dets = dets[remain_inds]  

        # Kalman predict (state evolution)  
        T = len(self.trackers)  
        to_del = []  
        kf_preds = []  
        for ti in range(T):  
            pos_kf_raw = self.trackers[ti].predict()[0]  
            pos_kf = np.asarray(pos_kf_raw, dtype=np.float32).reshape(-1)  
            if pos_kf.size > 4:  
                pos_kf = pos_kf[:4]  
            kf_preds.append(pos_kf.copy())  
            if np.any(np.isnan(pos_kf)):  
                to_del.append(ti)  

        # Clean invalid trackers and sync kf_preds  
        if len(to_del) > 0:  
            keep_mask = np.ones(T, dtype=bool)  
            keep_mask[to_del] = False  
            self.trackers = [trk for i, trk in enumerate(self.trackers) if keep_mask[i]]  
            kf_preds = [pred for i, pred in enumerate(kf_preds) if keep_mask[i]]  

        # Update T  
        T = len(self.trackers)  
        if T == 0:  
            for i in range(dets.shape[0]):  
                trk = KalmanBoxTracker(dets[i, :], delta_t=self.delta_t, ema_obs=self.ema_obs, ema_pred=self.ema_pred)  
                self.trackers.append(trk)  
            return np.empty((0, 5))  

        # Batched D2MP-OB predict with trust gating/fusion  
        preds_px = self._batched_predict(self.trackers, (img_info[0], img_info[1]), kf_preds=np.array(kf_preds, dtype=np.float32))  

        # trks for association  
        trks = np.zeros((T, 5), dtype=np.float32)  
        trks[:, :4] = preds_px  

        # Metadata for association  
        last_boxes = np.array([trk.last_observation for trk in self.trackers], dtype=np.float32)  
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers], dtype=np.float32)  
        bad_kp = (k_observations[:, :4] < 0).any(axis=1)  
        if bad_kp.any():  
            k_observations[bad_kp, :4] = last_boxes[bad_kp, :4]  
            if k_observations.shape[1] >= 5 and last_boxes.shape[1] >= 5:  
                k_observations[bad_kp, 4] = last_boxes[bad_kp, 4]  

        depth_emas = np.array([trk.depth_ema for trk in self.trackers], dtype=np.float32)  

        # velocities for association: last -> pred unit vector  
        cx_lo = (last_boxes[:, 0] + last_boxes[:, 2]) * 0.5  
        cy_lo = (last_boxes[:, 1] + last_boxes[:, 3]) * 0.5  
        cx_pr = (trks[:, 0] + trks[:, 2]) * 0.5  
        cy_pr = (trks[:, 1] + trks[:, 3]) * 0.5  
        dv = np.stack([cy_pr - cy_lo, cx_pr - cx_lo], axis=1).astype(np.float32)  
        nrm = np.linalg.norm(dv, axis=1, keepdims=True); nrm[nrm < 1e-6] = 1.0  
        velocities_assoc = dv / nrm  

        # ---------- First round: baseline associate ----------  
        matched, unmatched_dets, unmatched_trks = associate(  
            dets, trks, self.iou_threshold, velocities_assoc, k_observations, self.inertia  
        )  
        for m in matched:  
            self.trackers[m[1]].update(dets[m[0], :])  

        # ---------- Second round: BYTE + Soft-Depth ----------  
        if self.use_byte and dets_second.shape[0] > 0 and unmatched_trks.shape[0] > 0:  
            u_trks = trks[unmatched_trks]  
            iou_left = np.array(self.asso_func(dets_second, u_trks))  
            if iou_left.size > 0:  
                dets_second_xy = dets_second[:, :4]  
                u_trks_xy = u_trks[:, :4]  
                u_trk_emas = depth_emas[unmatched_trks]  
                A = self._depth_affinity(dets_second_xy, u_trks_xy, u_trk_emas, img_h)  
                iou_adj = self._blend_iou_with_depth(iou_left, A)  
            else:  
                iou_adj = iou_left  

            # thr2 = max(0.1, self.iou_threshold - 0.05)  
            thr2 = max(0.15, self.iou_threshold - 0.02)  
            
            
            if iou_adj.size > 0 and iou_adj.max() > thr2:  
                matched_indices = linear_assignment(-iou_adj)  
                to_remove_trk_indices = []  
                for m2 in matched_indices:  
                    det_local, trk_local = m2[0], m2[1]  
                    if iou_adj[det_local, trk_local] < thr2:  
                        continue  
                    trk_ind = unmatched_trks[trk_local]  
                    self.trackers[trk_ind].update(dets_second[det_local, :])  
                    to_remove_trk_indices.append(trk_ind)  
                if len(to_remove_trk_indices) > 0:  
                    unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))  

        # ---------- Third round: Re-association + Soft-Depth ----------  
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:  
            left_dets = dets[unmatched_dets]  
            left_trks_last = last_boxes[unmatched_trks]  
            left_trks_pred = trks[unmatched_trks]  

            iou_last = np.array(self.asso_func(left_dets, left_trks_last))  
            iou_pred = np.array(self.asso_func(left_dets, left_trks_pred))  
            iou_max = iou_last if iou_pred.size == 0 else np.maximum(iou_last, iou_pred)  

            if iou_max.size > 0:  
                dets_xy = left_dets[:, :4]  
                trks_xy = left_trks_pred[:, :4]  
                trk_emas = depth_emas[unmatched_trks]  
                A = self._depth_affinity(dets_xy, trks_xy, trk_emas, img_h)  
                iou_adj = self._blend_iou_with_depth(iou_max, A)  
            else:  
                iou_adj = iou_max  

            # thr3 = max(0.1, self.iou_threshold - 0.05)  
            thr3 = max(0.15, self.iou_threshold - 0.02)  
            if iou_adj.size > 0 and iou_adj.max() > thr3:  
                rematched_indices = linear_assignment(-iou_adj)  
                to_remove_det_indices = []  
                to_remove_trk_indices = []  
                for m3 in rematched_indices:  
                    det_local, trk_local = m3[0], m3[1]  
                    if iou_adj[det_local, trk_local] < thr3:  
                        continue  
                    det_ind = unmatched_dets[det_local]  
                    trk_ind = unmatched_trks[trk_local]  
                    self.trackers[trk_ind].update(dets[det_ind, :])  
                    to_remove_det_indices.append(det_ind)  
                    to_remove_trk_indices.append(trk_ind)  
                if len(to_remove_det_indices) > 0:  
                    unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))  
                if len(to_remove_trk_indices) > 0:  
                    unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))  

        # Unmatched trackers: no update  
        for m in unmatched_trks:  
            self.trackers[m].update(None)  

        # Create new trackers for unmatched detections  
        for i in unmatched_dets:  
            trk = KalmanBoxTracker(dets[i, :], delta_t=self.delta_t, ema_obs=self.ema_obs, ema_pred=self.ema_pred)  
            self.trackers.append(trk)  

        # Output and cleanup  
        i = len(self.trackers)  
        ret = []  
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
        self.frame_count += 1  
        det_scores = np.ones((dets.shape[0], 1), dtype=np.float32)  
        dets = np.concatenate((dets.astype(np.float32), det_scores), axis=1)  

        remain_inds = scores > self.det_thresh  
        cates = cates[remain_inds]  
        dets = dets[remain_inds]  

        trks = np.zeros((len(self.trackers), 5), dtype=np.float32)  
        to_del = []  
        ret = []  
        for t, trk in enumerate(trks):  
            pos = np.asarray(self.trackers[t].predict()[0], dtype=np.float32).reshape(-1)  
            cat = self.trackers[t].cate  
            trk[:] = np.array([float(pos[0]), float(pos[1]), float(pos[2]), float(pos[3]), float(cat)], dtype=np.float32)  
            if np.any(np.isnan(pos)):  
                to_del.append(t)  
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))  
        for t in reversed(to_del):  
            self.trackers.pop(t)  

        velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0, 0), dtype=np.float32) for trk in self.trackers], dtype=np.float32)  
        last_boxes = np.array([trk.last_observation for trk in self.trackers], dtype=np.float32)  
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers], dtype=np.float32)  

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
            cate_matrix = np.zeros((num_dets, num_trks), dtype=np.float32)  
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
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))  
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))  

        for i in unmatched_dets:  
            trk = KalmanBoxTracker(dets[i, :], ema_obs=self.ema_obs, ema_pred=self.ema_pred)  
            trk.cate = cates[i]  
            self.trackers.append(trk)  
        i = len(self.trackers)  

        for trk in reversed(self.trackers):  
            if trk.last_observation.sum() > 0:  
                d = trk.last_observation[:4]  
            else:  
                d = trk.get_state()[0]  
            if trk.time_since_update < 1:
                if (self.frame_count <= self.min_hits) or (trk.hit_streak >= self.min_hits):
                    ret.append(np.concatenate((d, [trk.id + 1], [trk.cate], [0])).reshape(1, -1))
                if trk.hit_streak == self.min_hits:
                    for prev_i in range(self.min_hits - 1):
                        prev_observation = trk.history_observations[-(prev_i + 2)]
                        ret.append(
                            np.concatenate(
                                (prev_observation[:4], [trk.id + 1], [trk.cate], [-(prev_i + 1)])
                            ).reshape(1, -1)
                        )
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 7))