# from tqdm import tqdm
# from millify import millify
# import tensorflow as tf
# import numpy as np
# import cv2
# from pathlib import Path
# from sklearn.neighbors import NearestNeighbors

# import cvde
# # from models.pvn3d_e2e import PVN3D_E2E
# # from losses.pvn_loss import PvnLoss

# from datasets.blender import ValBlender
# from datasets.simpose import Val6IMPOSE
# from datasets.linemod import LineMOD


# class Evaluate(cvde.job.Job):
#     def run(self):
#         job_cfg = self.config

#         self.num_validate = job_cfg["num_validate"]
#         blender_config = job_cfg["ValBlender"]
#         simpose_config = job_cfg["Val6IMPOSE"]

#         # model = PVN3D_E2E(**job_cfg["PVN3D_E2E"])

#         # datasets = {}

#         # eval_len = lambda x: len(x) // x.batch_size

#         # with tf.device("/cpu:0"):
#         #     if "ValBlender" in job_cfg["datasets"]:
#         #         try:
#         #             blender = ValBlender(**blender_config)
#         #             blender_tf = blender.to_tf_dataset()
#         #             datasets["blender"] = (blender_tf, eval_len(blender))
#         #             self.mesh_vertices = blender.mesh_vertices
#         #         except Exception as e:
#         #             print("Blender dataset not found, skipping...")
#         #             print(e)

#         #     if "Val6IMPOSE" in job_cfg["datasets"]:
#         #         try:
#         #             simpose = Val6IMPOSE(**simpose_config)
#         #             simpose_tf = simpose.to_tf_dataset()
#         #             datasets["simpose"] = (simpose_tf, eval_len(simpose))
#         #             self.mesh_vertices = simpose.mesh_vertices
#         #         except Exception as e:
#         #             print("Simpose dataset not found, skipping...")
#         #             print(e)

#         #     if "LineMOD" in job_cfg["datasets"]:
#         #         try:
#         #             linemod = LineMOD(**job_cfg["LineMOD"])
#         #             linemod_tf = linemod.to_tf_dataset()
#         #             datasets["linemod"] = (linemod_tf, eval_len(linemod))
#         #             self.mesh_vertices = linemod.mesh_vertices

#         #         except Exception as e:
#         #             print("LineMOD dataset not found, skipping...")
#         #             print(e)

#         # pairwise_distance = np.linalg.norm(
#         #     self.mesh_vertices[None, :, :] - self.mesh_vertices[:, None, :], axis=-1
#         # )
#         # self.object_diameter = np.max(pairwise_distance)

#         # path = Path(job_cfg["weights"])
#         # path = str(sorted(list(path.parent.glob(path.name)))[-1].resolve())
#         # print("Loading weights from: ", path)
#         # model.load_weights(path)

#         # loss_fn = PvnLoss(**job_cfg["PvnLoss"])

#         # for name, (dataset_tf, length) in datasets.items():
#         #     loss_vals = self.eval(name, model, dataset_tf, length, loss_fn)
#         #     if loss_vals is None:
#         #         return
#         #     res = self.get_visualization(name, model, dataset_tf)
#         #     if res == "abort":
#         #         return
#         #     for k, v in loss_vals.items():
#         #         self.tracker.log(f"{name}_{k}", v, 0)

#     def get_visualization(self, name: str, model: PVN3D_E2E, dataset_tf):
#         i = 0
#         bar = tqdm(total=self.num_validate, desc=f"Visualizing {name}")
#         for x, y in dataset_tf.take(self.num_validate):
#             if self.is_stopped():
#                 return "abort"

#             (
#                 b_rgb,
#                 _,
#                 b_intrinsics,
#                 b_roi,
#                 _,
#             ) = x

#             b_RT_gt, _ = y

#             (
#                 b_R,
#                 b_t,
#                 _,
#                 _,
#             ) = model(x, training=False)

#             h, w = tf.shape(b_rgb)[1], tf.shape(b_rgb)[2]
#             b_roi, b_crop_factor = model.get_crop_index(
#                 b_roi,
#                 h,
#                 w,
#                 model.resnet_input_shape[0],
#                 model.resnet_input_shape[1],
#             )

#             b_rgb = b_rgb.numpy()
#             b_roi = b_roi.numpy()
#             b_R = b_R.numpy()
#             b_t = b_t.numpy()
#             b_RT_gt = b_RT_gt.numpy()
#             b_intrinsics = b_intrinsics.numpy()

#             # load mesh
#             b_mesh = np.tile(self.mesh_vertices, (b_R.shape[0], 1, 1))
#             # transform mesh_vertices
#             b_mesh_pred = (
#                 np.matmul(b_mesh, b_R[:, :3, :3].transpose((0, 2, 1))) + b_t[:, tf.newaxis, :]
#             )
#             b_mesh_gt = (
#                 np.matmul(b_mesh, b_RT_gt[:, :3, :3].transpose((0, 2, 1)))
#                 + b_RT_gt[:, tf.newaxis, :3, 3]
#             )

#             b_mesh_pred = self.project_batch_to_image(b_mesh_pred, b_intrinsics)[..., :2].astype(
#                 np.int32
#             )
#             b_mesh_gt = self.project_batch_to_image(b_mesh_gt, b_intrinsics)[..., :2].astype(
#                 np.int32
#             )

#             for rgb, roi, mesh_vertices, mesh_vertices_gt in zip(
#                 b_rgb, b_roi, b_mesh_pred, b_mesh_gt
#             ):
#                 if self.is_stopped():
#                     bar.close()
#                     return

#                 vis_mesh = self.draw_object_mesh(rgb.copy(), roi, mesh_vertices, mesh_vertices_gt)
#                 self.tracker.log(name, vis_mesh, index=i)

#                 i = i + 1
#                 bar.update(1)
#                 if i >= self.num_validate:
#                     bar.close()
#                     return

#     def crop_to_roi(self, rgb, roi, margin=50):
#         (
#             y1,
#             x1,
#             y2,
#             x2,
#         ) = roi[:4]
#         h, w = rgb.shape[:2]
#         x1 = np.clip(x1 - margin, 0, w)
#         x2 = np.clip(x2 + margin, 0, w)
#         y1 = np.clip(y1 - margin, 0, h)
#         y2 = np.clip(y2 + margin, 0, h)
#         return rgb[y1:y2, x1:x2]

#     def draw_object_mesh(self, rgb, roi, mesh_vertices, mesh_vertices_gt):
#         h, w = rgb.shape[:2]
#         clipped_mesh_vertices = np.clip(mesh_vertices, 0, [w - 1, h - 1])
#         clipped_mesh_vertices_gt = np.clip(mesh_vertices_gt, 0, [w - 1, h - 1])
#         rgb_pred = rgb.copy()
#         rgb_gt = rgb.copy()
#         for x, y in clipped_mesh_vertices:
#             cv2.circle(rgb_pred, (x, y), 1, (0, 0, 255), -1)
#         for x, y in clipped_mesh_vertices_gt:
#             cv2.circle(rgb_gt, (x, y), 1, (0, 255, 0), -1)

#         rgb_pred = self.crop_to_roi(rgb_pred, roi)
#         rgb_gt = self.crop_to_roi(rgb_gt, roi)
#         # assemble images side-by-side
#         vis_mesh = np.concatenate([rgb_pred, rgb_gt], axis=1)
#         return vis_mesh

#     def project_batch_to_image(self, pts, b_intrinsics):
#         cam_cx, cam_cy = b_intrinsics[:, 0, 2], b_intrinsics[:, 1, 2]  # [b]
#         cam_fx, cam_fy = b_intrinsics[:, 0, 0], b_intrinsics[:, 1, 1]  # [b]

#         f = tf.stack([cam_fx, cam_fy], axis=1)  # [b, 2]
#         c = tf.stack([cam_cx, cam_cy], axis=1)  # [b, 2]

#         rank = tf.rank(pts)
#         insert_n_dims = rank - 2
#         for _ in range(insert_n_dims):
#             f = tf.expand_dims(f, axis=1)
#             c = tf.expand_dims(c, axis=1)

#         coors = pts[..., :2] / pts[..., 2:] * f + c
#         coors = tf.floor(coors)
#         return tf.concat([coors, pts[..., 2:]], axis=-1).numpy()

#     def eval(self, name, model, dataset, length, loss_fn):
#         loss_vals = {}
#         # self.mesh_vertices
#         ad = []
#         ads = []
#         for x, y in tqdm(
#             dataset,
#             desc=f"Validating {name}",
#             total=length,
#         ):
#             b_RT_gt, _ = y
#             if self.is_stopped():
#                 return None
#             batch_R, batch_t, _, pred = model(x, training=False)
#             l = loss_fn.call(y, pred)
#             for k, v in zip(["loss", "loss_cp", "loss_kp", "loss_seg"], l):
#                 loss_vals[k] = loss_vals.get(k, []) + [v]

#             # get batch homogeneous transformation matrix from batch_R and batch_t
#             average_distance, average_distance_s = self.calc_ad_ads(batch_R, batch_t, b_RT_gt)
#             ad.extend(average_distance)
#             ads.extend(average_distance_s)

#             # loss_vals["add"] = loss_vals.get("add", []) + [average_distance]
#             # loss_vals["add_s"] = loss_vals.get("add_s", []) + [average_distance_s]

#         for k, v in loss_vals.items():
#             loss_vals[k] = tf.reduce_mean(v)

#         threshold = 0.1 * self.object_diameter
#         ad = np.array(ad)
#         ads = np.array(ads)
#         ad = np.count_nonzero(ad < threshold) / len(ad)
#         ads = np.count_nonzero(ads < threshold) / len(ads)
#         loss_vals["add"] = np.array(ad * 100.0)
#         loss_vals["add_s"] = np.array(ads * 100.0)
#         return loss_vals

#     def calc_ad_ads(self, batch_R, batch_t, b_RT_gt):
#         b_mesh = np.tile(self.mesh_vertices, (b_RT_gt.shape[0], 1, 1))
#         # transform mesh_vertices
#         b_mesh_pred = (
#             np.matmul(b_mesh, tf.transpose(batch_R[:, :3, :3], (0, 2, 1)))
#             + batch_t[:, tf.newaxis, :]
#         )  # [bs, n_pts, 3]
#         b_mesh_gt = (
#             np.matmul(b_mesh, tf.transpose(b_RT_gt[:, :3, :3], (0, 2, 1)))
#             + b_RT_gt[:, tf.newaxis, :3, 3]
#         )  # [bs, n_pts, 3]

#         # average distance
#         distances = np.linalg.norm(b_mesh_pred - b_mesh_gt, axis=-1)  # [bs, n_pts]
#         ad = np.mean(distances, -1)  # [bs,]

#         # average minimum distance (for add-s)
#         distances = np.linalg.norm(
#             b_mesh_pred[:, :, np.newaxis] - b_mesh_gt[:, np.newaxis], axis=-1
#         )  # [bs, n_pts, n_pts]
#         ad_s = np.mean(np.min(distances, axis=-1), axis=-1)  # [bs]

#         return ad, ad_s
