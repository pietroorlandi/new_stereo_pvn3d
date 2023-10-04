from tqdm import tqdm
from millify import millify
import tensorflow as tf
import numpy as np
import cv2

import cvde
from losses.stereopvn3d_loss import StereoPvn3dLoss
from models.stereopvn3d import StereoPvn3d
from losses.stereopvn3d_loss import StereoPvn3dLoss



class TrainE2E(cvde.job.Job):
    def run(self):
        job_cfg = self.config

        self.num_validate = job_cfg["num_validate"]

        print("Running job: ", self.name)
        print(f"job_cfg['StereoPvn3d']: {job_cfg['StereoPvn3d']}")
        
        self.model = model = StereoPvn3d(**job_cfg["StereoPvn3d"])
        # self.loss = StereoPvn3dLoss(**job_cfg["StereoPvn3dLoss"])

        if job_cfg["dataset"] == 'blender':
            train_config = job_cfg["TrainBlender"]
            val_config = job_cfg["ValBlender"]
            #from datasets.blender import TrainBlender, ValBlender
            #train_gen = TrainBlender
            #val_gen = ValBlender
        elif job_cfg["dataset"] == '6IMPOSE':
            train_config = job_cfg["Train6IMPOSE"]
            val_config = job_cfg["Val6IMPOSE"]
            from datasets.simpose import Train6IMPOSE, Val6IMPOSE
            train_gen = Train6IMPOSE
            val_gen = Val6IMPOSE

        train_set = train_gen(**train_config)
        val_set = val_gen(**val_config)
        self.demo_set = val_gen(**val_config)
        self.demo_set_tf = self.demo_set.to_tf_dataset().take(self.num_validate)
        self.mesh_vertices = val_set.mesh_vertices

        self.loss_fn = loss_fn = StereoPvn3dLoss(**job_cfg["StereoPvn3dLoss"])
        optimizer = tf.keras.optimizers.Adam(**job_cfg["Adam"])

        if 'weights' in job_cfg:
            self.model.load_weights(job_cfg['weights'])

        
        self.log_visualization(-1)

        

        num_epochs = job_cfg["epochs"]
        train_set_tf = train_set.to_tf_dataset()
        val_set_tf = val_set.to_tf_dataset()
        cumulative_steps = 0

        
        for epoch in range(num_epochs):
            bar = tqdm(
                total=len(train_set) // train_set.batch_size,
                desc=f"Train ({epoch}/{num_epochs-1})",
            )

            loss_vals = {}
            for x, y in train_set_tf:
                print(f"len(x): {len(x)}")
                print(f"len(y): {len(y)}")

                if self.is_stopped():
                    return
                with tf.GradientTape() as tape:
                    pred = model(x, training=True)
                    loss_combined, mse_loss, mlse_loss, ssim_loss, loss_cp, loss_kp, loss_seg = loss_fn.call(y, pred)
                gradients = tape.gradient(loss_combined, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                loss_vals["loss"] = loss_vals.get("loss", []) + [loss_combined.numpy()]
                loss_vals["mse_loss"] = loss_vals.get("mse_loss", []) + [mse_loss.numpy()]
                loss_vals["mlse_loss"] = loss_vals.get("mlse_loss", []) + [mlse_loss.numpy()]
                loss_vals["ssim_loss"] = loss_vals.get("ssim_loss", []) + [ssim_loss.numpy()]
                loss_vals["loss_cp"] = loss_vals.get("loss_cp", []) + [loss_cp.numpy()]
                loss_vals["loss_kp"] = loss_vals.get("loss_kp", []) + [loss_kp.numpy()]
                loss_vals["loss_seg"] = loss_vals.get("loss_seg", []) + [loss_seg.numpy()]

                bar.update(1)  # update by batchsize
                bar.set_postfix(
                    {
                        "loss": millify(loss_combined.numpy(), precision=4),
                    }
                )
                cumulative_steps+=1

            bar.close()
            model.save(f"checkpoints/{self.tracker.name}/model_{epoch:02}", save_format="tf")

            for k, v in loss_vals.items():
                loss_vals[k] = tf.reduce_mean(v)
                self.tracker.log(f"train_{k}", loss_vals[k], epoch)

            self.log_visualization(epoch)

            loss_vals = {}
            for x, y in tqdm(
                val_set_tf,
                desc=f"Val ({epoch}/{num_epochs-1})",
                total=len(val_set) // val_set.batch_size,
            ):

                _, _, _, pred = model(x, training=False)
                print(f"len(pred): {len(pred)}")

                l = loss_fn.call(y, pred)
                for k, v in zip(["loss","mlse_loss","ssim_loss","loss_cp", "loss_kp", "loss_seg"], l):
                    loss_vals[k] = loss_vals.get(k, []) + [v]

            for k, v in loss_vals.items():
                loss_vals[k] = tf.reduce_mean(v)
                self.tracker.log(f"val_{k}", loss_vals[k], epoch)

    def log_visualization(self, epoch: int, logs=None):
        i = 0
        for x, y in self.demo_set_tf:
            (
                b_rgb,
                b_rgb_R,
                b_baseline,
                b_intrinsics,
                b_roi,
                b_mesh_kpts,
            ) = x

            b_depth_gt, b_RT_gt, b_mask_gt = y

            (
                b_R,
                b_t,
                b_kpts_pred,
                (b_depth_pred, b_kp_offset_pred, b_seg_pred, b_cp_offset_pred, b_xyz_pred, b_sampled_inds_in_original_image, b_mesh_kpts, b_norm_bbox, b_cropped_rgbs_l, b_cropped_rgbs_r, b_weights),
            ) = self.model(x, training=False)
            # b_depth_pred, b_kp_offset_pred, b_mask_selected_pred, b_cp_offset_pred, b_norm_bbox = self.model(x, training=False)

            h, w = tf.shape(b_rgb)[1], tf.shape(b_rgb)[2]

            # get xyz gt
            b_xyz_gt = self.model.pcld_processor_tf_by_index(b_depth_gt, b_intrinsics, b_sampled_inds_in_original_image)
            b_mask_gt = tf.expand_dims(b_mask_gt, axis=-1) # better to move it to simpose.py
            b_mask_selected = tf.gather_nd(b_mask_gt, b_sampled_inds_in_original_image)
            # get gt offset
            b_kp_offsets_gt, b_cp_offsets_gt = self.loss_fn.get_offst(b_RT_gt, b_xyz_gt, b_mask_selected, b_mesh_kpts)

            # crop b_depth_gt
            crop_h, crop_w = b_depth_pred.shape[1:3]
        
            # crop_depth_gt = full_depth_gt[y1:y2, x1:x2]
            b_depth_gt = tf.image.crop_and_resize(
                tf.cast(b_depth_gt, tf.float32),
                b_norm_bbox,
                tf.range(tf.shape(b_depth_gt)[0]),
                [crop_h, crop_w]
            )

            # attention weights
            [batch_w1, batch_w2, batch_w3, batch_w4, batch_w5] = b_weights


            b_rgb = b_rgb.numpy()
            b_roi = b_roi.numpy()
            b_mask = b_mask_gt.numpy()
            b_seg_pred = b_seg_pred.numpy()
            b_sampled_inds = b_sampled_inds_in_original_image.numpy()
            b_kp_offset = b_kp_offset_pred.numpy()
            b_cropped_rgbs_l = b_cropped_rgbs_l.numpy()
            b_cropped_rgbs_r = b_cropped_rgbs_r.numpy()
            b_cp_offset = b_cp_offset_pred.numpy()
            b_R = b_R.numpy()
            b_t = b_t.numpy()
            b_RT_gt = b_RT_gt.numpy()
            b_intrinsics = b_intrinsics.numpy()
            b_depth_pred = b_depth_pred.numpy()
            b_depth_gt = b_depth_gt.numpy()
            # b_xyz_gt = b_xyz_gt.numpy()
            #b_crop_factor = b_crop_factor.numpy()


            # get gt offsets
            b_offsets_pred = np.concatenate([b_kp_offset, b_cp_offset], axis=2)
            b_offsets_gt = np.concatenate([b_kp_offsets_gt, b_cp_offsets_gt], axis=2)

            # keypoint vectors
            b_kpts_vectors_pred = b_xyz_pred[:, :, None, :].numpy() + b_offsets_pred  # [b, n_pts, 9, 3]
            b_kpts_vectors_gt = b_xyz_gt[:, :, None, :].numpy() + b_offsets_gt

            # keypoints
            b_kpts_gt = (
                tf.matmul(b_mesh_kpts, tf.transpose(b_RT_gt[:, :3, :3], (0, 2, 1)))
                + b_RT_gt[:, tf.newaxis, :3, 3]
            )

            # load mesh
            b_mesh = np.tile(self.mesh_vertices, (b_R.shape[0], 1, 1))
            # transform mesh_vertices
            b_mesh_pred = (
                np.matmul(b_mesh, b_R[:, :3, :3].transpose((0, 2, 1))) + b_t[:, tf.newaxis, :]
            )
            b_mesh_gt = (
                np.matmul(b_mesh, b_RT_gt[:, :3, :3].transpose((0, 2, 1)))
                + b_RT_gt[:, tf.newaxis, :3, 3]
            )

            b_mesh_pred = self.project_batch_to_image(b_mesh_pred, b_intrinsics)[..., :2].astype(
                np.int32
            )
            b_mesh_gt = self.project_batch_to_image(b_mesh_gt, b_intrinsics)[..., :2].astype(
                np.int32
            )
            
            b_kpts_gt = self.project_batch_to_image(b_kpts_gt, b_intrinsics)
            print(f"b_kpts_pred before project_batch_to_image: {b_kpts_pred}")
            b_kpts_pred = self.project_batch_to_image(b_kpts_pred, b_intrinsics)
            print(f"b_kpts_pred after project_batch_to_image: {b_kpts_pred}")
            b_kpts_vectors_gt = self.project_batch_to_image(b_kpts_vectors_gt, b_intrinsics)
            b_kpts_vectors_pred = self.project_batch_to_image(b_kpts_vectors_pred, b_intrinsics)
            b_xyz_projected_pred = self.project_batch_to_image(b_xyz_pred, b_intrinsics)  # [b, n_pts, 3]
            b_xyz_projected_gt = self.project_batch_to_image(b_xyz_gt, b_intrinsics)  # [b, n_pts, 3]



            for (
                rgb,
                crop_rgb,
                crop_rgb_r,
                crop_depth_pred,
                crop_depth_gt,
                norm_bbox,
                roi,
                seg_pred,
                sampled_inds,
                kpts_gt,
                kpts_pred,
                mesh_vertices,
                mesh_vertices_gt,
                offsets_pred,
                offsets_gt,
                # crop_factor,
                kpts_vectors_pred,
                kpts_vectors_gt,
                xyz_projected_pred,
                xyz_projected_gt,
                w1,
                w2,
                w3,
                w4,
                w5,
            ) in zip(
                b_rgb,
                b_cropped_rgbs_l,
                b_cropped_rgbs_r,
                b_depth_pred,
                b_depth_gt,
                b_norm_bbox,
                b_roi,
                b_seg_pred,
                b_sampled_inds,
                b_kpts_gt,
                b_kpts_pred,
                b_mesh_pred,
                b_mesh_gt,
                b_offsets_pred,
                b_offsets_gt,
                # b_crop_factor,
                b_kpts_vectors_pred,
                b_kpts_vectors_gt,
                b_xyz_projected_pred,
                b_xyz_projected_gt,
                batch_w1,
                batch_w2,
                batch_w3,
                batch_w4,
                batch_w5,
            ):
 
                vis_seg = self.draw_segmentation(rgb.copy(), sampled_inds, seg_pred, roi)
                self.tracker.log(f"RGB ({i})", vis_seg, index=epoch)

                vis_depth = self.draw_depth(crop_depth_pred, crop_depth_gt)
                self.tracker.log(f"Depth ({i})", vis_depth, index=epoch)

                self.tracker.log(f"RGB R (crop) ({i})", crop_rgb_r.astype(np.uint8), index=epoch)
                self.tracker.log(f"RGB L (crop) ({i})", crop_rgb.astype(np.uint8), index=epoch)


                vis_mesh = self.draw_object_mesh(rgb.copy(), roi, mesh_vertices, mesh_vertices_gt)
                self.tracker.log(f"RGB (mesh) ({i})", vis_mesh, index=epoch)


                print(f'kpts_gt {kpts_gt} - kpts_gt {kpts_gt.shape}')
                print(f'kpts_pred {kpts_pred} - kpts_pred {kpts_pred.shape}')

                vis_kpts = self.draw_keypoint_correspondences(rgb.copy(), roi, kpts_gt, kpts_pred)
                self.tracker.log(f"RGB (kpts) ({i})", vis_kpts, index=epoch)
                print(f'crop_rgb_r.shape {crop_rgb_r.shape}')

                #self.tracker.log(f"RGBR ({i})", crop_rgb_r, index=epoch)

                vis_offsets = self.draw_keypoint_offsets(
                    rgb.copy(),
                    roi,
                    offsets_pred,
                    sampled_inds,
                    kpts_pred,
                    seg=seg_pred,
                    radius=1, # int(crop_factor),
                )
                vis_offsets_gt = self.draw_keypoint_offsets(
                    rgb.copy(),
                    roi,
                    offsets_gt,
                    sampled_inds,
                    kpts_gt,
                    radius=1, # int(crop_factor),
                )
                # assemble images side-by-side
                vis_offsets = np.concatenate([vis_offsets, vis_offsets_gt], axis=1)
                self.tracker.log(f"RGB (offsets) ({i})", vis_offsets, index=epoch)

                #vis_vectors_gt = self.draw_keypoint_vectors(rgb.copy(), roi, offsets_gt, kpts_gt, xyz_projected_gt, kpts_vectors_gt
                #)

                # vis_vectors = self.draw_keypoint_vectors(
                #     rgb.copy(),
                #     roi,
                #     offsets_pred,
                #     kpts_pred,
                #     xyz_projected,
                #     kpts_vectors_pred,
                #     seg=seg_pred,
                # )

                # # assemble images side-by-side
                # vis_vectors = np.concatenate([vis_vectors, vis_vectors_gt], axis=1)
                # self.tracker.log(f"RGB (vectors) ({i})", vis_vectors, index=epoch)


                vis_w1 = self.draw_weights(w1)
                self.tracker.log(f"Weights 1 ({i})", vis_w1, index=epoch)

                vis_w2 = self.draw_weights(w2)
                self.tracker.log(f"Weights 2 ({i})", vis_w2, index=epoch)

                vis_w3 = self.draw_weights(w3)
                self.tracker.log(f"Weights 3 ({i})", vis_w3, index=epoch)

                vis_w4 = self.draw_weights(w4)
                self.tracker.log(f"Weights 4 ({i})", vis_w4, index=epoch)

                vis_w5 = self.draw_weights(w5)
                self.tracker.log(f"Weights 5 ({i})", vis_w5, index=epoch)



                i = i + 1
                if i >= self.num_validate:
                    return

    def crop_to_roi(self, rgb, roi, margin=50):
        (
            y1,
            x1,
            y2,
            x2,
        ) = roi[:4]
        h, w = rgb.shape[:2]
        x1 = np.clip(x1 - margin, 0, w)
        x2 = np.clip(x2 + margin, 0, w)
        y1 = np.clip(y1 - margin, 0, h)
        y2 = np.clip(y2 + margin, 0, h)
        return rgb[y1:y2, x1:x2]

    def draw_segmentation(self, rgb, sampled_inds, seg_pred, roi):
        (
            y1,
            x1,
            y2,
            x2,
        ) = roi[:4]
        cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # draw gray cirlces at sampled inds
        # sampled_inds: [num_points, 3]
        for (_, h_ind, w_ind), seg in zip(sampled_inds, seg_pred):
            color = (0, 0, 255) if seg > 0.5 else (255, 0, 0)
            cv2.circle(rgb, (w_ind, h_ind), 1, color, -1)
        return rgb
    
    def draw_depth(self, crop_depth_pred, crop_depth_gt):
        h, w = crop_depth_pred.shape[:2]
        assembled = np.zeros((h, 2 * w, 3), dtype=np.uint8)
        color_depth = lambda d: cv2.applyColorMap(
            cv2.convertScaleAbs(d, alpha=255 / 1.5), cv2.COLORMAP_JET
        )

        colored_gt = color_depth(crop_depth_gt)
        colored_pred = color_depth(crop_depth_pred)

        assembled[:, :w, :] = colored_gt
        assembled[:, w:, :] = colored_pred
        print(f"crop_depth_pred.shape: {crop_depth_pred.shape}")
        print(f"crop_depth_gt.shape: {crop_depth_gt.shape}")
        print(f"assembled.shape: {assembled.shape}")

        return assembled

    def draw_object_mesh(self, rgb, roi, mesh_vertices, mesh_vertices_gt):
        h, w = rgb.shape[:2]
        clipped_mesh_vertices = np.clip(mesh_vertices, 0, [w - 1, h - 1])
        clipped_mesh_vertices_gt = np.clip(mesh_vertices_gt, 0, [w - 1, h - 1])
        rgb_pred = rgb.copy()
        rgb_gt = rgb.copy()
        for x, y in clipped_mesh_vertices:
            cv2.circle(rgb_pred, (x, y), 1, (0, 0, 255), -1)
        for x, y in clipped_mesh_vertices_gt:
            cv2.circle(rgb_gt, (x, y), 1, (0, 255, 0), -1)

        rgb_pred = self.crop_to_roi(rgb_pred, roi)
        rgb_gt = self.crop_to_roi(rgb_gt, roi)
        # assemble images side-by-side
        vis_mesh = np.concatenate([rgb_pred, rgb_gt], axis=1)
        return vis_mesh

    def draw_keypoint_correspondences(self, rgb, roi, kpts_gt, kpts_pred):
        # normalize z_coords of keypoints to [0, 1]
        kpts_gt[..., 2] = (kpts_gt[..., 2] - np.min(kpts_gt[..., 2])) / (
            np.max(kpts_gt[..., 2]) - np.min(kpts_gt[..., 2])
        )
        kpts_pred[..., 2] = (kpts_pred[..., 2] - np.min(kpts_pred[..., 2])) / (
            np.max(kpts_pred[..., 2]) - np.min(kpts_pred[..., 2])
        )
        print(f'In draw_keypoint_correspondences kpts_pred[..., 2]: {kpts_pred[..., 2]}')

        for (x_gt, y_gt, z_gt), (x_pred, y_pred, z_pred) in zip(kpts_gt, kpts_pred):
            gt_color = np.array((0, 255, 0), dtype=np.uint8)
            pred_color = np.array((255, 0, 0), dtype=np.uint8)

            scale_marker = lambda z: 10 + int(z * 20)

            cv2.drawMarker(
                rgb,
                (int(x_gt), int(y_gt)),
                gt_color.tolist(),
                cv2.MARKER_CROSS,
                scale_marker(z_gt),
                1,
            )
            cv2.drawMarker(
                rgb,
                (int(x_pred), int(y_pred)),
                pred_color.tolist(),
                cv2.MARKER_TILTED_CROSS,
                scale_marker(z_pred),
                1,
            )
            cv2.line(
                rgb,
                (int(x_gt), int(y_gt)),
                (int(x_pred), int(y_pred)),
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return self.crop_to_roi(rgb, roi)

    def project_batch_to_image(self, pts, b_intrinsics):
        cam_cx, cam_cy = b_intrinsics[:, 0, 2], b_intrinsics[:, 1, 2]  # [b]
        cam_fx, cam_fy = b_intrinsics[:, 0, 0], b_intrinsics[:, 1, 1]  # [b]

        f = tf.stack([cam_fx, cam_fy], axis=1)  # [b, 2]
        c = tf.stack([cam_cx, cam_cy], axis=1)  # [b, 2]

        rank = tf.rank(pts)
        insert_n_dims = rank - 2
        for _ in range(insert_n_dims):
            f = tf.expand_dims(f, axis=1)
            c = tf.expand_dims(c, axis=1)

        coors = pts[..., :2] / pts[..., 2:] * f + c
        coors = tf.floor(coors)
        return tf.concat([coors, pts[..., 2:]], axis=-1).numpy()

    def draw_keypoint_offsets(self, rgb, roi, offsets, sampled_inds, kpts_gt, radius=1, seg=None):
        cropped_rgb = self.crop_to_roi(rgb, roi, margin=10)
        h, w = cropped_rgb.shape[:2]
        vis_offsets = np.zeros((h * 3, w * 3, 3), dtype=np.uint8).astype(np.uint8)

        if seg is None:
            seg = [None] * len(sampled_inds)

        for i in range(9):
            # offset_view = np.zeros_like(rgb, dtype=np.uint8)
            offset_view = rgb.copy() // 3

            # get color hue from offset
            hue = np.arctan2(offsets[:, i, 1], offsets[:, i, 0]) / np.pi
            hue = (hue + 1) / 2
            hue = (hue * 180).astype(np.uint8)
            # value = np.ones_like(hue) * 255
            value = (np.linalg.norm(offsets[:, i, :], axis=-1) / 0.1 * 255).astype(np.uint8)
            hsv = np.stack([hue, np.ones_like(hue) * 255, value], axis=-1)
            colors_offset = cv2.cvtColor(hsv[None], cv2.COLOR_HSV2RGB).astype(np.uint8)[0]

            for (_, h_ind, w_ind), color, seg_ in zip(sampled_inds, colors_offset, seg):
                if seg_ is None or seg_ > 0.5:
                    cv2.circle(
                        offset_view,
                        (w_ind, h_ind),
                        radius,
                        tuple(map(int, color)),
                        -1,
                    )

            # mark correct keypoint
            cv2.drawMarker(
                offset_view,
                (int(kpts_gt[i, 0]), int(kpts_gt[i, 1])),
                (0, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=1,
            )

            offset_view = self.crop_to_roi(offset_view, roi, margin=10)
            vis_offsets[
                h * (i // 3) : h * (i // 3 + 1), w * (i % 3) : w * (i % 3 + 1)
            ] = offset_view
        return vis_offsets

    def draw_keypoint_vectors(
        self, rgb, roi, offsets, kpts, xyz_projected, keypoints_from_pcd, seg=None
    ):
        cropped_rgb = self.crop_to_roi(rgb, roi, margin=10)
        h, w = cropped_rgb.shape[:2]
        vis_offsets = np.zeros((h * 3, w * 3, 3), dtype=np.uint8).astype(np.uint8)

        if seg is None:
            seg = [None] * len(xyz_projected)

        for i in range(9):
            # offset_view = np.zeros_like(rgb, dtype=np.uint8)
            offset_view = rgb.copy() // 3

            # get color hue from offset
            hue = np.arctan2(offsets[:, i, 1], offsets[:, i, 0]) / np.pi
            hue = (hue + 1) / 2
            hue = (hue * 180).astype(np.uint8)
            # value = np.ones_like(hue) * 255
            value = (np.linalg.norm(offsets[:, i, :], axis=-1) / 0.1 * 255).astype(np.uint8)
            hsv = np.stack([hue, np.ones_like(hue) * 255, value], axis=-1)
            colors_offset = cv2.cvtColor(hsv[None], cv2.COLOR_HSV2RGB).astype(np.uint8)[0]

            for start, target, color, seg_ in zip(
                xyz_projected, keypoints_from_pcd[:, i, :], colors_offset, seg
            ):
                # over all pcd points
                if seg_ is None or seg_ > 0.5:
                    cv2.line(
                        offset_view,
                        tuple(map(int, start[:2])),
                        tuple(map(int, target[:2])),
                        tuple(map(int, color)),
                        1,
                    )

            cv2.drawMarker(
                offset_view,
                (int(kpts[i, 0]), int(kpts[i, 1])),
                (0, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=1,
            )

            offset_view = self.crop_to_roi(offset_view, roi, margin=10)
            vis_offsets[
                h * (i // 3) : h * (i // 3 + 1), w * (i % 3) : w * (i % 3 + 1)
            ] = offset_view
        return vis_offsets

    def draw_weights(self, attention):
        color_attention = lambda d: cv2.applyColorMap(
                    cv2.convertScaleAbs(d, alpha=255 / 1.5), cv2.COLORMAP_JET
                )
        h_att, w_att = attention[:,:,0].shape[:2]
        assembled_att1 = np.zeros((h_att, 3 * w_att, 3), dtype=np.uint8)
        colored_attention1 = color_attention(attention[:,:,0].numpy())
        colored_attention2 = color_attention(attention[:,:,1].numpy())
        colored_attention3 = color_attention(attention[:,:,2].numpy())

        assembled_att1[:, :w_att, :] = colored_attention1
        assembled_att1[:, w_att:2*w_att, :] = colored_attention2
        assembled_att1[:, w_att*2:3*w_att, :] = colored_attention3
        return assembled_att1