from tqdm import tqdm
from millify import millify
import tensorflow as tf
import numpy as np
import cv2

import cvde
from losses.stereopvn3d_loss import StereoPvn3dLoss
from models.stereo_pvn3d_e2e import StereoPvn3dE2E
# from losses.stereopvn3d_loss import StereoPvn3dLoss


from models.pprocessnet import _InitialPoseModel


class TrainE2E(cvde.job.Job):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs) #super().__init__()
        self.gt_depth = None
        #self.image_index = 0

    def run(self):
        job_cfg = self.config

        self.num_validate = job_cfg["num_validate"]
        self.freeze = job_cfg['freeze']
        print("Running job: ", self.name)
        print(f"job_cfg['StereoPvn3dE2E']: {job_cfg['StereoPvn3dE2E']}")
        
        self.model = model = StereoPvn3dE2E(**job_cfg["StereoPvn3dE2E"])
        # print('model.alpha1', model.alpha1)
        # self.loss = StereoPvn3dLoss(**job_cfg["StereoPvn3dLoss"])

        if job_cfg["dataset"] == 'blender':
            train_config = job_cfg["TrainBlender"]
            val_config = job_cfg["ValBlender"]
            # from datasets.blender import TrainBlender, ValBlender
            # train_gen = TrainBlender
            # val_gen = ValBlender
        elif job_cfg["dataset"] == '6IMPOSE':
            train_config = job_cfg["Train6IMPOSE"]
            val_config = job_cfg["Val6IMPOSE"]
            from datasets.simpose import Train6IMPOSE, Val6IMPOSE
            train_gen = Train6IMPOSE
            val_gen = Val6IMPOSE
        elif job_cfg["dataset"] == "sptfrecord":
            train_config = job_cfg["TrainSPTFRecord"]
            val_config = job_cfg["ValSPTFRecord"]
            from datasets.sp_tfrecord import TrainSPTFRecord, ValSPTFRecord

            train_gen = TrainSPTFRecord
            val_gen = ValSPTFRecord
            

        train_set = train_gen(**train_config)
        # val_set = train_gen(**train_config)
        # self.demo_set = train_gen(**train_config)
        val_set = val_gen(**val_config)
        self.demo_set = val_gen(**val_config)
        self.demo_set_tf = self.demo_set.to_tf_dataset().take(self.num_validate)
        self.mesh_vertices = val_set.mesh_vertices
        self.loss_fn = loss_fn = StereoPvn3dLoss(**job_cfg["StereoPvn3dLoss"])
        optimizer = tf.keras.optimizers.Adam(**job_cfg["Adam"])

        if 'weights' in job_cfg:
            self.model.load_weights(job_cfg['weights'])

        
        # self.log_visualization(-1)

        self.initial_pose_model = _InitialPoseModel()

        num_epochs = job_cfg["epochs"]
        train_set_tf = train_set.to_tf_dataset()
        val_set_tf = val_set.to_tf_dataset()

        cumulative_steps = 0
        number_samples_corrupted = 0 # for statistics on dataset
        frequency_plot_results_initially = 500
        frequency_plot_results = frequency_plot_results_initially*10
        
        for epoch in range(num_epochs):
            bar = tqdm(
                total=len(train_set) // train_set.batch_size,
                desc=f"Train ({epoch}/{num_epochs-1})",
            )
            # model.save(f"checkpoints/{self.tracker.name}/model_{epoch:03}", save_format="tf")
            # return

            print(f'Starting epoch {epoch} ----------------------------------------')

            loss_vals = {}
            for x, y in train_set_tf:
                if self.is_stopped():
                    return
                #print(f'steps: {cumulative_steps*4} -  {cumulative_steps*4 +1} - {cumulative_steps*4 +2} - {cumulative_steps*4 +3} ')
                gt_depth = y[0]
                #self.image_index+=1
                baseline = x[2]
                K = x[3][0]
                focal_length = K[0,0]
                gt_disp = tf.math.divide_no_nan(baseline * focal_length, gt_depth)
                y = (y[0], y[1], y[2], gt_disp)
                b_rgb_l = x[0]
                b_roi = x[4]
                # print('len of model.layers', len(model.layers))
                for layer in model.layers[:self.freeze]:
                    print('layer: ', layer)
                    layer.trainable = False
                batch_is_valid = tf.reduce_all(tf.reduce_all(tf.not_equal(b_roi, 0), axis=1)) # avoid cases where roi == (0,0,0,0)
                if batch_is_valid:
                    with tf.GradientTape() as tape:
                        pred = model(x, training=True)
                        loss_combined, mse_loss, mlse_loss, mae_normals, loss_cp, loss_kp, loss_seg, loss_he, loss_dpt_emb = loss_fn.call(y, pred, model.s1, model.s2, model.s3, model.s4, model.s5) # , loss_pcld
                        print(f'loss_combined before applying backprop {loss_combined}')
                    gradients = tape.gradient(loss_combined, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                    print('s1: ', model.s1)
                    print('s2: ', model.s2)
                    print('s3: ', model.s3)
                    print('s4: ', model.s4)
                    print('s5: ', model.s5)
                    loss_vals["loss"] = loss_vals.get("loss", []) + [loss_combined.numpy()]
                    loss_vals["mse_loss"] = loss_vals.get("mse_loss", []) + [mse_loss.numpy()]
                    loss_vals["mlse_loss"] = loss_vals.get("mlse_loss", []) + [mlse_loss.numpy()]
                    loss_vals["mae_normals"] = loss_vals.get("mae_normals", []) + [mae_normals.numpy()]
                    # loss_vals["deriv_loss"] = loss_vals.get("deriv_loss", []) + [deriv_loss.numpy()]
                    loss_vals["loss_cp"] = loss_vals.get("loss_cp", []) + [loss_cp.numpy()]
                    loss_vals["loss_kp"] = loss_vals.get("loss_kp", []) + [loss_kp.numpy()]
                    loss_vals["loss_seg"] = loss_vals.get("loss_seg", []) + [loss_seg.numpy()]
                    loss_vals["loss_he"] = loss_vals.get("loss_he", []) + [loss_he.numpy()]
                    loss_vals["loss_dpt_emb"] = loss_vals.get("loss_dpt_emb", []) + [loss_dpt_emb.numpy()]

                    # loss_vals["loss_pcld"] = loss_vals.get("loss_pcld", []) + [loss_pcld.numpy()]


                    bar.update(1)  # update by batchsize
                    bar.set_postfix(
                        {
                            "loss": millify(loss_combined.numpy(), precision=4),
                        }
                    )
                    if cumulative_steps == 0:
                        print(f"Summary of the model: {model.summary()}")
                    #weights_resnet_encoder = model.get_layer('resnet_model').get_weights()[0]
                    # For the first 2000 steps visualization each 500 steps, then less frequent to save disk usage
                    if cumulative_steps < 2000:
                        if cumulative_steps%frequency_plot_results_initially == 0:
                            self.log_visualization(cumulative_steps//frequency_plot_results_initially)
                    else:
                        if cumulative_steps%frequency_plot_results == 0:
                            self.log_visualization(cumulative_steps//frequency_plot_results_initially)

                    cumulative_steps+=1
                else:
                    print(f"batch {cumulative_steps} is not valid")
                    cumulative_steps+=1
                    number_samples_corrupted+=1
                    print(f"number of corrupted batch until now: {number_samples_corrupted}")

            bar.close()

            model.save(f"checkpoints/{self.tracker.name}/model_{epoch:03}", save_format="tf")
            print(f"Saved model for epoch {epoch:03}")

            for k, v in loss_vals.items():
                loss_vals[k] = tf.reduce_mean(v)
                self.tracker.log(f"train_{k}", loss_vals[k], epoch)
            

            # self.log_visualization(epoch)

            loss_vals = {}
            corrupted_batch_validation = 0
            print(' Start validation phase ---------------------------------------------------------------')
            for x, y in tqdm(
                val_set_tf,
                desc=f"Val ({epoch}/{num_epochs-1})",
                total=len(val_set) // val_set.batch_size,
            ):

                
                #print(f"len(pred): {len(pred)}")

                gt_depth = y[0]
                baseline = x[2]
                K = x[3][0]
                focal_length = K[0,0]
                b_roi = x[4]

                batch_is_valid = tf.reduce_all(tf.reduce_all(tf.not_equal(b_roi, 0), axis=1)) # avoid cases where roi == (0,0,0,0)
                if batch_is_valid:
    
                    gt_disp = tf.math.divide_no_nan(baseline * focal_length, gt_depth)
                    y = (y[0], y[1], y[2], gt_disp)
                    _, _, _, pred = model(x, training=False)
                    loss_combined, mse_loss, mlse_loss, mae_normals, loss_cp, loss_kp, loss_seg, loss_he, loss_dpt_emb = loss_fn.call(y, pred, model.s1, model.s2, model.s3, model.s4, model.s5)
                    loss_vals["loss"] = loss_vals.get("loss", []) + [loss_combined.numpy()]
                    loss_vals["mse_loss"] = loss_vals.get("mse_loss", []) + [mse_loss.numpy()]
                    loss_vals["mlse_loss"] = loss_vals.get("mlse_loss", []) + [mlse_loss.numpy()]
                    loss_vals["mae_normals"] = loss_vals.get("mae_normals", []) + [mae_normals.numpy()]
                    # loss_vals["deriv_loss"] = loss_vals.get("deriv_loss", []) + [deriv_loss.numpy()]
                    loss_vals["loss_cp"] = loss_vals.get("loss_cp", []) + [loss_cp.numpy()]
                    loss_vals["loss_kp"] = loss_vals.get("loss_kp", []) + [loss_kp.numpy()]
                    loss_vals["loss_seg"] = loss_vals.get("loss_seg", []) + [loss_seg.numpy()]
                    loss_vals["loss_he"] = loss_vals.get("loss_he", []) + [loss_he.numpy()]
                    loss_vals["loss_dpt_emb"] = loss_vals.get("loss_dpt_emb", []) + [loss_dpt_emb.numpy()]

                else:
                    corrupted_batch_validation+=1
            print(f"total number of corrupted_batch_validation: {corrupted_batch_validation}")

            for k, v in loss_vals.items():
                loss_vals[k] = tf.reduce_mean(v)
                self.tracker.log(f"val_{k}", loss_vals[k], epoch)

    def log_visualization(self, epoch: int, logs=None):
        i = 0
        sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
        sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)

        sobel_x = tf.expand_dims(tf.expand_dims(sobel_x, axis=2), axis=3)
        sobel_y = tf.expand_dims(tf.expand_dims(sobel_y, axis=2), axis=3)

        # sobel_x = tf.expand_dims(sobel_x, axis=2)
        # sobel_y = tf.expand_dims(sobel_y, axis=2)
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

            #(
            #     batch_R,
            #     batch_t,
            #     voted_kpts,
            #     (depth, kp, sm, cp, xyz_pred, sampled_inds_in_original_image, mesh_kpts, norm_bbox, cropped_rgbs_l, cropped_rgbs_r, w, attended_right,
            #  intrinsics, crop_factor, w_factor_inv, h_factor_inv, disp, depth_emb )

            (
                b_R,
                b_t,
                b_kpts_pred,
                (b_depth_pred, b_kp_offset_pred, b_seg_pred, b_cp_offset_pred, b_xyz_pred, b_sampled_inds_in_original_image, b_mesh_kpts, b_norm_bbox, b_cropped_rgbs_l, b_cropped_rgbs_r, b_weights, b_attention, 
                 b_intrinsics_matrix, b_crop_factor, b_w_factor_inv, b_h_factor_inv, b_disp_pred, b_depth_emb, b_normal_feats),) = self.model(x, training=False)
            print(f'In log_visual kp = pred[1] {tf.math.reduce_mean(b_kp_offset_pred).numpy()}')
            h, w = tf.shape(b_rgb)[1], tf.shape(b_rgb)[2]

            # get xymz gt
            b_xyz_gt = self.model.pcld_processor_tf_by_index(b_depth_gt+0.000001, b_intrinsics, b_sampled_inds_in_original_image)
            b_mask_gt = tf.expand_dims(b_mask_gt, axis=-1) # better to move it to simpose.py
            b_mask_selected = tf.gather_nd(b_mask_gt, b_sampled_inds_in_original_image)
            # get gt offset
            b_kp_offsets_gt, b_cp_offsets_gt = self.loss_fn.get_offst(b_RT_gt, b_xyz_gt, b_mask_selected, b_mesh_kpts)

            # crop b_depth_gt
            crop_h, crop_w = b_depth_pred.shape[1:3]
        
            # crop_depth_gt = full_depth_gt[y1:y2, x1:x2]
            y = b_baseline * b_intrinsics[:,0,0]
            y = tf.reshape(y, [-1, 1, 1, 1])
            b_disp_gt = tf.math.divide_no_nan(y, b_depth_gt)

            b_depth_gt = tf.image.crop_and_resize(
                tf.cast(b_depth_gt, tf.float32),
                b_norm_bbox,
                tf.range(tf.shape(b_depth_gt)[0]),
                [crop_h, crop_w]
            )
            b_disp_gt = tf.image.crop_and_resize(
                tf.cast(b_disp_gt, tf.float32),
                b_norm_bbox,
                tf.range(tf.shape(b_depth_gt)[0]),
                [crop_h, crop_w]
            )
            b_w_factor_inv = tf.reshape(b_w_factor_inv, [-1, 1, 1, 1])
            b_disp_gt = b_disp_gt * b_w_factor_inv


            # roi
            px_max_disp = 200
            # print('VAL ----- b_roi before px_max_disp ', b_roi)
            b_roi = tf.stack([b_roi[:, 0], tf.math.maximum(tf.zeros_like(b_roi[:, 1]), tf.math.subtract(b_roi[:,1], px_max_disp)), b_roi[:, 2], b_roi[:, 3]], axis = -1)
            # print('VAL ----- b_roi after px_max_disp ', b_roi)

            # attention weights
            [batch_w1, batch_w2, batch_w3, batch_w4, batch_w5] = b_weights
            [batch_a1, batch_a2, batch_a3, batch_a4, batch_a5] = b_attention


            # initial pose model with ground truth pcld and predicted kpts
            b_R_hybrid, b_t_hybrid, b_kpts_pred_hybrid = self.initial_pose_model([b_xyz_gt, b_kp_offset_pred, b_cp_offset_pred, b_seg_pred, b_mesh_kpts])



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
            b_R_hybrid = b_R_hybrid.numpy()
            b_t_hybrid = b_t_hybrid.numpy()
            b_kpts_pred_hybrid = b_kpts_pred_hybrid.numpy()
            b_intrinsics = b_intrinsics.numpy()
            b_depth_pred = b_depth_pred.numpy()
            b_depth_gt = b_depth_gt.numpy()
            b_disp_gt = b_disp_gt.numpy()
            b_disp_pred = b_disp_pred.numpy()
            #print(f'b_disp shape{b_disp_pred.shape} - b disp gt shape{b_disp_gt.shape}')
            # b_xyz_gt = b_xyz_gt.numpy()
            #b_crop_factor = b_crop_factor.numpy()

            # b_roi
            # px_max_disp = 200
            # b_roi[:, 1] = np.max([np.zeros_like(b_roi[:, 1]), np.subtract(b_roi[:,1], px_max_disp)])

            # get gt offsets
            b_offsets_pred = np.concatenate([b_kp_offset, b_cp_offset], axis=2)
            b_offsets_gt = np.concatenate([b_kp_offsets_gt, b_cp_offsets_gt], axis=2)

            # keypoint vectors
            b_kpts_vectors_pred = b_xyz_pred[:, :, None, :].numpy() + b_offsets_pred  # [b, n_pts, 9, 3]
            b_kpts_vectors_gt = b_xyz_gt[:, :, None, :].numpy() + b_offsets_gt
            b_kpts_offset_pred = b_xyz_gt[:, :, None, :].numpy() + b_offsets_pred  # [b, n_pts, 9, 3]

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
            b_mesh_hybrid = (
                np.matmul(b_mesh, b_R_hybrid[:, :3, :3].transpose((0, 2, 1))) + b_t_hybrid[:, tf.newaxis, :]
            )

            b_mesh_pred = self.project_batch_to_image(b_mesh_pred, b_intrinsics)[..., :2].astype(
                np.int32
            )
            b_mesh_gt = self.project_batch_to_image(b_mesh_gt, b_intrinsics)[..., :2].astype(
                np.int32
            )
            b_mesh_hybrid = self.project_batch_to_image(b_mesh_hybrid, b_intrinsics)[..., :2].astype(
                np.int32
            )
            
            b_kpts_gt = self.project_batch_to_image(b_kpts_gt, b_intrinsics)
            #print(f"b_kpts_pred before project_batch_to_image: {b_kpts_pred}")
            b_kpts_pred = self.project_batch_to_image(b_kpts_pred, b_intrinsics)
            b_kpts_pred_hybrid = self.project_batch_to_image(b_kpts_pred_hybrid, b_intrinsics)
            #print(f"b_kpts_pred after project_batch_to_image: {b_kpts_pred}")
            b_kpts_vectors_gt = self.project_batch_to_image(b_kpts_vectors_gt, b_intrinsics)
            b_kpts_vectors_pred = self.project_batch_to_image(b_kpts_vectors_pred, b_intrinsics)
            b_kpts_offset_pred = self.project_batch_to_image(b_kpts_offset_pred, b_intrinsics)
            b_xyz_projected_pred = self.project_batch_to_image(b_xyz_pred, b_intrinsics)  # [b, n_pts, 3]
            b_xyz_projected_gt = self.project_batch_to_image(b_xyz_gt, b_intrinsics)  # [b, n_pts, 3]

            # print(f'Check for Nan xyz {b_xyz_gt} - xyz_projected {b_xyz_projected_gt }')
            # print(f'np.where(np.isnan(b_xyz_projected_gt)) {np.where(np.isnan(b_xyz_projected_gt))}') # \n b_xyz_gt[np.where(np.isnan(b_xyz_projected_gt))] {b_xyz_gt[np.where(np.isnan(b_xyz_projected_gt))]}')
             # Apply Sobel filter in the y-direction using convolution
            b_sobel_x_der = tf.nn.conv2d(b_depth_gt, sobel_x, strides=[1, 1, 1, 1], padding='SAME').numpy()
            b_sobel_y_der = tf.nn.conv2d(b_depth_gt, sobel_y, strides=[1, 1, 1, 1], padding='SAME').numpy()
            # b_sobel_xx_der = tf.nn.conv2d(b_sobel_x_der, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
            # b_sobel_yy_der = tf.nn.conv2d(b_sobel_y_der, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
            # b_sobel_xy_der = tf.nn.conv2d(b_sobel_x_der, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
            # b_sobel_yx_der = tf.nn.conv2d(b_sobel_y_der, sobel_x, strides=[1, 1, 1, 1], padding='SAME')




            for (
                rgb,
                crop_rgb,
                crop_rgb_r,
                crop_depth_pred,
                crop_depth_gt,
                sobel_x_der,
                sobel_y_der,
                norm_bbox,
                roi,
                seg_pred,
                sampled_inds,
                kpts_gt,
                kpts_pred,
                mesh_vertices,
                mesh_vertices_gt,
                mesh_vertices_hybrid,
                offsets_pred,
                offsets_gt,
                crop_factor,
                kpts_vectors_pred,
                kpts_vectors_gt,
                xyz_projected_pred,
                xyz_projected_gt,
                kpts_offset_pred,
                R_hybrid,
                t_hybrid,
                kpts_pred_hybrid,
                disp_pred,
                disp_gt,
                w1,
                w2,
                w3,
                w4,
                w5,
                a1,
                a2,
                a3,
                a4,
                a5,
            ) in zip(
                b_rgb,
                b_cropped_rgbs_l,
                b_cropped_rgbs_r,
                b_depth_pred,
                b_depth_gt,
                b_sobel_x_der,
                b_sobel_y_der,
                b_norm_bbox,
                b_roi,
                b_seg_pred,
                b_sampled_inds,
                b_kpts_gt,
                b_kpts_pred,
                b_mesh_pred,
                b_mesh_gt,
                b_mesh_hybrid,
                b_offsets_pred,
                b_offsets_gt,
                b_crop_factor,
                b_kpts_vectors_pred,
                b_kpts_vectors_gt,
                b_xyz_projected_pred,
                b_xyz_projected_gt,
                b_kpts_offset_pred,
                b_R_hybrid,
                b_t_hybrid,
                b_kpts_pred_hybrid,
                b_disp_pred,
                b_disp_gt,
                batch_w1,
                batch_w2,
                batch_w3,
                batch_w4,
                batch_w5,
                batch_a1,
                batch_a2,
                batch_a3,
                batch_a4,
                batch_a5,
            ):
                

               
                # Compute the gradient magnitude (norm)
                gradient_magnitude = np.sqrt(np.square(sobel_x_der) + np.square(sobel_y_der))
                grad_mask = np.where(gradient_magnitude < 0.1, 1, 0)
                vis_der_grad = self.draw_depth(sobel_x_der, sobel_y_der)
                #self.tracker.log(f"Derivative ({i})", vis_der_grad, index=epoch)
                vis_mask_grad = self.draw_depth(grad_mask, grad_mask)
                #self.tracker.log(f"mask grad ({i})", vis_mask_grad, index=epoch)
                


        
                vis_seg = self.draw_segmentation(rgb.copy(), sampled_inds, seg_pred, roi)
                self.tracker.log(f"RGB ({i})", vis_seg, index=epoch)
                
                vis_depth = self.draw_depth(crop_depth_pred, crop_depth_gt)
                self.tracker.log(f"Depth ({i})", vis_depth, index=epoch)

                vis_disp = self.draw_depth(disp_pred, disp_gt, alpha=1.275)
                self.tracker.log(f"Disp ({i})", vis_disp, index=epoch)

                self.tracker.log(f"RGB R (crop) ({i})", (crop_rgb_r*255.).astype(np.uint8), index=epoch)
                self.tracker.log(f"RGB L (crop) ({i})", (crop_rgb*255.).astype(np.uint8), index=epoch)

                vis_mesh = self.draw_object_mesh(rgb.copy(), roi, mesh_vertices, mesh_vertices_hybrid, mesh_vertices_gt)
                self.tracker.log(f"RGB (mesh) ({i})", vis_mesh, index=epoch)

                # print(f'kpts_gt {kpts_gt} - kpts_gt {kpts_gt.shape}')
                # print(f'kpts_pred {kpts_pred} - kpts_pred {kpts_pred.shape}')

                vis_kpts = self.draw_keypoint_correspondences(rgb.copy(), roi, kpts_gt, kpts_pred, kpts_pred_hybrid)
                self.tracker.log(f"RGB (kpts) ({i})", vis_kpts, index=epoch)

                #self.tracker.log(f"RGBR ({i})", crop_rgb_r, index=epoch)

                vis_offsets = self.draw_keypoint_offsets(
                    rgb.copy(),
                    roi,
                    offsets_pred,
                    sampled_inds,
                    kpts_gt,
                    seg=seg_pred,
                    radius=int(crop_factor),
                )

                vis_offsets_hybrid = self.draw_keypoint_offsets(
                    rgb.copy(),
                    roi,
                    offsets_pred,
                    sampled_inds,
                    kpts_gt,
                    seg=seg_pred,
                    radius=int(crop_factor),
                )

                vis_offsets_gt = self.draw_keypoint_offsets(
                    rgb.copy(),
                    roi,
                    offsets_gt,
                    sampled_inds,
                    kpts_gt,
                    radius=1 #int(crop_factor),
                )

                
                # assemble images side-by-side
                vis_offsets = np.concatenate([vis_offsets_gt, vis_offsets, vis_offsets_hybrid], axis=1)
                self.tracker.log(f"RGB (offsets) ({i})", vis_offsets, index=epoch)

                vis_vectors = self.draw_keypoint_vectors(
                    rgb.copy(),
                    roi,
                    offsets_pred,
                    kpts_gt, #kpts_pred
                    xyz_projected_pred,
                    kpts_vectors_pred,
                    seg=seg_pred,
                )

                vis_vectors_gt = self.draw_keypoint_vectors(
                    rgb.copy(),
                    roi,
                    offsets_gt,
                    kpts_gt,
                    xyz_projected_gt,
                    kpts_vectors_gt
                )

                vis_vectors_hybrid = self.draw_keypoint_vectors(
                    rgb.copy(),
                    roi,
                    offsets_gt,
                    kpts_gt,
                    xyz_projected_gt, 
                    kpts_offset_pred
                )




                # assemble images side-by-side
                vis_vectors = np.concatenate([vis_vectors_gt, vis_vectors, vis_vectors_hybrid], axis=1)
                self.tracker.log(f"RGB (vectors) ({i})", vis_vectors, index=epoch)


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

                # draw also attended_right images (x output of the model)
                vis_a1 = self.draw_weights(a1)
                self.tracker.log(f"Attentded Right 1 ({i})", vis_a1, index=epoch)

                vis_a2 = self.draw_weights(a2)
                self.tracker.log(f"Attended Right 2 ({i})", vis_a2, index=epoch)

                vis_a3 = self.draw_weights(a3)
                self.tracker.log(f"Attended Right 3 ({i})", vis_a3, index=epoch)

                vis_a4 = self.draw_weights(a4)
                self.tracker.log(f"Attended Right 4 ({i})", vis_a4, index=epoch)

                vis_a5 = self.draw_weights(a5)
                self.tracker.log(f"Attended Right 5 ({i})", vis_a5, index=epoch)



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
    
    def draw_depth(self, crop_depth_pred, crop_depth_gt, alpha=170.):
        h, w = crop_depth_pred.shape[:2]
        assembled = np.zeros((h, 3 * w, 3), dtype=np.uint8)
        scaled_image = (crop_depth_pred - np.mean(crop_depth_pred)) / (np.max(crop_depth_pred) - np.min(crop_depth_pred)) #(img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data)) and (img_data_normalized * 255).astype(np.uint8)
        # print(f'scaled_image.shape {scaled_image.shape}')
        a = np.concatenate([scaled_image, scaled_image, scaled_image], axis=-1)
        # print('scaled image after concat', a.shape)
        im = (a*255).astype(np.uint8)
        im = np.clip(im, 0, 255)
        # print(f"statistic pred_depth plot: min:{np.min(im)} - max: {np.max(im)} - mean: {np.mean(im)} - std: {np.std(im)}")
        color_depth = lambda d: cv2.applyColorMap(
            cv2.convertScaleAbs(d, alpha=alpha), cv2.COLORMAP_JET
        )

        colored_gt = color_depth(crop_depth_gt)
        # colored_pred = color_depth(crop_depth_pred)
        # crop_depth_pred = np.clip(crop_depth_pred, 0 ,255).astype(np.uint8)# 
        # crop_depth_pred = cv2.normalize(crop_depth_pred, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        colored_pred = color_depth(crop_depth_pred)

        assembled[:, :w, :] = colored_gt
        assembled[:, w:w*2, :] = colored_pred
        assembled[:, w*2:, :] = im

        # print(f"crop_depth_pred.shape: {crop_depth_pred.shape}")
        # print(f"crop_depth_gt.shape: {crop_depth_gt.shape}")
        # print(f"assembled.shape: {assembled.shape}")

        return assembled
    

    def draw_object_mesh(self, rgb, roi, mesh_vertices_pred, mesh_vertices_hyb, mesh_vertices_gt):
        h, w = rgb.shape[:2]
        clipped_mesh_vertices_pred = np.clip(mesh_vertices_pred, 0, [w - 1, h - 1])
        clipped_mesh_vertices_hyb = np.clip(mesh_vertices_hyb, 0, [w - 1, h - 1])
        clipped_mesh_vertices_gt = np.clip(mesh_vertices_gt, 0, [w - 1, h - 1])
        rgb_pred = rgb.copy()
        rgb_hyb = rgb.copy()
        rgb_gt = rgb.copy()

        for x, y in clipped_mesh_vertices_pred:
            cv2.circle(rgb_pred, (x, y), 1, (255, 0, 0), -1)
        for x, y in clipped_mesh_vertices_hyb:
            cv2.circle(rgb_hyb, (x, y), 1, (0, 0, 255), -1)
        for x, y in clipped_mesh_vertices_gt:
            cv2.circle(rgb_gt, (x, y), 1, (0, 255, 0), -1)

        rgb_pred = self.crop_to_roi(rgb_pred, roi)
        rgb_hyb = self.crop_to_roi(rgb_hyb, roi)
        rgb_gt = self.crop_to_roi(rgb_gt, roi)
        # assemble images side-by-side
        vis_mesh = np.concatenate([rgb_gt, rgb_pred, rgb_hyb], axis=1)
        return vis_mesh

    def draw_keypoint_correspondences(self, rgb, roi, kpts_gt, kpts_pred, kpts_hybrid):
        # normalize z_coords of keypoints to [0, 1]
        rgb_pred = rgb.copy()
        rgb_hyb = rgb.copy()


        kpts_gt[..., 2] = (kpts_gt[..., 2] - np.min(kpts_gt[..., 2])) / (
            np.max(kpts_gt[..., 2]) - np.min(kpts_gt[..., 2])
        )
        kpts_pred[..., 2] = (kpts_pred[..., 2] - np.min(kpts_pred[..., 2])) / (
            np.max(kpts_pred[..., 2]) - np.min(kpts_pred[..., 2])
        )
        kpts_hybrid[..., 2] = (kpts_hybrid[..., 2] - np.min(kpts_hybrid[..., 2])) / (
            np.max(kpts_hybrid[..., 2]) - np.min(kpts_hybrid[..., 2])
        )
        # print(f'In draw_keypoint_correspondences kpts_pred[..., 2]: {kpts_pred[..., 2]}')

        for (x_gt, y_gt, z_gt), (x_pred, y_pred, z_pred) in zip(kpts_gt, kpts_pred):
            gt_color = np.array((0, 255, 0), dtype=np.uint8)
            pred_color = np.array((255, 0, 0), dtype=np.uint8)

            scale_marker = lambda z: 10 + int(z * 20)

            cv2.drawMarker(
                rgb_pred,
                (int(x_gt), int(y_gt)),
                gt_color.tolist(),
                cv2.MARKER_CROSS,
                scale_marker(z_gt),
                1,
            )
            cv2.drawMarker(
                rgb_pred,
                (int(x_pred), int(y_pred)),
                pred_color.tolist(),
                cv2.MARKER_TILTED_CROSS,
                scale_marker(z_pred),
                1,
            )
            cv2.line(
                rgb_pred,
                (int(x_gt), int(y_gt)),
                (int(x_pred), int(y_pred)),
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        for (x_gt, y_gt, z_gt), (x_pred, y_pred, z_pred) in zip(kpts_gt, kpts_hybrid):
            gt_color = np.array((0, 255, 0), dtype=np.uint8)
            hyb_color = np.array((0, 0, 255), dtype=np.uint8)

            scale_marker = lambda z: 10 + int(z * 20)

            cv2.drawMarker(
                rgb_hyb,
                (int(x_gt), int(y_gt)),
                gt_color.tolist(),
                cv2.MARKER_CROSS,
                scale_marker(z_gt),
                1,
            )
            cv2.drawMarker(
                rgb_hyb,
                (int(x_pred), int(y_pred)),
                hyb_color.tolist(),
                cv2.MARKER_TILTED_CROSS,
                scale_marker(z_pred),
                1,
            )
            cv2.line(
                rgb_hyb,
                (int(x_gt), int(y_gt)),
                (int(x_pred), int(y_pred)),
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        rgb_pred = self.crop_to_roi(rgb_pred, roi)
        rgb_hyb = self.crop_to_roi(rgb_hyb, roi)
        
        return np.concatenate([rgb_pred, rgb_hyb], axis =1)

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
            value = np.clip((np.linalg.norm(offsets[:, i, :], axis=-1) / 0.1 * 255), 0, 255).astype(np.uint8)
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
                    cv2.convertScaleAbs(d, alpha=255), cv2.COLORMAP_JET
                )
        h_att, w_att, channels = attention[:,:,:].shape[:3]

        assembled_att1 = np.zeros((h_att, np.min([channels, 7]) * w_att, 3), dtype=np.uint8)
        for i in range(channels):
            colored_attention = color_attention(attention[:,:,-i].numpy())
            # print(f'colored_attention {colored_attention.shape}')
            # print(f'starting index {w_att*i} - send index {w_att*(i+1)}')
            assembled_att1[:, w_att*i:w_att*(i+1), :] = colored_attention
            #assembled_att1[:, w_att:2*w_att, :] = colored_attention2
            #assembled_att1[:, w_att*2:3*w_att, :] = colored_attention3
            if i == 6:
                break
        return assembled_att1
    
    