from cvde.tf import Dataset as _Dataset

import os
import tensorflow as tf

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
from PIL import Image
import cv2
import json
import yaml
from scipy.spatial.transform import Rotation as R
import pathlib
import streamlit as st
import itertools as it
import open3d as o3d
import itertools as it
from typing import Dict


class LineMOD(_Dataset):
    def __init__(
        self,
        *,
        cls_type,
        batch_size,
        use_cache,
        root,
    ):
        super().__init__()
        self.cls_dict = {
            "ape": 1,
            "benchvise": 2,
            "bowl": 3,
            "cam": 4,
            "can": 5,
            "cat": 6,
            "cup": 7,
            "driller": 8,
            "duck": 9,
            "eggbox": 10,
            "glue": 11,
            "holepuncher": 12,
            "iron": 13,
            "lamp": 14,
            "phone": 15,
        }
        if cls_type.startswith("lm_"):
            cls_type = cls_type[3:]
        self.cls_type = cls_type
        self.cls_id = self.cls_dict[cls_type]
        self.batch_size = batch_size
        self.use_cache = use_cache

        self.root = pathlib.Path(root)
        self.data_root = data_root = self.root.joinpath(f"linemod/data/{self.cls_id:02}")

        self.total_n_imgs = len(list(self.data_root.joinpath("rgb").glob("*.png")))

        mesh_kpts_path = self.root / "lm_obj_kpts" / self.cls_type
        kpts = np.loadtxt(mesh_kpts_path / "farthest.txt")
        corners = np.loadtxt(mesh_kpts_path / "corners.txt")
        center = [corners.mean(0)]
        self.mesh_kpts = np.concatenate([kpts, center], axis=0)

        mesh_path = self.root / "lm_obj_mesh" / f"obj_{self.cls_id:02}.ply"
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        self.mesh_vertices = np.asarray(mesh.sample_points_poisson_disk(1000).points) / 1000.

        self.intrinsic_matrix = np.array(
            [[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]]
        ).astype(np.float32)

        with open(os.path.join(self.data_root, "gt.yml"), "r") as meta_file:
            self.gt_list = yaml.load(meta_file, Loader=yaml.FullLoader)

        print("Initialized LineMOD Dataset.")
        print(f"\t# of all images: {self.total_n_imgs}")
        print(f"\tCls root: {data_root}")
        print(f"\t# of augmented datapoints: {len(self)}")
        # print(f"\nIntrinsic matrix: {self.intrinsic_matrix}")
        print()

    def to_tf_dataset(self):
        if self.use_cache:
            cache_path = self.data_root / f"cache"
            try:
                tfds = self.from_cache(cache_path)
            except FileNotFoundError:
                self.cache(cache_path)
                tfds = self.from_cache(cache_path)

            print("USING CACHE FROM ", cache_path)

            def arrange_as_xy_tuple(d):
                return (d["rgb"], d["depth"], d["intrinsics"], d["roi"], d["mesh_kpts"]), (
                    d["RT"],
                    d["mask"],
                )

            print("USING CACHE FROM ", cache_path)
            return (
                tfds.map(arrange_as_xy_tuple)
                .batch(self.batch_size, drop_remainder=True)
                .prefetch(tf.data.AUTOTUNE)
            )

        raise NotImplementedError

    def visualize_example(self, example):
        color_depth = lambda x: cv2.applyColorMap(
            cv2.convertScaleAbs(x, alpha=255 / 2), cv2.COLORMAP_JET
        )

        rgb = example["rgb"]
        depth = example["depth"]
        intrinsics = example["intrinsics"].astype(np.float32)
        bboxes = example["roi"]
        kpts = example["mesh_kpts"]

        RT = example["RT"]
        mask = example["mask"]

        (
            y1,
            x1,
            y2,
            x2,
        ) = bboxes[:4]
        out_rgb = cv2.rectangle(rgb.copy(), (x1, y1), (x2, y2), (0, 255, 0), 2)
        rvec = cv2.Rodrigues(RT[:3, :3])[0]
        tvec = RT[:3, 3]
        cv2.drawFrameAxes(out_rgb, intrinsics, np.zeros((4,)), rvec=rvec, tvec=tvec, length=0.1)

        c1, c2 = st.columns(2)
        c1.image(out_rgb, caption=f"RGB_L {rgb.shape} ({rgb.dtype})")
        c1.image(
            color_depth(depth),
            caption=f"Depth {depth.shape} ({depth.dtype})",
        )
        c1.image(mask * 255, caption=f"Mask {mask.shape} ({mask.dtype})")

        c2.write(intrinsics)
        c2.write(kpts)
        c2.write(RT)

        from losses.pvn_loss import PvnLoss
        from models.pvn3d_e2e import PVN3D_E2E

        num_samples = st.select_slider("num_samples", [2**i for i in range(5, 13)])
        margin = st.slider("margin", 0, 200, 0, step=50)

        h, w = rgb.shape[:2]
        _bbox, _crop_factor = PVN3D_E2E.get_crop_index(
            bboxes[None], h, w, 160, 160
        )  # bbox: [b, 4], crop_factor: [b]

        # xyz: [b, num_points, 3]
        # inds:  # [b, num_points, 3] with last 3 is index into original image
        xyz, feats, inds = PVN3D_E2E.pcld_processor_tf(
            (rgb[None] / 255.0).astype(np.float32),
            depth[None].astype(np.float32),
            intrinsics[None].astype(np.float32),
            _bbox,
            num_samples,
        )
        mask_selected = tf.gather_nd(mask[None], inds)
        kp_offsets, cp_offsets = PvnLoss.get_offst(
            RT[None].astype(np.float32),
            xyz,
            mask_selected,
            self.mesh_kpts[None].astype(np.float32),
        )
        # [1, n_pts, 8, 3] | [1, n_pts, 1, 3]

        all_offsets = np.concatenate([kp_offsets, cp_offsets], axis=-2)  # [b, n_pts, 9, 3]

        offset_views = {}
        cam_cx, cam_cy = intrinsics[0, 2], intrinsics[1, 2]  # [b]
        cam_fx, cam_fy = intrinsics[0, 0], intrinsics[1, 1]  # [b]

        def to_image(pts):
            coors = (
                pts[..., :2] / pts[..., 2:] * tf.stack([cam_fx, cam_fy], axis=0)[tf.newaxis, :]
                + tf.stack([cam_cx, cam_cy], axis=0)[tf.newaxis, :]
            )
            coors = tf.floor(coors)
            return tf.concat([coors, pts[..., 2:]], axis=-1).numpy()

        projected_keypoints = self.mesh_kpts @ RT[:3, :3].T + RT[:3, 3]
        projected_keypoints = to_image(projected_keypoints)

        # for each pcd point add the offset
        keypoints_from_pcd = xyz[:, :, None, :].numpy() + all_offsets  # [b, n_pts, 9, 3]
        keypoints_from_pcd = to_image(keypoints_from_pcd.astype(np.float32))
        projected_pcd = to_image(xyz)  # [b, n_pts, 3]

        for i in range(9):
            # offset_view = np.zeros_like(rgb, dtype=np.uint8)
            offset_view = rgb.copy() // 3

            # get color hue from offset
            hue = np.arctan2(all_offsets[0, :, i, 1], all_offsets[0, :, i, 0]) / np.pi
            hue = (hue + 1) / 2
            hue = (hue * 180).astype(np.uint8)
            # value = np.ones_like(hue) * 255
            value = (np.linalg.norm(all_offsets[0, :, i, :], axis=-1) / 0.1 * 255).astype(np.uint8)
            hsv = np.stack([hue, np.ones_like(hue) * 255, value], axis=-1)
            colored_offset = cv2.cvtColor(hsv[None], cv2.COLOR_HSV2RGB).astype(np.uint8)

            sorted_inds = np.argsort(np.linalg.norm(all_offsets[0, :, i, :], axis=-1), axis=-1)[
                ::-1
            ]
            keypoints_from_pcd[0, :, i, :] = keypoints_from_pcd[0, sorted_inds, i, :]
            colored_offset[0] = colored_offset[0, sorted_inds, :]
            sorted_xyz = projected_pcd[0, sorted_inds, :]
            for start, target, color in zip(
                sorted_xyz, keypoints_from_pcd[0, :, i, :], colored_offset[0]
            ):
                # over all pcd points
                cv2.line(
                    offset_view,
                    tuple(map(int, start[:2])),
                    tuple(map(int, target[:2])),
                    tuple(map(int, color)),
                    1,
                )

            # # mark correct keypoint
            cv2.drawMarker(
                offset_view,
                (int(projected_keypoints[i, 0]), int(projected_keypoints[i, 1])),
                (0, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=1,
            )

            h, w = offset_view.shape[:2]
            y1, x1, y2, x2 = _bbox[0]
            x1 = np.clip(x1 - margin, 0, w)
            x2 = np.clip(x2 + margin, 0, w)
            y1 = np.clip(y1 - margin, 0, h)
            y2 = np.clip(y2 + margin, 0, h)
            offset_view = offset_view[y1:y2, x1:x2]
            name = f"Keypoint {i}" if i < 8 else "Center"
            offset_views.update({name: offset_view})

        cols = it.cycle(st.columns(3))

        for col, (name, offset_view) in zip(cols, offset_views.items()):
            col.image(offset_view, caption=name)

    def __len__(self):
        return self.total_n_imgs

    def __getitem__(self, idx):
        # this cycles n_aug_per_image times through the dataset
        rgb = self.get_rgb(idx)
        depth = self.get_depth(idx)
        mask = np.where(self.get_mask(idx) == 255, 1, 0).astype(np.uint8)
        bbox = self.get_bbox(idx)
        RT = self.get_RT(idx)

        return {
            "rgb": rgb.astype(np.uint8),
            "depth": depth.astype(np.float32),
            "intrinsics": self.intrinsic_matrix.astype(np.float32),
            "roi": bbox.astype(np.int32),
            "RT": RT.astype(np.float32),
            "mask": mask.astype(np.uint8),
            "mesh_kpts": self.mesh_kpts.astype(np.float32),
        }

    def get_rgb(self, index) -> np.ndarray:
        rgb_path = os.path.join(self.data_root, "rgb", f"{index:04}.png")
        with Image.open(rgb_path) as rgb:
            rgb = np.array(rgb).astype(np.uint8)
        return rgb

    def get_mask(self, index):
        with Image.open(os.path.join(self.data_root, "mask", f"{index:04}.png")) as mask:
            mask = np.array(mask)
        return mask

    def get_depth(self, index):
        with Image.open(os.path.join(self.data_root, "depth", f"{index:04}.png")) as depth:
            dpt = np.array(depth)[..., np.newaxis]
            return dpt / 1000.0  # to m

    def get_bbox(self, index):
        gt = self.gt_list[index]

        if self.cls_id == 2:
            gt = [x for x in gt if x["obj_id"] == self.cls_id][0]
        else:
            gt = gt[0]

        # elif self.cls_id == 16:
        #     for i in range(0, len(meta)):
        #         meta_list.append(meta[i])

        bbox = np.array(gt["obj_bb"])

        bbox = self.convert2pascal(bbox)
        bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]  # [ymin, xmin, ymax, xmax]
        return np.array(bbox)

    @staticmethod
    def convert2pascal(box_coor):
        """
        params: box_coor [xmin, ymin, w, h]
        return: box_coor in pascal_voc format [xmin, ymin, xmax, ymax] (left upper corner and right bottom corner)
        """
        x_min, y_min, w, h = box_coor
        x_max = x_min + w
        y_max = y_min + h
        return [x_min, y_min, x_max, y_max]

    def get_RT(self, index):
        gt = self.gt_list[index]
        if self.cls_id == 2:
            gt = [x for x in gt if x["obj_id"] == self.cls_id][0]
        else:
            gt = gt[0]

        R = np.resize(np.array(gt["cam_R_m2c"]), (3, 3))
        T = np.array(gt["cam_t_m2c"]) / 1000.0
        RT = np.zeros((4, 4))
        RT[:3, :3] = R
        RT[:3, 3] = T
        return RT
