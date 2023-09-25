from cvde.tf import Dataset as _Dataset

import os
import tensorflow as tf

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
from PIL import Image
import cv2
import json
from scipy.spatial.transform import Rotation as R
import pathlib
import albumentations as A
import streamlit as st
import itertools as it
import open3d as o3d
import typing

from .augment import add_background_depth, augment_depth, augment_rgb, rotate_datapoint


class _Blender(_Dataset):
    def __init__(
        self,
        *,
        data_name,
        if_augment,
        is_train,
        cls_type,
        im_size,
        batch_size,
        use_cache,
        root,
        train_split,
        n_aug_per_image: int,
        cutoff=None,
    ):
        super().__init__()
        self.cls_dict = {
            "cpsduck": 1,
            "stapler": 2,
            "cpsglue": 3,
            "wrench_13": 4,
            "chew_toy": 5,
            "pliers": 6,
            "all": 100,  # hacked to cpsduck for now
        }
        self.colormap = [
            [0, 0, 0],
            [255, 255, 0],
            [0, 0, 255],
            [240, 240, 240],
            [0, 255, 0],
            [255, 0, 50],
            [0, 255, 255],
        ]
        self.labelmap = {v: k for k, v in self.cls_dict.items()}
        self.cls_type = cls_type
        self.cls_id = self.cls_dict[self.cls_type]
        self.current_cls_root = 0
        self.if_augment = if_augment
        self.im_size = im_size
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.is_train = is_train
        self.n_aug_per_image = n_aug_per_image

        self.data_root = data_root = pathlib.Path(root) / data_name

        self.kpts_root = data_root / "kpts"
        if self.cls_type != "all":
            self.cls_root = data_root / f"{self.cls_id:02}"
            self.roots_and_ids = self.get_roots_and_ids_for_cls_root(self.cls_root)

        else:
            raise AssertionError("Not implemented yet.")
            self.all_cls_roots = [data_root / x for x in data_root.iterdir()]
            self.cls_root = self.all_cls_roots[0]
            all_roots_and_ids = [
                self.get_roots_and_ids_for_cls_root(x) for x in self.all_cls_roots
            ]
            self.roots_and_ids = []
            [self.roots_and_ids.extend(x) for x in zip(*all_roots_and_ids)]

        if cutoff is not None:
            self.roots_and_ids = self.roots_and_ids[:cutoff]

        total_n_imgs = len(self.roots_and_ids)

        split_ind = np.floor(len(self.roots_and_ids) * train_split).astype(int)
        if is_train:
            self.roots_and_ids = self.roots_and_ids[:split_ind]
        else:
            self.roots_and_ids = self.roots_and_ids[split_ind:]

        with open(os.path.join(self.cls_root, "gt.json")) as f:
            json_dict = json.load(f)
        self.intrinsic_matrix = np.array(json_dict["camera_matrix"])
        self.baseline = np.array(json_dict["stereo_baseline"])

        self.rgbmask_augment = A.Compose(
            [
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5),
                A.RandomGamma(p=0.2),
                A.AdvancedBlur(p=0.2),
                A.GaussNoise(p=0.2),
                A.FancyPCA(p=0.2),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
            ],
        )

        mesh_kpts_path = self.kpts_root / self.cls_type
        kpts = np.loadtxt(mesh_kpts_path / "farthest.txt")
        center = [np.loadtxt(mesh_kpts_path / "center.txt")]
        self.mesh_kpts = np.concatenate([kpts, center], axis=0)

        mesh_path = self.data_root / "meshes" / (self.cls_type + ".ply")
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        self.mesh_vertices = np.asarray(mesh.sample_points_poisson_disk(1000).points)

        print("Initialized Blender Dataset.")
        print(f"\tData name: {data_name}")
        print(f"\tTotal split # of images: {len(self.roots_and_ids)}")
        print(f"\tCls root: {self.cls_root}")
        print(f"\t# of all images: {total_n_imgs}")
        # print(f"\nIntrinsic matrix: {self.intrinsic_matrix}")
        print()

    def to_tf_dataset(self):
        if self.use_cache:
            cache_name = "train" if self.is_train else "val"
            cache_path = self.cls_root / f"cache_{cache_name}"
            try:
                tfds = self.from_cache(cache_path)
            except FileNotFoundError:
                self.cache(cache_path)
                tfds = self.from_cache(cache_path)

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

    def get_roots_and_ids_for_cls_root(self, cls_root: pathlib.Path):
        all_files = (cls_root / "rgb").glob("*")
        files = [x for x in all_files if "_R" not in str(x)]
        numeric_file_ids = list([int(x.stem.split("_")[1]) for x in files])
        numeric_file_ids.sort()
        return [(cls_root, id) for id in numeric_file_ids]

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
            x1 = np.clip(x1 - 100, 0, w)
            x2 = np.clip(x2 + 100, 0, w)
            y1 = np.clip(y1 - 100, 0, h)
            y2 = np.clip(y2 + 100, 0, h)
            offset_view = offset_view[y1:y2, x1:x2]
            name = f"Keypoint {i}" if i < 8 else "Center"
            offset_views.update({name: offset_view})

        cols = it.cycle(st.columns(3))

        for col, (name, offset_view) in zip(cols, offset_views.items()):
            col.image(offset_view, caption=name)

    def __len__(self):
        return len(self.roots_and_ids) * self.n_aug_per_image

    def __getitem__(self, idx):
        # this cycles n_aug_per_image times through the dataset
        self.cls_root, i = self.roots_and_ids[idx % len(self.roots_and_ids)]

        rgb = self.get_rgb(i)
        mask = self.get_mask(i)
        depth = self.get_depth(i)

        rt_list = self.get_RT_list(i)  # FOR SINGLE OBJECT

        # always add depth background
        obj_pos = rt_list[0][:3, 3]  # FOR SINGLE OBJECT
        depth = add_background_depth(
            depth[tf.newaxis, :, :],
            obj_pos=obj_pos,
            camera_matrix=self.intrinsic_matrix.astype(np.float32),
            rgb2noise=None,
        )[
            0
        ].numpy()  # add and remove batch

        # 6IMPOSE data augmentation
        if self.if_augment:
            depth = augment_depth(depth[tf.newaxis])[0].numpy()
            rgb = augment_rgb(rgb.astype(np.float32) / 255.0) * 255
            rgb, mask, depth, rt_list = rotate_datapoint(img_likes=[rgb, mask, depth], Rt=rt_list)

        rt = rt_list[0]
        bboxes = self.get_gt_bbox(i, mask=mask)  # FOR SINGLE OBJECT
        if bboxes is None:
            # create a bbox in the top left corner as negative example
            bboxes = np.array([[0, 0, 200, 200]])
        bboxes = bboxes[0]
        mask = np.where(mask == self.cls_id, 1, 0).astype(np.uint8)  # FOR SINGLE OBJECT

        return {
            "rgb": rgb.astype(np.uint8),
            "depth": depth.astype(np.float32),
            "intrinsics": self.intrinsic_matrix.astype(np.float32),
            "roi": bboxes[:4].astype(np.int32),
            "mesh_kpts": self.mesh_kpts.astype(np.float32),
            "RT": rt.astype(np.float32),
            "mask": mask.astype(np.uint8),
        }

    def get_rgb(self, index):
        rgb_path = os.path.join(self.cls_root, "rgb", f"rgb_{index:04}.png")
        with Image.open(rgb_path) as rgb:
            rgb = np.array(rgb).astype(np.uint8)
        return rgb

    def get_mask(self, index):
        mask_path = os.path.join(self.cls_root, "mask", f"segmentation_{index:04}.exr")
        mask = cv2.imread(mask_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        mask = mask[:, :, :1]
        return mask  # .astype(np.uint8)

    def get_depth(self, index):
        depth_path = os.path.join(self.cls_root, "depth", f"depth_{index:04}.exr")
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth = depth[:, :, :1]
        depth_mask = depth < 5  # in meters, we filter out the background( > 5m)
        depth = depth * depth_mask
        return depth

    def get_RT_list(self, index):
        """return a list of tuples of RT matrix and cls_id [(RT_0, cls_id_0), (RT_1,, cls_id_1) ..., (RT_N, cls_id_N)]"""
        with open(os.path.join(self.cls_root, "gt", f"gt_{index:05}.json")) as f:
            shot = json.load(f)

        cam_quat = shot["cam_rotation"]
        cam_rot = R.from_quat([*cam_quat[1:], cam_quat[0]])
        cam_pos = np.array(shot["cam_location"])
        cam_Rt = np.eye(4)
        cam_Rt[:3, :3] = cam_rot.as_matrix().T
        cam_Rt[:3, 3] = -cam_rot.as_matrix() @ cam_pos

        objs = shot["objs"]

        RT_list = []

        if self.cls_type == "all":
            for obj in objs:
                cls_type = obj["name"]
                cls_id = self.cls_dict[cls_type]
                pos = np.array(obj["pos"])
                quat = obj["rotation"]
                rot = R.from_quat([*quat[1:], quat[0]])
                Rt = np.eye(4)
                Rt[:3, :3] = cam_rot.as_matrix().T @ rot.as_matrix()
                Rt[:3, 3] = cam_rot.as_matrix().T @ (pos - cam_pos)
                # RT_list.append((Rt, cls_id))

        else:
            for obj in objs:  # here we only consider the single obj
                if obj["name"] == self.cls_type:
                    cls_type = obj["name"]
                    pos = np.array(obj["pos"])
                    quat = obj["rotation"]
                    rot = R.from_quat([*quat[1:], quat[0]])
                    Rt = np.eye(4)
                    Rt[:3, :3] = cam_rot.as_matrix().T @ rot.as_matrix()
                    Rt[:3, 3] = cam_rot.as_matrix().T @ (pos - cam_pos)
                    # RT_list.append((Rt, self.cls_id))
                    RT_list.append(Rt)
        return np.array(RT_list)

    def get_gt_bbox(self, index, mask=None) -> typing.Union[np.ndarray, None]:
        bboxes = []
        if mask is None:
            mask = self.get_mask(index)
        if self.cls_type == "all":
            for cls, gt_mask_value in self.cls_dict.items():
                bbox = self.get_bbox_from_mask(mask, gt_mask_value)
                if bbox is None:
                    continue
                bbox = list(bbox)
                bbox.append(self.cls_dict[cls])
                bboxes.append(bbox)
        else:
            bbox = self.get_bbox_from_mask(mask, gt_mask_value=self.cls_id)
            if bbox is None:
                return None
            bbox = list(bbox)
            bbox.append(self.cls_id)
            bboxes.append(bbox)

        return np.array(bboxes).astype(np.int32)

    def get_bbox_from_mask(self, mask, gt_mask_value):
        """mask with object as 255 -> bbox [x1,y1, x2, y2]"""

        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        y, x = np.where(mask == gt_mask_value)
        inds = np.stack([x, y])
        if 0 in inds.shape:
            return None
        x1, y1 = np.min(inds, 1)
        x2, y2 = np.max(inds, 1)

        if (x2 - x1) * (y2 - y1) < 1600:
            return None

        return (y1, x1, y2, x2)


class TrainBlender(_Blender):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, if_augment=True, is_train=True, **kwargs)


class ValBlender(_Blender):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, if_augment=False, is_train=False, **kwargs)
