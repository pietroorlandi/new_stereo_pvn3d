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
import streamlit as st
import itertools as it
import open3d as o3d
import itertools as it
from typing import Dict
#import minexr
from .utils import EXR
from losses.pvn_loss import PvnLoss
from models.stereopvn3d import StereoPvn3d
from models.utils import get_pcld_rgb
from .augment import add_background_depth, augment_depth, augment_rgb

"""
data in the 6IMPOSE datasets:
for each datapoint:
    - [File] rgb
    - [File] depth
    - [File] mask_visib (using instance id and visible pixels)
    - for each object:
        - [FILE] masks without occlusion mask/mask_{obj.object_id:04}_{dataset_index:04}.exr (only for labelled objects)
    - [in gt.json]:
        - "cam_rotation": [x,y,z,w] (like scipy)
        - "cam_location": [x,y,z]
        - "cam_matrix": [3,3] intrinsic matrix (openCV)
        for each object:
            - "class": str
            - "object id": int (instance id)
            - "pos": [x,y,z]
            - "rotation": [x,y,z,w] (like scipy)
            - "bbox_visib": [x1,y1,x2,y2] in pixels, bbox of the object in the image
            - "bbox_obj": [x1,y1,x2,y2] in pixels, bbox of the object in the image without occlusion
            - "px_count_visib": int, number of visible pixels for this object,
            - "px_count_valid": int, number of visible pixels with valid depth for this object,
            - "px_count_all": int, number of visible pixels without occlusions for this object
            - "visib_fract": px_count_visib / px_count_all (or 0.)
        
"""


class _6IMPOSE(_Dataset):
    def __init__(
        self,
        *,
        if_augment,
        is_train,
        cls_type,
        batch_size,
        use_cache,
        root,
        train_split,
        if_pose,
        n_aug_per_image: int,
        n_objects_per_image: int,
        add_bbox_noise: bool,
        bbox_noise: int,
        cutoff=None,
    ):
        super().__init__()
        self.colormap = [
            [0, 0, 0],
            [255, 255, 0],
            [0, 0, 255],
            [240, 240, 240],
            [0, 255, 0],
            [255, 0, 50],
            [0, 255, 255],
        ]
        self.n_sample_points = 1500
        self.cls_type = cls_type
        self.if_augment = if_augment
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.if_pose = if_pose
        self.is_train = is_train
        self.n_aug_per_image = n_aug_per_image
        self.n_objects_per_image = n_objects_per_image
        self.add_bbox_noise = add_bbox_noise
        self.bbox_noise = bbox_noise

        self.data_root = pathlib.Path(root).joinpath(cls_type)

        all_files = self.data_root.joinpath("rgb").glob("*")
        files = [x for x in all_files if "_R" not in str(x)]
        numeric_file_ids = list([int(x.stem.split("_")[1]) for x in files])
        numeric_file_ids.sort()
        self.file_ids = [id for id in numeric_file_ids]

        if cutoff is not None:
            self.file_ids = self.file_ids[:cutoff]

        total_n_imgs = len(self.file_ids)

        split_ind = np.floor(len(self.file_ids) * train_split).astype(int)
        if is_train:
            self.file_ids = self.file_ids[:split_ind]
        else:
            self.file_ids = self.file_ids[split_ind:]

        mesh_path = self.data_root.joinpath(f"meshes/{self.cls_type}.obj")
        if not mesh_path.exists():
            mesh_path = mesh_path.with_suffix(".ply")
        if not mesh_path.exists():
            raise ValueError(f"Mesh file {mesh_path} does not exist.")
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        self.mesh_vertices = np.asarray(mesh.sample_points_poisson_disk(1000).points)

        mesh_kpts_path = self.data_root.parent / "0_kpts" / self.cls_type
        if not mesh_kpts_path.exists():
            mesh_kpts_path.mkdir(parents=True)
            print("Generating mesh keypoints...")
            print("Make sure to use the correct keypoints!")
            center_point = mesh.get_center()
            np.savetxt(mesh_kpts_path / "center.txt", center_point)
            mesh_kpts = np.asarray(mesh.sample_points_poisson_disk(8).points)
            np.savetxt(mesh_kpts_path / "farthest.txt", mesh_kpts)

        kpts = np.loadtxt(mesh_kpts_path / "farthest.txt")
        center = [np.loadtxt(mesh_kpts_path / "center.txt")]

        self.mesh_kpts = np.concatenate([kpts, center], axis=0)

        print("Initialized 6IMPOSE Dataset.")
        print(f"\t# of all images: {total_n_imgs}")
        print(f"\tCls root: {self.data_root}")
        print(f"\t# of images for this split: {len(self.file_ids)}")
        print(f"\t# of augmented datapoints: {len(self)}")
        # print(f"\nIntrinsic matrix: {self.intrinsic_matrix}")
        print()

    def to_tf_dataset(self):
        if self.use_cache:
            cache_name = "train" if self.is_train else "val"
            cache_path = self.data_root / f"cache_{cache_name}"
            try:
                tfds = self.from_cache(cache_path)
            except FileNotFoundError:
                self.cache(cache_path)
                tfds = self.from_cache(cache_path)

            print("USING CACHE FROM ", cache_path)

            def arrange_as_xy_tuple(d):
                return (d["rgb"], d["rgb_R"], d["baseline"], d["intrinsics"], d["roi"], d["mesh_kpts"]), (
                    d["depth"],
                    d["RT"],
                    d["mask"],
                )

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
        rgb_R = example["rgb"]
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
        from models.stereopvn3d import StereoPvn3d

        num_samples = st.select_slider("num_samples", [2**i for i in range(5, 13)])
        margin = st.slider("margin", 0, 200, 0, step=50)

        h, w = rgb.shape[:2]
        _bbox, _crop_factor = StereoPvn3d.get_crop_index(
            bboxes[None], h, w, 160, 160
        )  # bbox: [b, 4], crop_factor: [b]

        print(f"Intrinsics.shape: {intrinsics.shape} - intrinsics: {intrinsics} - intrisics[None]: {intrinsics[None].astype(np.float32)}")
        # xyz: [b, num_points, 3]
        # inds:  # [b, num_points, 3] with last 3 is index into original image
        xyz, feats, inds = StereoPvn3d.pcld_processor_tf(
            (rgb[None] / 255.0).astype(np.float32),
            depth[None].astype(np.float32),
            intrinsics[None].astype(np.float32),
            _bbox,
            num_samples,
        )
        mask_selected = tf.gather_nd(mask[None], inds)
        print(f"mask_selected.shape after gather_nd {mask_selected.shape}")
        mask_selected = tf.expand_dims(mask_selected, axis=-1)
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
        return len(self.file_ids) * self.n_aug_per_image * self.n_objects_per_image
    
    

    def __getitem__(self, idx):
        # this cycles n_aug_per_image times through the dataset
        i = self.file_ids[idx % len(self.file_ids)]  # 0,1,2,3... i, 0, 1,2,..i
        object_index = (
            idx // len(self.file_ids) % self.n_objects_per_image
        )  # 0,0,0,0,1,1,1,1,2,2,2,2

        print(f"idx: {idx}")
        print(f"len(self.file_ids): {len(self.file_ids)}")
        print(f"self.n_objects_per_image: {self.n_objects_per_image}")
        print(f"__getitem__ object_index: {object_index}")

        rgb = self.get_rgb(i)
        rgb_R = self.get_rgb_R(i)
        depth = self.get_depth(i)
        mask_visib = self.get_mask(i)

        # always add depth background
        # depth = add_background_depth(depth[tf.newaxis, :, :])[0].numpy()  # add and remove batch

        # 6IMPOSE data augmentation
        if self.if_augment:
            # depth = augment_depth(depth[tf.newaxis])[0].numpy()
            rgb = augment_rgb(rgb.astype(np.float32) / 255.0) * 255

        with open(os.path.join(self.data_root, "gt", f"gt_{i:05}.json")) as f:
            shot = json.load(f)

        intrinsics = np.array(shot["cam_matrix"])
        baseline = np.array(shot["stereo_baseline"])

        cam_quat = shot["cam_rotation"]
        cam_rot = R.from_quat(cam_quat)
        cam_pos = np.array(shot["cam_location"])
        cam_Rt = np.eye(4)
        cam_Rt[:3, :3] = cam_rot.as_matrix().T
        cam_Rt[:3, 3] = -cam_rot.as_matrix() @ cam_pos

        all_objs = shot["objs"]
        cls_objs = [obj for obj in all_objs if obj["class"] == self.cls_type]
        # filter out objects that are too close to the image border
        # too close means one of the bbox corners is one the edge

        h, w = rgb.shape[:2]
        objs = [
            obj
            for obj in all_objs
            if obj["bbox_visib"][0] > 0
            and obj["bbox_visib"][1] > 0
            and obj["bbox_visib"][2] < w - 1
            and obj["bbox_visib"][3] < h - 1
        ]

        if len(objs) == 0:
            # crude fix. If all objects clip the image, then so be it
            # probably camera is too close in these cases
            objs = cls_objs

        px_count = lambda x: x["px_count_valid"]
        objs = sorted(objs, key=px_count, reverse=True)
        #print(f"objs: {objs}")
        # print(f"before filter - len(objs): {len(objs)} - objs: {objs}")

        # if len(objs)<=1:
        #     # # valid_fract = lambda x: x['valid_fract']
        #     objs = [el for el in objs if el['visib_fract']>0.3]
        #     print(f"after filter - len(objs): {len(objs)} - objs: {objs}")

        obj = objs[object_index % len(objs)]
        instance_id = obj["object id"]
        pos = np.array(obj["pos"])
        rot = R.from_quat(obj["rotation"])
        RT = np.eye(4)
        RT[:3, :3] = cam_rot.as_matrix().T @ rot.as_matrix()
        RT[:3, 3] = cam_rot.as_matrix().T @ (pos - cam_pos)

        bbox = obj["bbox_visib"]
        bbox = np.array((bbox[1], bbox[0], bbox[3], bbox[2]))  # change order to y1,x1,y2,x2
        mask = np.where(mask_visib == instance_id, 1, 0)

        if self.add_bbox_noise:
            bbox += np.random.randint(-self.bbox_noise, self.bbox_noise, size=bbox.shape)
        
        return {
            "rgb": rgb.astype(np.uint8),
            "rgb_R": rgb_R.astype(np.uint8),
            "intrinsics": intrinsics.astype(np.float32),
            "baseline": baseline.astype(np.float32),
            "depth": depth.astype(np.float32),
            "roi": bbox.astype(np.int32),
            "RT": RT.astype(np.float32),
            "mask": mask.astype(np.uint8),
            "mesh_kpts": self.mesh_kpts.astype(np.float32),
        }

    def get_rgb(self, index) -> np.ndarray:
        rgb_path = os.path.join(self.data_root, "rgb", f"rgb_{index:04}.png")
        with Image.open(rgb_path) as rgb:
            rgb = np.array(rgb).astype(np.uint8)
        return rgb
    
    def get_rgb_R(self, index)-> np.ndarray:
        rgb_path = os.path.join(self.data_root, "rgb", f"rgb_{index:04}_R.png")
        with Image.open(rgb_path) as rgb:
            rgb = np.array(rgb).astype(np.uint8)
        return rgb

    def get_mask(self, index) -> np.ndarray:
        # with pathlib.Path(self.data_root).joinpath(f"mask/mask_{index:04}.exr").open("rb") as F:
        #     reader = minexr.load(F)
        # mask = reader.select(["visib.R"]).astype(np.uint8)
        mask_path = pathlib.Path(self.data_root).joinpath(f"mask/mask_{index:04}.exr")
        mask = EXR(mask_path).read("visib.R").astype(np.uint8)
        return mask

    def get_depth(self, index) -> np.ndarray:
        depth_path = os.path.join(self.data_root, "depth", f"depth_{index:04}.exr")
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth = depth[:, :, :1]
        depth_mask = depth < 5  # in meters, we filter out the background( > 5m)
        depth = depth * depth_mask
        return depth
    
 
    def get_label_list(self, mask, sampled_index):

        """
        params: index: index for mask.png
                sampled_index: indexes for points selected
        return: a list of point-wise label
        """

        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        mask_selected = mask.flatten()[sampled_index].astype("uint8")
        label_list = []

        if self.cls_type == 'all':
            for i in mask_selected:
                label = np.zeros(shape=self.n_classes)
                cls_id = np.where(self.mask_value_array == i)[0]
                if len(cls_id) == 0:
                    label[0] = 1
                else:
                    label[cls_id] = 1

                label_list.append(label)
        else:
            mask_selected = (mask_selected > 0).astype("uint8")
            for i in mask_selected:
                label = np.zeros(shape=self.n_classes)
                cls_id = np.where(self.mask_binary_array == i)[0]
                label[cls_id] = 1
                label_list.append(label)

        return label_list, mask_selected
    

    def choose_index(self, pcld_index):
        """
        pcld_index: an array: 1 X N
        """
        print(f"len(pcld_index: {len(pcld_index)})")
        if len(pcld_index) < 400:
            return None

        pcld_index_id = np.array([i for i in range(len(pcld_index))])

        if len(pcld_index_id) > self.n_sample_points:
            c_mask = np.zeros(len(pcld_index_id), dtype=int)
            c_mask[:self.n_sample_points] = 1
            np.random.shuffle(c_mask)
            pcld_index_id = pcld_index_id[c_mask.nonzero()]
        else:
            pcld_index_id = np.pad(pcld_index_id, (0, self.n_sample_points - len(pcld_index_id)), "wrap")

        return pcld_index_id


class Train6IMPOSE(_6IMPOSE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, if_augment=True, is_train=True, **kwargs) # if_augment original was True


class Val6IMPOSE(_6IMPOSE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, if_augment=False, is_train=False, **kwargs)
