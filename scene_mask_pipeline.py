import os
import cv2
import torch
import numpy as np
import open3d as o3d
import argparse
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
import json


# ==============================================================================
# 1. Projection Function
# ==============================================================================
def project2img(pc, depth_img, pose, intrinsic, height, width):
    """
    Projection function logic optimized for memory.
    Args:
        pc: (N, 3) numpy array
        depth_img: (H, W) uint16 numpy array (raw mm)
        pose: (3, 4) or (4, 4) numpy array (World-to-Camera)
        intrinsic: (3, 3) numpy array
        height, width: int (Image dimensions)
    """

    points_3d_hom = np.hstack((pc, np.ones((pc.shape[0], 1))))

    # processing pose to 3x4
    pose_3x4 = pose[:3, :] if pose.shape == (4, 4) else pose

    # cam_coords = (pose @ world_coords)
    # intrinsic @ cam_coords -> pixel coords

    cam_coords = pose_3x4 @ points_3d_hom.T

    point_depth_vals = cam_coords[2, :]

    projected_points = intrinsic @ cam_coords

    valid_z = point_depth_vals > 0
    if not np.any(valid_z):
        return np.array([]), np.array([])

    projected_points = projected_points[:, valid_z]
    point_depth_vals = point_depth_vals[valid_z]

    u = (projected_points[0, :] / projected_points[2, :]).astype(int)
    v = (projected_points[1, :] / projected_points[2, :]).astype(int)

    valid_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)

    u = u[valid_bounds]
    v = v[valid_bounds]
    point_depth_vals = point_depth_vals[valid_bounds]

    if len(u) == 0:
        return np.array([]), np.array([])

    sensor_depth_val = depth_img[v, u] / 1000.0

    diff = np.abs(sensor_depth_val - point_depth_vals)
    visibility_mask = diff <= 0.2

    visible_pixels = np.stack((u[visibility_mask], v[visibility_mask]), axis=1)

    boundary_pixels = np.stack((u, v), axis=1)

    return boundary_pixels, visible_pixels


# ==============================================================================
# 2. Data Loader
# ==============================================================================
class SceneDataLoader:
    def __init__(self, scene_id, scannet_rgbd_root, stride=5, height=480, width=640):
        self.scene_id = scene_id
        self.rgbd_root = os.path.join(scannet_rgbd_root, scene_id)
        self.height = height
        self.width = width
        self.stride = stride

        self.xyz = None
        self.intrinsics = None


        self.rgb_files = []
        self.depth_imgs_narrow = []
        self.w2c_poses_narrow = []

        self._load_metadata()
        self._preload_depth_and_pose()

    def _load_metadata(self):
        # load intrinsics
        try:
            k_path = os.path.join(self.rgbd_root, "intrinsics", "intrinsic_color.txt")
            self.intrinsics = np.loadtxt(k_path)[:3, :3]
        except:
            print("Warning: Intrinsic not found, using default.")
            self.intrinsics = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])

        # load point cloud
        ply_path = os.path.join(self.rgbd_root, f"{self.scene_id}_vh_clean_2.ply")
        if not os.path.exists(ply_path):
            ply_path = os.path.join(self.rgbd_root, f"{self.scene_id}.ply")

        if os.path.exists(ply_path):
            pcd = o3d.io.read_point_cloud(ply_path)
            self.xyz = np.asarray(pcd.points)
        else:
            print(f"[Error] PLY not found: {ply_path}")
            self.xyz = np.zeros((1, 3))

    def _preload_depth_and_pose(self):
        color_dir = os.path.join(self.rgbd_root, "color")
        depth_dir = os.path.join(self.rgbd_root, "depth")
        pose_dir = os.path.join(self.rgbd_root, "pose")

        # get all files and sort
        all_rgb = sorted(os.listdir(color_dir), key=lambda x: int(os.path.splitext(x)[0]))
        all_depth = sorted(os.listdir(depth_dir), key=lambda x: int(os.path.splitext(x)[0]))
        all_pose = sorted(os.listdir(pose_dir), key=lambda x: int(os.path.splitext(x)[0]))

        # stride selection
        selected_indices = range(0, len(all_rgb), self.stride)

        print(f"Loading data with stride={self.stride}. Total frames: {len(selected_indices)}")

        # read first image to check size
        first_rgb_path = os.path.join(color_dir, all_rgb[0])
        first_img = cv2.imread(first_rgb_path)
        h_raw, w_raw = first_img.shape[:2]

        if (h_raw, w_raw) != (self.height, self.width):
            scale_x = self.width / w_raw
            scale_y = self.height / h_raw
            self.intrinsics[0, 0] *= scale_x
            self.intrinsics[1, 1] *= scale_y
            self.intrinsics[0, 2] *= scale_x
            self.intrinsics[1, 2] *= scale_y

        for i in tqdm(selected_indices, desc="Pre-loading Depth & Pose"):
            # 1. save RGB file paths
            self.rgb_files.append(os.path.join(color_dir, all_rgb[i]))

            # 2. process pose
            pose = np.loadtxt(os.path.join(pose_dir, all_pose[i]))
            if np.isinf(pose).any() or np.isnan(pose).any():
                w2c = np.eye(4)[:3, :]
            else:
                w2c = np.linalg.inv(pose)[:3, :]
            self.w2c_poses_narrow.append(w2c)

            # 3. process depth
            d_path = os.path.join(depth_dir, all_depth[i])
            d_img = cv2.imread(d_path, cv2.IMREAD_UNCHANGED)
            if d_img.shape[:2] != (self.height, self.width):
                d_img = cv2.resize(d_img, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            self.depth_imgs_narrow.append(d_img)

    def get_rgb_image(self, index):
        """Lazy loader for RGB image"""
        path = self.rgb_files[index]
        img = cv2.imread(path)
        if img.shape[:2] != (self.height, self.width):
            img = cv2.resize(img, (self.width, self.height))
        return img


# ==============================================================================
# 3. Pipeline
# ==============================================================================
class MaskPipeline:
    def __init__(self, data_loader, mask_predictor):
        self.loader = data_loader
        self.predictor = mask_predictor

    def _get_bbox(self, pts, h, w):
        if len(pts) == 0: return 0, 0, w, h
        pts = np.array(pts)
        x_min, x_max = np.min(pts[:, 0]), np.max(pts[:, 0])
        y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])
        padding = 10
        return (max(0, int(x_min - padding)), max(0, int(y_min - padding)),
                min(w, int(x_max + padding)), min(h, int(y_max + padding)))

    def _save_5_images(self, save_dir, idx, rgb_img, best_mask, visible_pts):
        # function to save 2d representation
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        # 1. RGB
        cv2.imwrite(os.path.join(save_dir, f"{idx}_rgb.png"), rgb_img)

        # 2. Crop
        h, w = rgb_img.shape[:2]
        x1, y1, x2, y2 = self._get_bbox(visible_pts, h, w)
        if x2 > x1 and y2 > y1:
            cv2.imwrite(os.path.join(save_dir, f"{idx}_crop.png"), rgb_img[y1:y2, x1:x2])

        # 3. BBox
        bbox_img = rgb_img.copy()
        cv2.rectangle(bbox_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(save_dir, f"{idx}_bbox.png"), bbox_img)

        # 4. Mask Points Projection
        pts_img = rgb_img.copy()
        for pt in visible_pts:
            cv2.circle(pts_img, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
        cv2.imwrite(os.path.join(save_dir, f"{idx}_mask.png"), pts_img)

        # 5. SAM Overlay
        green_img = rgb_img.copy()
        color_mask = np.zeros_like(green_img)
        color_mask[best_mask] = [0, 255, 0]

        mask_bool = best_mask.astype(bool)
        green_img[mask_bool] = cv2.addWeighted(green_img[mask_bool], 0.5, color_mask[mask_bool], 0.5, 0)
        cv2.imwrite(os.path.join(save_dir, f"{idx}_sam_mask.png"), green_img)

    def run(self, mask_path, save_dir):
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        print(f"Loading masks from {mask_path}")
        if mask_path.endswith('.pt'):
            masks_3d = torch.load(mask_path)
            if isinstance(masks_3d, dict): masks_3d = masks_3d['masks']
            if isinstance(masks_3d, torch.Tensor): masks_3d = masks_3d.numpy()
            masks_3d = masks_3d.T
        elif mask_path.endswith('.npy'):
            masks_3d = np.load(mask_path)
        else:
            raise ValueError("Unknown mask format")

        xyz = self.loader.xyz
        depths = self.loader.depth_imgs_narrow
        poses = self.loader.w2c_poses_narrow
        K = self.loader.intrinsics
        H, W = self.loader.height, self.loader.width

        # process each mask
        for idx in tqdm(range(masks_3d.shape[0]), desc="Processing Masks"):
            mask_ind = np.where(masks_3d[idx] != 0)[0]
            if len(mask_ind) < 20: continue  # 忽略太小的mask

            obj_points = xyz[mask_ind]

            best_score = -1
            best_visible_pts = []
            best_frame_idx = -1

            # --- top views selection ---
            for i in range(len(poses)):
                _, visible_pts = project2img(
                    obj_points, depths[i], poses[i], K, H, W
                )

                score = len(visible_pts)
                if score > best_score:
                    best_score = score
                    best_visible_pts = visible_pts
                    best_frame_idx = i

            if best_frame_idx == -1 or len(best_visible_pts) == 0:
                continue

            rgb_image_best = self.loader.get_rgb_image(best_frame_idx)

            # --- run SAM ---
            rgb_for_sam = cv2.cvtColor(rgb_image_best, cv2.COLOR_BGR2RGB)
            self.predictor.predictor.set_image(rgb_for_sam)

            best_sam_mask = None
            max_sam_iou = -1.0
            pts_pool = np.array(best_visible_pts)

            for _ in range(5):
                if len(pts_pool) < 2: break

                n_sample = min(5, len(pts_pool))
                indices = np.random.choice(len(pts_pool), n_sample, replace=False)
                sel_pts = pts_pool[indices]

                masks, scores, _ = self.predictor.predictor.predict(
                    point_coords=sel_pts,
                    point_labels=np.ones(len(sel_pts)),
                    multimask_output=True
                )

                curr_best_idx = np.argmax(scores)
                if scores[curr_best_idx] > max_sam_iou:
                    max_sam_iou = scores[curr_best_idx]
                    best_sam_mask = masks[curr_best_idx]

            if best_sam_mask is not None:
                self._save_5_images(save_dir, idx, rgb_image_best, best_sam_mask, best_visible_pts)

            # 显式清理
            del rgb_image_best
            # torch.cuda.empty_cache()


# ==============================================================================
# 4. Main Execution
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate 2D visualizations from 3D masks.")
    parser.add_argument('--config', type=str, default="config.json", help='Path to shared JSON config')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, 'r') as f:
        config = json.load(f)

    paths = config['paths']
    params = config['parameters']

    scannet_root = paths['scannet_root']
    mask_3d_dir = paths['mask_3d_dir']
    output_dir = paths['mask_2d_save_dir']
    scene_txt = paths['scene_list_file']
    sam_ckpt = paths['sam_checkpoint']
    stride = params['stride']

    if not os.path.exists(sam_ckpt):
        raise FileNotFoundError(f"SAM checkpoint not found: {sam_ckpt}")

    # initial SAM
    print("Initializing SAM...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt)
    sam.to(device)
    predictor_wrapper = SamPredictor(sam)


    class Wrapper:
        def __init__(self, p): self.predictor = p


    with open(scene_txt, 'r') as f:
        scene_ids = [line.strip() for line in f.readlines()]

    for scene_id in tqdm(scene_ids, desc="Processing Scenes"):
        save_path = os.path.join(output_dir, scene_id)
        mask_path = os.path.join(mask_3d_dir, f"{scene_id}_masks.pt")

        if not os.path.exists(mask_path):
            print(f"[Error] Mask file not found: {mask_path}")
        else:

            print(f"Initializing Scene Loader for {scene_id} with stride {stride}...")
            loader = SceneDataLoader(
                scene_id,
                scannet_root,
                stride=stride,
                height=params['image_height'],
                width=params['image_width']
            )

            pipeline = MaskPipeline(loader, Wrapper(predictor_wrapper))
            pipeline.run(mask_path, save_path)