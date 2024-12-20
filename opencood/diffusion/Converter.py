import numpy as np
import cv2
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)
# 7x7 diamond kernel
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ],
    dtype=np.uint8,
)
def fill_in_fast(depth_map, max_depth=15.0, custom_kernel=DIAMOND_KERNEL_7, extrapolate=False, blur_type="bilateral"):
    """Fast, in-place depth completion.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        custom_kernel: kernel to apply initial dilation
        extrapolate: whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'bilateral' - preserves local structure (recommended)
            'gaussian' - provides lower RMSE

    Returns:
        depth_map: dense depth map
    """

    # Invert
    valid_pixels = depth_map > 0.1
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # Dilate
    depth_map = cv2.dilate(depth_map, custom_kernel)

    # Hole closing
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Fill empty spaces with dilated values
    empty_pixels = depth_map < 0.1
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image
    if extrapolate:
        top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
        top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]

        for pixel_col_idx in range(depth_map.shape[1]):
            depth_map[0 : top_row_pixels[pixel_col_idx], pixel_col_idx] = top_pixel_values[pixel_col_idx]

        # Large Fill
        empty_pixels = depth_map < 0.1
        dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
        depth_map[empty_pixels] = dilated[empty_pixels]

    # Median blur
    depth_map = cv2.medianBlur(depth_map, 5)

    # Bilateral or Gaussian blur
    if blur_type == "bilateral":
        # Bilateral blur
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
    elif blur_type == "gaussian":
        # Gaussian blur
        valid_pixels = depth_map > 0.1
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    # Invert
    valid_pixels = depth_map > 0.1
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    return depth_map

class Converter:
    def __init__(self):
        pass

    def to_harmonic(self, input: np.ndarray):
        """转换成齐次坐标形式"""
        M = input.shape[0]
        input = np.concatenate([input, np.ones[[M, 1]]], axis=1)
        return input

    def proj_3to2(self, xyz: np.ndarray, extrinsic, intrinsic):
        """
        不懂原理, 直接照抄
        :param xyz: shape: [M, 3]
        :param extrinsic: shape: [3, 3]
        :param intrinsic: shape: [4, 4]
        """
        xyz = self.to_harmonic(xyz)
        xyz = np.linalg.inv(extrinsic) @ xyz.T
        uvd = intrinsic @ xyz[0:3]
        uvd = uvd.T
        uv, d = uvd[:, 0:2] / (uvd[:, -1:] + 1e-5), uvd[:, -1]
        return uv, d

    def proj_pc2dpt(self, point_cloud: np.ndarray, extrinsic, intrinsic, h, w):
        uv, dpt = self.proj_3to2(point_cloud, intrinsic, extrinsic)
        mask_w = (uv[:, 0] < w) & (uv[:, 0] >= 0)
        mask_h = (uv[:, 1] < h) & (uv[:, 1] >= 0)
        # mask mask off the back-project points
        mask_d = dpt > 0.05
        mask = mask_h & mask_w & mask_d
        uv = uv[mask].astype(np.int32)
        dpt = dpt[mask]
        result = np.ones([h, w]) * 10000
        for i in range(uv.shape[0]):
            u, v = uv[i]
            d = dpt[i]
            result[v, u] = min(result[v, u], d)
        result[result > 9999] = 0.0
        return result
