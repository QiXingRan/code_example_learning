import cv2
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def generate_synthetic_data(img_size=(240, 240), K_params=(300, 300, 160, 120)):
    """
    生成模拟图像和深度图。
    Ref：原图，Curr：位姿变换图
    """
    h, w = img_size
    fx, fy, cx, cy = K_params # 假设相机参数与矩阵k
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float64)
    # 参考图像
    ref_img = np.zeros((h, w), dtype=np.uint8)
    tile_size = 20
    for r in range(h // tile_size):
        for c in range(w // tile_size):
            if (r + c) % 2 == 0:
                ref_img[r * tile_size: (r + 1) * tile_size,
                c * tile_size: (c + 1) * tile_size] = 200
            else:
                ref_img[r * tile_size: (r + 1) * tile_size,
                c * tile_size: (c + 1) * tile_size] = 50
    ref_img = ref_img.astype(np.float32) / 255.0  # 归一化到 [0, 1]
    # 参考深度图（假设图片有深度）
    ref_depth = np.zeros((h, w), dtype=np.float32)
    for r in range(h):
        for c in range(w):
            ref_depth[r, c] = 2.0 + 0.005 * r  # 模拟一个倾斜的平面
    # 模拟一个小的相机位姿变换 (Ref -> Curr)
    # X轴平移0.1米，Z轴平移0.05米
    theta_y = np.deg2rad(0)
    tx = 0.1
    ty = 0.0
    tz = 0.05
    R_gt = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                     [0, 1, 0],
                     [-np.sin(theta_y), 0, np.cos(theta_y)]], dtype=np.float64)
    t_gt = np.array([tx, ty, tz], dtype=np.float64)
    T_ref_to_curr_gt = np.eye(4)
    T_ref_to_curr_gt[:3, :3] = R_gt
    T_ref_to_curr_gt[:3, 3] = t_gt
    # 根据Ref图像和深度图，以及GT位姿，生成Curr图像
    curr_img = np.zeros_like(ref_img)
    curr_depth = np.zeros_like(ref_depth)
    # 遍历Ref图像的每个像素，投影到Curr图像
    for r in range(h):
        for c in range(w):
            z_ref = ref_depth[r, c]
            if z_ref <= 0: continue  # 避免无效深度
            # 3D点在Ref相机坐标系下
            P_ref_cam = np.array([(c - cx) * z_ref / fx,
                                  (r - cy) * z_ref / fy,
                                  z_ref], dtype=np.float64)
            # 变换到Curr相机坐标系
            P_curr_cam = R_gt @ P_ref_cam + t_gt
            z_curr = P_curr_cam[2]
            if z_curr <= 0: continue  # 避免负深度
            # 投影到Curr图像平面
            u_curr = fx * P_curr_cam[0] / z_curr + cx
            v_curr = fy * P_curr_cam[1] / z_curr + cy
            # 如果投影点在图像范围内，则赋值
            if 0 <= u_curr < w and 0 <= v_curr < h:
                curr_img[int(v_curr), int(u_curr)] = ref_img[r, c]
                curr_depth[int(v_curr), int(u_curr)] = z_curr
    return ref_img, ref_depth, curr_img, curr_depth, K, T_ref_to_curr_gt


# --- 2. 辅助函数：将SE(3)表示为6个参数 (李代数se(3)的向量形式) ---
# params = [rx, ry, rz, tx, ty, tz]
def Rodrigues(r_vec):
    """将旋转向量转换为旋转矩阵"""
    theta = np.linalg.norm(r_vec)
    if theta < 1e-6:
        return np.eye(3)
    r_vec_norm = r_vec / theta
    K_skew = np.array([[0, -r_vec_norm[2], r_vec_norm[1]],
                       [r_vec_norm[2], 0, -r_vec_norm[0]],
                       [-r_vec_norm[1], r_vec_norm[0], 0]])
    return np.eye(3) + np.sin(theta) * K_skew + (1 - np.cos(theta)) * (K_skew @ K_skew)


def se3_to_T(params):
    """将6个参数 (rx, ry, rz, tx, ty, tz) 转换为4x4齐次变换矩阵"""
    R = Rodrigues(params[:3])
    t = params[3:]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# --- 3. 光度误差函数 ---
def photometric_error(params, curr_img, K, valid_ref_pixels_3d_coords_cam, valid_ref_pixels_intensity, img_width, img_height):
    """
    计算光度误差。
    params: [rx, ry, rz, tx, ty, tz] - 从参考相机到当前相机的位姿变换。
    curr_img: 当前图像。
    K: 相机内参矩阵。
    valid_ref_pixels_3d_coords_cam: 预选的参考像素在参考相机坐标系下的3D坐标。
    valid_ref_pixels_intensity: 预选的参考像素的强度值。
    img_width, img_height: 图像的宽度和高度，用于边界检查。
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # 从参数构建变换矩阵
    T_ref_to_curr = se3_to_T(params)
    R = T_ref_to_curr[:3, :3]
    t = T_ref_to_curr[:3, 3]
    num_points = valid_ref_pixels_3d_coords_cam.shape[0]
    # 预分配投影点坐标数组
    projected_uv = np.zeros((num_points, 2), dtype=np.float32)
    valid_mask = np.ones(num_points, dtype=bool)
    # 向量化处理3D点变换和投影
    P_curr_cam_all = (R @ valid_ref_pixels_3d_coords_cam.T).T + t
    z_curr_all = P_curr_cam_all[:, 2]
    # 检查深度是否有效 (z_curr > 0)
    valid_mask = valid_mask & (z_curr_all > 1e-6)  # Use a small epsilon to avoid division by near zero
    # 投影到图像平面
    # Avoid division by zero for invalid depths
    u_curr_all = np.full(num_points, -1.0, dtype=np.float32)  # Initialize with out-of-bounds value
    v_curr_all = np.full(num_points, -1.0, dtype=np.float32)  # Initialize with out-of-bounds value
    u_curr_all[valid_mask] = fx * P_curr_cam_all[valid_mask, 0] / z_curr_all[valid_mask] + cx
    v_curr_all[valid_mask] = fy * P_curr_cam_all[valid_mask, 1] / z_curr_all[valid_mask] + cy
    valid_mask = valid_mask & (u_curr_all >= 0) & (u_curr_all < img_width - 1) & \
                 (v_curr_all >= 0) & (v_curr_all < img_height - 1)
    projected_uv[valid_mask] = np.vstack((u_curr_all[valid_mask], v_curr_all[valid_mask])).T
    interpolated_values = np.zeros(num_points, dtype=np.float32)
    for i in range(num_points):
        if valid_mask[i]:
            u, v = projected_uv[i, 0], projected_uv[i, 1]
            try:
                interpolated_values[i] = cv2.getRectSubPix(curr_img, (1, 1), (u, v))[0, 0]
            except cv2.error:
                valid_mask[i] = False
                interpolated_values[i] = 0.0  # Or some default value if interpolation fails
    # 计算光度误差 for all points
    errors = valid_ref_pixels_intensity - interpolated_values
    # 将无效点的误差设置为0 (不贡献到总误差)
    errors[~valid_mask] = 0.0
    return errors


# --- 4. 主执行部分 ---
if __name__ == "__main__":
    ref_img_orig, ref_depth_orig, curr_img_orig, curr_depth_orig, K, T_gt = generate_synthetic_data()
    print("相机内参 K:\n", K)
    print("真实位姿 T_ref_to_curr_gt:\n", T_gt)
    # --- B. 预处理参考图像和深度图，提取有效点 ---
    h, w = ref_img_orig.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # 找到所有有效的参考像素坐标 (r, c)
    valid_pixels_ref_r, valid_pixels_ref_c = np.where(ref_depth_orig > 0)
    # Vectorized computation for P_ref_cam
    z_ref_all = ref_depth_orig[valid_pixels_ref_r, valid_pixels_ref_c]
    u_ref_all = valid_pixels_ref_c.astype(np.float64)
    v_ref_all = valid_pixels_ref_r.astype(np.float64)

    X_ref_cam = (u_ref_all - cx) * z_ref_all / fx
    Y_ref_cam = (v_ref_all - cy) * z_ref_all / fy

    valid_ref_pixels_3d_coords_cam = np.vstack((X_ref_cam, Y_ref_cam, z_ref_all)).T
    valid_ref_pixels_intensity = ref_img_orig[valid_pixels_ref_r, valid_pixels_ref_c]
    # 初始化位姿参数 (稍微偏离真实值，模拟初始估计)
    # rx, ry, rz, tx, ty, tz
    initial_params = np.array([0.05, -0.01, 0.02, 0.05, 0.01, 0.03], dtype=np.float64)  # 稍微扰动一下

    print("\n初始位姿参数:", initial_params)
    print("初始位姿矩阵 T_init:\n", se3_to_T(initial_params))

    # 使用least_squares进行优化
    print("\n开始优化...")
    result = least_squares(photometric_error,
                           initial_params,
                           args=(curr_img_orig, K,
                                 valid_ref_pixels_3d_coords_cam,
                                 valid_ref_pixels_intensity,
                                 w, h),  # Pass image width and height
                           verbose=2,  # 打印优化过程
                           ftol=1e-6,  # 函数值容差
                           xtol=1e-6,  # 参数值容差
                           max_nfev=100)  # 最大函数评估次数

    optimized_params = result.x
    optimized_T = se3_to_T(optimized_params)

    print("\n优化完成！")
    print("估计位姿参数:", optimized_params)
    print("估计位姿矩阵 T_estimated:\n", optimized_T)

    print("\n真实位姿矩阵 T_gt:\n", T_gt)

    # 评估结果
    diff_R = np.linalg.norm(optimized_T[:3, :3] - T_gt[:3, :3])
    diff_t = np.linalg.norm(optimized_T[:3, 3] - T_gt[:3, 3])
    print(f"\n旋转矩阵差异 (Frobenius norm): {diff_R:.6f}")
    print(f"平移向量差异 (Euclidean norm): {diff_t:.6f}")

    # --- D. 可视化投影效果 ---
    # 1. 初始位姿的投影效果
    initial_T = se3_to_T(initial_params)
    projected_img_initial = np.zeros_like(curr_img_orig)
    for i in range(len(valid_ref_pixels_3d_coords_cam)):
        P_ref_cam = valid_ref_pixels_3d_coords_cam[i]
        ref_intensity = valid_ref_pixels_intensity[i]
        P_curr_cam = initial_T[:3, :3] @ P_ref_cam + initial_T[:3, 3]
        z_curr = P_curr_cam[2]
        if z_curr > 0:
            u_curr = fx * P_curr_cam[0] / z_curr + cx
            v_curr = fy * P_curr_cam[1] / z_curr + cy
            if 0 <= u_curr < w and 0 <= v_curr < h:
                projected_img_initial[int(v_curr), int(u_curr)] = ref_intensity
    # 2. 优化后位姿的投影效果
    projected_img_optimized = np.zeros_like(curr_img_orig)
    for i in range(len(valid_ref_pixels_3d_coords_cam)):
        P_ref_cam = valid_ref_pixels_3d_coords_cam[i]
        ref_intensity = valid_ref_pixels_intensity[i]

        P_curr_cam = optimized_T[:3, :3] @ P_ref_cam + optimized_T[:3, 3]
        z_curr = P_curr_cam[2]
        if z_curr > 0:
            u_curr = fx * P_curr_cam[0] / z_curr + cx
            v_curr = fy * P_curr_cam[1] / z_curr + cy
            if 0 <= u_curr < w and 0 <= v_curr < h:
                projected_img_optimized[int(v_curr), int(u_curr)] = ref_intensity