# (这部分代码将被放入 dcp_icp_pipeline.py)
import argparse
import numpy as np
import open3d as o3d
import torch
import time
from scipy.spatial.transform import Rotation as R
    
# import open3d.t.pipelines.registration as treg
def load_pcd(path):
    # 支持 .pcd/.ply/.xyz/.ply 等图形文件或 .npy 点阵文件
    if path.lower().endswith(('.pcd', '.ply', '.xyz', '.xyzn', '.pts')):
        pcd = o3d.io.read_point_cloud(path)
        if pcd.is_empty():
            raise RuntimeError(f"读取点云失败或为空: {path}")
        return pcd
    elif path.lower().endswith('.npy'):
        pts = np.load(path)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd
    else:
        raise ValueError("不支持的文件格式: " + path)
# ...existing code...


def load_pcds_with_reference(src_path, tgt_path, npoints=1024):
    # tgt为大点云A，src为小点云B
    tgt_pcd = o3d.io.read_point_cloud(tgt_path)
    tgt_pts = np.asarray(tgt_pcd.points).astype(np.float32)
    if tgt_pts.shape[0] >= npoints:
        tgt_idx = np.random.choice(tgt_pts.shape[0], npoints, replace=False)
    else:
        tgt_idx = np.random.choice(tgt_pts.shape[0], npoints, replace=True)
    tgt_pts_sampled = tgt_pts[tgt_idx]
    # 保存原始点云用于可视化
    tgt_original = tgt_pts.copy()
    
    # 计算A的中心和缩放因子
    centroid = tgt_pts_sampled.mean(axis=0, keepdims=True)
    tgt_pts_centered = tgt_pts_sampled - centroid
    scale = np.max(np.linalg.norm(tgt_pts_centered, axis=1))
    tgt_pts_norm = tgt_pts_centered / (scale + 1e-9)

    # 对B用A的中心和缩放因子处理
    src_pcd = o3d.io.read_point_cloud(src_path)
    src_pts = np.asarray(src_pcd.points).astype(np.float32)
    if src_pts.shape[0] >= npoints:
        src_idx = np.random.choice(src_pts.shape[0], npoints, replace=False)
    src_pts = src_pts[src_idx]
    src_pts_norm = (src_pts - centroid) / (scale + 1e-9)

    return src_pts_norm, tgt_pts_norm

def try_import_model():
    # 仓库中 model.py 定义了 DCP，直接从本地导入
    try:
        from model import DCP as DCPModel
        return DCPModel
    except Exception as e:
        raise ImportError("请从 model.py 导入 DCP,错误: " + str(e))
    
def refine_icp(src, tgt, init_trans, voxel_size):
    distance_threshold = voxel_size * 0.4
    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    result = o3d.pipelines.registration.registration_icp(
        src, tgt, distance_threshold, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

# def refine_icp_gpu(src, tgt, init_trans, voxel_size):
#     """
#     使用 Open3D 的 Tensor API 在 GPU 上执行 ICP。
#     """
#     # 检查是否有可用的 CUDA 设备
#     if not o3d.core.cuda.is_available():
#         raise RuntimeError("CUDA is not available. Please check your Open3D installation and CUDA setup.")

#     device = o3c.Device("CUDA:0")
    
#     # 将初始变换矩阵转换为 GPU 上的张量
#     init_trans_tensor = o3c.Tensor(init_trans, device=device)
#      # 将 open3d.geometry.PointCloud 转换为 open3d.t.geometry.PointCloud
#     # 并将其发送到 GPU
#     src_t = o3d.t.geometry.PointCloud.from_legacy(src, o3c.float64, device)
#     tgt_t = o3d.t.geometry.PointCloud.from_legacy(tgt, o3c.float64, device)
#     # 在 GPU 上估计法线
#     src_t.estimate_normals()
#     tgt_t.estimate_normals()
#     # 设置 ICP 的收敛标准
#     criteria = treg.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=30)
#     # 在 GPU 上运行 ICP
#     result = treg.icp(
#         src_t, 
#         tgt_t, 
#         voxel_size * 0.4,  # 对应 CPU 版本中的 distance_threshold
#         init_trans_tensor,
#         treg.TransformationEstimationPointToPlane(),
#         criteria
#     )
#     # 将结果从 GPU 张量转换回 NumPy 数组
#     return result.transformation.cpu().numpy(), result.fitness, result.inlier_rmse


def run_dcp_coarse_registration(src_path, tgt_path, checkpoint_path, npoints=1024):
    """
    运行 DCP 模型进行粗配准，并返回 4x4 变换矩阵。
    """
    # 1. 加载并归一化点云
    src, tgt = load_pcds_with_reference(src_path, tgt_path, npoints)
    # src = load_pcd_as_numpy(args.src, args.npoints)
    # tgt = load_pcd_as_numpy(args.tgt, args.npoints)
    print(npoints, "个点加载完成")
    # 2. 加载并运行 DCP 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)
    ckpt = torch.load(args.dcp_checkpoint, map_location=device, weights_only=True)
    ckpt_args = ckpt.get('args', None)

    DCPModel = try_import_model()
    # 修改这里：构造一个 args 对象
    if ckpt_args is not None:
        model_args = ckpt_args
    else:
        model_args = argparse.Namespace(
            emb_nn='dgcnn',
            pointer='transformer',
            head='svd',
            emb_dims=512,
            n_blocks=1,
            n_heads=4,
            ff_dims=1024,
            dropout=0.0,
            cycle=False
        )
    model = DCPModel(model_args)
    model.to(device)
    model.eval()

    # 3. 模型推理
    src_t = torch.from_numpy(src).unsqueeze(0).to(device)
    tgt_t = torch.from_numpy(tgt).unsqueeze(0).to(device)
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    with torch.no_grad():
        try:
            out = model(src_t.transpose(2, 1), tgt_t.transpose(2, 1))
        except Exception as e:
            raise RuntimeError("模型 forward 调用失败: " + str(e))

    if isinstance(out, (list, tuple)):
        R, t = out[0], out[1]
    else:
        raise RuntimeError("模型输出格式未知，请打印 out 查看")
    # 4. 组合成 4x4 变换矩阵
    transform_4x4 = np.eye(4)
    transform_4x4[:3, :3] = R.cpu().numpy().squeeze()
    transform_4x4[:3, 3] = t.cpu().numpy().squeeze()

    return transform_4x4

# 我们将重用 refine_icp, refine_icp_gpu, load_pcd 等函数
# ... (从 ../ICP/demo.py 复制过来) ...

def dcp_icp_pipeline(src_path, tgt_path, dcp_checkpoint, voxel_size=0.1, use_gpu_icp=False):
    """
    整合了 DCP 和 ICP 的完整流程。
    """
    # 1. 使用 DCP 进行粗配准
    print("--- Running DCP for Coarse Registration ---")
    init_trans = run_dcp_coarse_registration(src_path, tgt_path, dcp_checkpoint)
    print("DCP Coarse Transformation:\n", init_trans)
    # 2. 加载原始点云用于 ICP 精配准
    src = load_pcd(src_path)
    tgt = load_pcd(tgt_path)
    
    # 3. 使用 ICP 进行精配准
    print("\n--- Refining with ICP ---")
    if use_gpu_icp:
        print("Refining with ICP on GPU... sorry , currently disabled.")
        # try:
        #     # 注意：确保您的 Open3D 支持 CUDA！
        #     T_icp, fitness, rmse = refine_icp_gpu(src, tgt, init_trans, voxel_size)
        #     print("GPU ICP fitness:", fitness, "rmse:", rmse)
        #     final_transformation = T_icp
        # except Exception as e:
        #     print(f"GPU ICP failed: {e}. Falling back to CPU.")
        #     result_icp = refine_icp(src, tgt, init_trans, voxel_size)
        #     print("CPU ICP fitness:", result_icp.fitness, "rmse:", result_icp.inlier_rmse)
        #     final_transformation = result_icp.transformation
    else:
        result_icp = refine_icp(src, tgt, init_trans, voxel_size)
        print("CPU ICP fitness:", result_icp.fitness, "rmse:", result_icp.inlier_rmse)
        final_transformation = result_icp.transformation

    print("\nFinal Transformation (DCP + ICP):\n", final_transformation)
    
    # 4. 可视化最终结果
    src_transformed = src.transform(final_transformation)
    src_transformed.paint_uniform_color([1, 0, 0]) # 红色
    tgt.paint_uniform_color([0, 1, 0]) # 绿色
    o3d.visualization.draw_geometries([src_transformed, tgt])

    return final_transformation

# (这部分代码将被放入 dcp_icp_pipeline.py)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCP+ICP Registration Pipeline")
    parser.add_argument("--src", required=True, help="源点云文件")
    parser.add_argument("--tgt", required=True, help="目标点云文件")
    parser.add_argument("--dcp_checkpoint", required=True, help="预训练的 DCP 模型权重路径")
    parser.add_argument("--voxel_size", type=float, default=0.1, help="ICP 使用的体素大小")
    parser.add_argument("--use_gpu_icp", action="store_true", help="为 ICP 步骤使用 GPU (需要 CUDA 版 Open3D)")
    args = parser.parse_args()

    dcp_icp_pipeline(
        args.src,
        args.tgt,
        args.dcp_checkpoint,
        voxel_size=args.voxel_size,
        use_gpu_icp=args.use_gpu_icp
    )
