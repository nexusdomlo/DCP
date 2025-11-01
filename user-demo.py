# inference_dcp.py
# 使用方法:
# python3 inference_dcp.py --checkpoint /path/to/checkpoint.pth --src src.pcd --tgt tgt.pcd --npoints 1024 --out aligned_src.pcd --vis [none|minimal|animation|full]

import argparse
import numpy as np
import open3d as o3d
import torch
import time
from scipy.spatial.transform import Rotation as R

def load_pcd_as_numpy(path, npoints=1024):
    p = o3d.io.read_point_cloud(path)
    pts = np.asarray(p.points).astype(np.float32)
    # 保存原始点云用于可视化
    original = pts.copy()
    # 随机下采样或重复补点到 npoints
    if pts.shape[0] >= npoints:
        idx = np.random.choice(pts.shape[0], npoints, replace=False)
    else:
        idx = np.random.choice(pts.shape[0], npoints, replace=True)
    pts = pts[idx]
    return pts, original[idx] if pts.shape[0] == original.shape[0] else original

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

def save_transformed(src_pts, R, t, out_path):
    pts = (R @ src_pts.T).T + t.reshape(1,3)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(out_path, pc)

def try_import_model():
    # 仓库中 model.py 定义了 DCP，直接从本地导入
    try:
        from model import DCP as DCPModel
        return DCPModel
    except Exception as e:
        raise ImportError("请从 model.py 导入 DCP,错误: " + str(e))
    

def visualize_initial(src_norm, tgt_norm, window_name="归一化空间初始点云"):
    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(src_norm)
    pcd_src.paint_uniform_color([1, 0, 0])  # 红色-源点云
    
    pcd_tgt = o3d.geometry.PointCloud()
    pcd_tgt.points = o3d.utility.Vector3dVector(tgt_norm)
    pcd_tgt.paint_uniform_color([0, 1, 0])  # 绿色-目标点云
    
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    o3d.visualization.draw_geometries([pcd_src, pcd_tgt, coord_frame], window_name=window_name)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--src", required=True)
    parser.add_argument("--tgt", required=True)
    parser.add_argument("--npoints", type=int, default=1024)
    parser.add_argument("--out", default="aligned_src.pcd")

    args = parser.parse_args()

    # src, tgt = load_pcds_with_reference(args.src, args.tgt, args.npoints)
    src, _ = load_pcd_as_numpy(args.src, args.npoints)
    tgt, _ = load_pcd_as_numpy(args.tgt, args.npoints)
    print(args.npoints, "个点加载完成")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
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

    R = R.cpu().numpy().squeeze()
    t = t.cpu().numpy().squeeze()
    transform_4x4 = np.eye(4)
    transform_4x4[:3, :3] = R
    transform_4x4[:3, 3] = t
    print("4x4变换矩阵:\n", transform_4x4)
    save_transformed(src, R, t, args.out)
    print("保存对齐结果到", args.out)
    # 在保存结果后添加


if __name__ == "__main__":
    main()
