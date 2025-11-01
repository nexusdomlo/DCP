#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import torch
import numpy as np
import open3d as o3d
import os

# 动态导入项目中的DCP模型和工具函数
try:
    from model import DCP
    from util import transform_point_cloud
except ImportError as e:
    print("错误: 无法从本地文件导入'model'或'util'。")
    print("请确保此脚本与 model.py 和 util.py 在同一目录下。")
    raise e

def load_pcd_to_numpy(path, n_points=1024):
    """
    从指定路径加载PCD文件，并下采样/上采样到固定点数。
    """
    try:
        pcd = o3d.io.read_point_cloud(path)
        if not pcd.has_points():
            raise ValueError(f"PCD文件 {path} 为空或加载失败。")
    except Exception as e:
        raise IOError(f"无法读取PCD文件: {path}. 错误: {e}")

    points = np.asarray(pcd.points, dtype=np.float32)
    
    if points.shape[0] < n_points:
        indices = np.random.choice(points.shape[0], n_points, replace=True)
    else:
        indices = np.random.choice(points.shape[0], n_points, replace=False)
        
    sampled_points = points[indices]
    return sampled_points

def save_numpy_to_pcd(points, path):
    """
    将numpy点云数组保存为PCD文件。
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(path, pcd)
    print(f"对齐后的点云已保存到: {path}")

def register_pcds(args):
    """
    执行点云配准的核心函数。
    """
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"正在使用设备: {device}")

    print(f"正在从 '{args.src_path}' 加载源点云...")
    src_points_np = load_pcd_to_numpy(args.src_path, args.num_points)
    
    print(f"正在从 '{args.tgt_path}' 加载目标点云...")
    tgt_points_np = load_pcd_to_numpy(args.tgt_path, args.num_points)
    print(f"点云加载并采样到 {args.num_points} 个点。")

    model_args = argparse.Namespace(
        emb_nn=args.emb_nn,
        pointer=args.pointer,
        head=args.head,
        emb_dims=args.emb_dims,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        ff_dims=args.ff_dims,
        dropout=args.dropout,
        cycle=args.cycle
    )
    net = DCP(model_args).to(device)
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"找不到预训练模型文件: {args.model_path}")
    
    print(f"正在加载预训练模型: {args.model_path}")
    net.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
    net.eval()

    src_tensor = torch.from_numpy(src_points_np).unsqueeze(0).to(device)
    tgt_tensor = torch.from_numpy(tgt_points_np).unsqueeze(0).to(device)

    src_tensor = src_tensor.transpose(2, 1).contiguous()
    tgt_tensor = tgt_tensor.transpose(2, 1).contiguous()

    print("正在执行点云配准...")
    with torch.no_grad():
        rotation_ab, translation_ab, _, _ = net(src_tensor, tgt_tensor)

    rotation_ab_np = rotation_ab.detach().cpu().numpy().squeeze()
    translation_ab_np = translation_ab.detach().cpu().numpy().squeeze()

    print("\n计算出的变换矩阵 (R):")
    print(rotation_ab_np)
    print("\n计算出的平移向量 (t):")
    print(translation_ab_np)

    # 准备原始点云张量用于变换
    src_tensor_orig_shape = torch.from_numpy(src_points_np).unsqueeze(0).to(device)
    
    # <--- 最终修正点: 在调用前转置点云张量
    aligned_src_tensor = transform_point_cloud(src_tensor_orig_shape.transpose(2, 1), rotation_ab, translation_ab)
    
    # 将变换后的点云转置回来以便保存
    aligned_src_np = aligned_src_tensor.transpose(2, 1).detach().cpu().numpy().squeeze()
    
    save_numpy_to_pcd(aligned_src_np, args.output_path)


def main():
    # =================== 用户需要修改的参数 ===================
    src_path = "C:/Abandon/PCD_Data/data_2_cut.pcd"
    tgt_path = "C:/Abandon/PCD_Data/data_2_cut_transformed.pcd"
    model_path = "pretrained/dcp_v2.t7"
    output_path = "aligned_src_from_script.pcd"
    # =========================================================

    args = argparse.Namespace(
        src_path=src_path,
        tgt_path=tgt_path,
        model_path=model_path,
        output_path=output_path,
        
        emb_nn='dgcnn',
        pointer='transformer',
        head='svd',
        emb_dims=512,
        n_blocks=1,
        n_heads=4,
        ff_dims=1024,
        dropout=0.0,
        cycle=False,

        num_points=16384,
        no_cuda=False
    )
    
    register_pcds(args)


if __name__ == '__main__':
    main()
