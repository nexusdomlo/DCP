import torch
import numpy as np
import open3d as o3d
from model import DCP
from util import transform_point_cloud
import argparse

def load_pcd_as_tensor(filepath, num_points):
    """加载PCD文件并转换为PyTorch张量"""
    # 1. 使用 open3d 加载点云
    pcd = o3d.io.read_point_cloud(filepath)
    
    # 2. 将点云转换为 numpy 数组
    points = np.asarray(pcd.points)
    
    # 3. 采样或填充到指定点数
    if len(points) > num_points:
        # 如果点数过多，随机采样
        indices = np.random.choice(len(points), num_points, replace=False)
        points = points[indices]
    elif len(points) < num_points:
        # 如果点数不足，重复采样
        indices = np.random.choice(len(points), num_points, replace=True)
        points = points[indices]
        
    # 4. 转换为 PyTorch 张量并增加 batch 维度
    # DCP模型需要 (Batch, 3, Num_points) 的格式
    tensor = torch.from_numpy(points).float().unsqueeze(0).transpose(1, 2)
    return tensor

def main():
    print("DCP Custom PCD Registration Demo")
    parser = argparse.ArgumentParser(description='Test DCP on custom PCD files')
    parser.add_argument('--src_path', type=str, required=True, help='Path to the source PCD file')
    parser.add_argument('--tgt_path', type=str, required=True, help='Path to the target PCD file')
    parser.add_argument('--model_path', type=str, default='pretrained/dcp_v2.t7', help='Path to the pretrained model')
    parser.add_argument('--num_points', type=int, default=1024, help='Number of points to use')
    
    # --- 这些参数需要和预训练模型匹配 ---
    parser.add_argument('--emb_nn', type=str, default='dgcnn', choices=['pointnet', 'dgcnn'])
    parser.add_argument('--pointer', type=str, default='transformer', choices=['identity', 'transformer'])
    parser.add_argument('--head', type=str, default='svd', choices=['mlp', 'svd'])
    # ------------------------------------

    args = parser.parse_args()
    # 在 test_custom.py 的 main() 函数中添加
    args.emb_dims = 512
    args.emb_nn = 'dgcnn'
    args.pointer = 'transformer'
    args.head = 'svd'
    args.cycle = False
    # 1. 加载模型
    net = DCP(args).cuda()
    net.load_state_dict(torch.load(args.model_path), strict=False)
    net.eval()
    print(f"Model loaded from {args.model_path}")

    # 2. 加载你的PCD文件
    src_tensor = load_pcd_as_tensor(args.src_path, args.num_points).cuda()
    tgt_tensor = load_pcd_as_tensor(args.tgt_path, args.num_points).cuda()
    print(f"Source PCD loaded from {args.src_path}")
    print(f"Target PCD loaded from {args.tgt_path}")

    # 3. 使用DCP模型进行预测
    with torch.no_grad():
        # 模型会预测两个方向的变换，我们通常只需要 src -> tgt
        rotation_ab, translation_ab, _, _ = net(src_tensor, tgt_tensor)

    # 4. 应用预测的变换
    src_transformed_tensor = transform_point_cloud(src_tensor, rotation_ab, translation_ab)

    # 5. 结果可视化 (可选)
    src_points_transformed = src_transformed_tensor.squeeze(0).transpose(0, 1).cpu().numpy()
    tgt_points = tgt_tensor.squeeze(0).transpose(0, 1).cpu().numpy()
    src_points_original = src_tensor.squeeze(0).transpose(0, 1).cpu().numpy()

    pcd_src_orig = o3d.geometry.PointCloud()
    pcd_src_orig.points = o3d.utility.Vector3dVector(src_points_original)
    pcd_src_orig.paint_uniform_color([1, 0, 0]) # 原始源点云 (红色)

    pcd_tgt = o3d.geometry.PointCloud()
    pcd_tgt.points = o3d.utility.Vector3dVector(tgt_points)
    pcd_tgt.paint_uniform_color([0, 1, 0]) # 目标点云 (绿色)

    pcd_src_transformed = o3d.geometry.PointCloud()
    pcd_src_transformed.points = o3d.utility.Vector3dVector(src_points_transformed)
    pcd_src_transformed.paint_uniform_color([0, 0, 1]) # 配准后的源点云 (蓝色)

    print("Showing visualization: Red=Original Source, Green=Target, Blue=Aligned Source")
    o3d.visualization.draw_geometries([pcd_src_orig, pcd_tgt, pcd_src_transformed])

if __name__ == '__main__':
    main()