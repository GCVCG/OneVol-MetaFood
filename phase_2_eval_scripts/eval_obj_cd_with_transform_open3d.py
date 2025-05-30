import os
import json
import trimesh
import numpy as np
from scipy.spatial import cKDTree
import argparse
import open3d as o3d

NUM_SAMPLES = 16384
NUM_TRIALS = 10

def load_obj_as_pointcloud(file_path, num_points=16384):
    mesh = trimesh.load(file_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points

def apply_transformation(pc, transform_matrix):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.transform(transform_matrix)
    return np.asarray(pcd.points)

def chamfer_distance_l1(pc1, pc2):
    kdtree1 = cKDTree(pc1)
    kdtree2 = cKDTree(pc2)

    d1, _ = kdtree1.query(pc2)
    d2, _ = kdtree2.query(pc1)

    return np.mean(np.abs(d1)) + np.mean(np.abs(d2))

def compute_average_chamfer_distance(gt_folder, pred_folder, transform_file):
    with open(transform_file, 'r') as f:
        transforms = json.load(f)

    filenames = sorted([f for f in os.listdir(gt_folder) if f.endswith(".obj")])
    all_scores = []

    for fname in filenames:
        gt_path = os.path.join(gt_folder, fname)
        pred_path = os.path.join(pred_folder, fname)

        if not os.path.exists(pred_path):
            print(f"Missing prediction for {fname}, skipping.")
            continue

        trial_scores = []
        for _ in range(NUM_TRIALS):
            gt_points = load_obj_as_pointcloud(gt_path, NUM_SAMPLES)
            pred_points = load_obj_as_pointcloud(pred_path, NUM_SAMPLES)

            if fname in transforms:
                transform_matrix = np.array(transforms[fname])
                pred_points = apply_transformation(pred_points, transform_matrix)

            dist = chamfer_distance_l1(gt_points, pred_points)
            trial_scores.append(dist)

        avg_score = np.mean(trial_scores)
        all_scores.append(avg_score)
        print(f"{fname}: {avg_score:.6f}")

    return np.mean(all_scores)

def main():
    parser = argparse.ArgumentParser(description="Compute average L1 Chamfer Distance between .obj files with transformation using Open3D.")
    parser.add_argument('ground_truth_folder', type=str, help="Path to ground truth folder")
    parser.add_argument('prediction_folder', type=str, help="Path to prediction folder")
    parser.add_argument('transform_file', type=str, help="Path to JSON file with transformation matrices")

    args = parser.parse_args()

    if not os.path.isdir(args.ground_truth_folder):
        print(f"Invalid directory: {args.ground_truth_folder}")
        return

    if not os.path.isdir(args.prediction_folder):
        print(f"Invalid directory: {args.prediction_folder}")
        return

    if not os.path.isfile(args.transform_file):
        print(f"Invalid file: {args.transform_file}")
        return

    final_score = compute_average_chamfer_distance(args.ground_truth_folder, args.prediction_folder, args.transform_file)
    print(f"\nFinal Average Chamfer Distance: {final_score:.6f}")

if __name__ == "__main__":
    main()

