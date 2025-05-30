import os
import trimesh
import numpy as np
from scipy.spatial import cKDTree
import argparse
import random

NUM_SAMPLES = 16384
NUM_TRIALS = 10

def load_obj_as_pointcloud(file_path, num_points=16384):
    mesh = trimesh.load(file_path)
    if isinstance(mesh, trimesh.Scene):  # sometimes .obj loads as scene
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points

def chamfer_distance_l1(pc1, pc2):
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)

    d1, _ = tree1.query(pc2)
    d2, _ = tree2.query(pc1)

    return np.mean(np.abs(d1)) + np.mean(np.abs(d2))

def compute_average_chamfer_distance(gt_folder, pred_folder):
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
            dist = chamfer_distance_l1(gt_points, pred_points)
            trial_scores.append(dist)

        avg_trial_score = np.mean(trial_scores)
        all_scores.append(avg_trial_score)
        print(f"{fname}: {avg_trial_score:.6f}")

    overall_score = np.mean(all_scores)
    return overall_score

def main():
    parser = argparse.ArgumentParser(description="Compute average L1 Chamfer Distance over .obj files.")
    parser.add_argument('ground_truth_folder', type=str, help="Path to ground truth .obj files")
    parser.add_argument('prediction_folder', type=str, help="Path to predicted .obj files")

    args = parser.parse_args()

    if not os.path.isdir(args.ground_truth_folder) or not os.path.isdir(args.prediction_folder):
        print("One of the provided directories does not exist.")
        return

    final_score = compute_average_chamfer_distance(args.ground_truth_folder, args.prediction_folder)
    print(f"\nFinal Average Chamfer Distance: {final_score:.6f}")

if __name__ == "__main__":
    main()

