import os
import json
import trimesh
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import argparse

NUM_SAMPLES = 16384
NUM_TRIALS = 1

def load_obj_as_pointcloud(file_path, num_points=16384):
    mesh = trimesh.load(file_path)
    if isinstance(mesh, trimesh.Scene):  # sometimes .obj loads as scene
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points

def apply_transformation(pc, transform_matrix):
    homogenous_pc = np.hstack((pc, np.ones((pc.shape[0], 1))))
    transformed_pc = homogenous_pc.dot(transform_matrix.T)
    return transformed_pc[:, :3]

def chamfer_distance_l1(pc1, pc2):
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)

    d1, _ = tree1.query(pc2)
    d2, _ = tree2.query(pc1)

    return np.mean(np.abs(d1)) + np.mean(np.abs(d2))

def compute_average_chamfer_distance(gt_folder, pred_folder, transform_file):
    with open(transform_file, 'r') as f:
        transforms = json.load(f)

    filenames = sorted([f for f in os.listdir(gt_folder) if f.endswith(".obj")])
    all_scores = []

    #for fname in filenames:
    for i in range(1, 35):
        fname = f"{i}.obj"
        gt_path = os.path.join(gt_folder, fname)
        pred_path = os.path.join(pred_folder, fname)

        if not os.path.exists(pred_path):
            print(f"Missing prediction for {fname}, skipping.")
            continue

        trial_scores = []
        for _ in range(NUM_TRIALS):
            gt_points = load_obj_as_pointcloud(gt_path, NUM_SAMPLES)
            pred_points = load_obj_as_pointcloud(pred_path, NUM_SAMPLES)
            w_t_pred_points = pred_points

            if fname in transforms:
                transform_matrix = np.array(transforms[fname])
                pred_points = apply_transformation(pred_points, transform_matrix)

            dist = chamfer_distance_l1(gt_points, pred_points)
            trial_scores.append(dist)

        avg_score = np.mean(trial_scores)
        all_scores.append(avg_score)
        print(f"{fname}: {avg_score:.6f}")

        # Visualize and save the point clouds
        visualize_and_save_point_clouds(gt_points, w_t_pred_points, pred_points, fname)

    return np.mean(all_scores)

def visualize_and_save_point_clouds(pc1, unchange_pc2, pc2, obj_name):
    fig = plt.figure(figsize=(12, 6))
                
    ax1 = fig.add_subplot(121, projection='3d')
    visualize_point_clouds([pc1, unchange_pc2], ['blue', 'red'], ax1, 'Original Point Clouds')
    
    ax2 = fig.add_subplot(122, projection='3d')
    visualize_point_clouds([pc1, pc2], ['blue', 'red'], ax2, 'Transformed Point Clouds')
    
    plt.tight_layout()
    plt.savefig('Figs/' + obj_name +'.png', bbox_inches = 'tight')
    #plt.show()

def visualize_point_clouds(point_clouds, colors, ax, title):
    for pc, color in zip(point_clouds, colors):
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=color, s=1)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def main():
    parser = argparse.ArgumentParser(description="Compute average L1 Chamfer Distance with transformation.")
    parser.add_argument('ground_truth_folder', type=str, help="Path to the ground truth folder")
    parser.add_argument('prediction_folder', type=str, help="Path to the prediction folder")
    parser.add_argument('transform_file', type=str, help="Path to the JSON file with transformation matrices")

    args = parser.parse_args()

    if not os.path.isdir(args.ground_truth_folder) or not os.path.isdir(args.prediction_folder):
        print("One of the provided directories is invalid.")
        return

    if not os.path.isfile(args.transform_file):
        print(f"Invalid transform file: {args.transform_file}")
        return

    final_score = compute_average_chamfer_distance(args.ground_truth_folder, args.prediction_folder, args.transform_file)
    print(f"\nFinal Average Chamfer Distance: {final_score:.6f}")

if __name__ == "__main__":
    main()

