"""Script to calculate the average number of objects per scene (Obj metric) in predicted layouts.
"""
import argparse
import numpy as np
import pickle

from threed_front.datasets import get_raw_dataset
from threed_front.evaluation import ThreedFrontResults

def main(argv):
    parser = argparse.ArgumentParser(
        description=("Compute the average number of objects per scene (Obj metric) "
                     "for both predicted and ground-truth layouts")
    )
    parser.add_argument(
        "result_file",
        help="Path to a pickled result file (ThreedFrontResults object)"
    )
    args = parser.parse_args(argv)

    # Load saved results
    with open(args.result_file, "rb") as f:
        threed_front_results = pickle.load(f)
    assert isinstance(threed_front_results, ThreedFrontResults)
    assert threed_front_results.floor_condition

    # Load dataset
    config = threed_front_results.config
    raw_dataset = get_raw_dataset(
        config["data"], 
        split=config["validation"].get("splits", ["test"]),
        include_room_mask=True
    ) 

    num_objects_pred = []
    num_objects_gt = []

    for scene_idx, scene_layout in threed_front_results:
        gt_scene_layout = raw_dataset.get_room_params(scene_idx)
        # Count number of objects in predicted and ground-truth layouts
        num_objects_pred.append(len(scene_layout["class_labels"]))
        num_objects_gt.append(len(gt_scene_layout["class_labels"]))

    avg_num_objects_pred = np.mean(num_objects_pred)
    avg_num_objects_gt = np.mean(num_objects_gt)

    print("Obj (average number of objects per scene):")
    print("  Predicted layouts: {:.2f}".format(avg_num_objects_pred))
    print("  Ground-truth layouts: {:.2f}".format(avg_num_objects_gt))


if __name__ == "__main__":
    main(None)
