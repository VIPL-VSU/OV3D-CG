import os
import numpy as np
import torch
from tqdm import tqdm


def convert_masks_to_pt(mask_source_dir, split_file_path, output_dir):
    """
    Reads ScanNet mask text files, merges them, and saves them in PyTorch .pt format.

    Args:
        mask_source_dir (str): Directory containing mask files and index txt files.
        split_file_path (str): Path to the txt file containing the list of scene_ids (e.g., scannetv2_val.txt).
        output_dir (str): Directory to save the .pt files.
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read scene_ids
    if not os.path.exists(split_file_path):
        raise FileNotFoundError(f"Split file not found: {split_file_path}")

    with open(split_file_path, 'r') as f:
        scene_ids = [line.strip() for line in f.readlines()]

    print(f"Start processing {len(scene_ids)} scenes...")

    for scene_id in tqdm(scene_ids):
        # Construct index file path
        index_file_path = os.path.join(mask_source_dir, f"{scene_id}.txt")

        if not os.path.exists(index_file_path):
            print(f"Warning: Index file for {scene_id} not found, skipping.")
            continue

        try:
            # Load file name list
            # ndmin=2 ensures the result is a 2D array even if there is only one line, avoiding [:, 0] errors
            file_data = np.loadtxt(index_file_path, dtype='str', delimiter=' ', ndmin=2)
            file_names = file_data[:, 0]

            masks = []
            for file_name in file_names:
                mask_file_path = os.path.join(mask_source_dir, file_name)
                # Load individual mask (Note: np.loadtxt is slow for text; use binary if possible)
                mask = np.loadtxt(mask_file_path)
                masks.append(mask)

            if not masks:
                print(f"Warning: No masks found for {scene_id}")
                continue

            # Stack and convert to Tensor
            masks_np = np.vstack(masks)
            masks_np = masks_np.T
            masks_tensor = torch.from_numpy(masks_np)

            # Optional: Cast to specific type if needed (e.g., int32 for labels)
            # masks_tensor = masks_tensor.to(torch.int32)

            # Save as .pt format
            save_path = os.path.join(output_dir, f"{scene_id}_masks.pt")
            torch.save(masks_tensor, save_path)

        except Exception as e:
            print(f"Error processing {scene_id}: {e}")

    print("Processing complete.")


# --- Usage Example ---

# Define paths
mask_path_dir = "/PATH/SAI3D/data/ScanNet/results/demo_scannet_5view_merge200_2-norm_semantic-sam_depth2/"
split_txt_path = "evaluation/val_scenes_scannet200.txt"
save_pt_dir = "sai3d_masks/"

# Execute function
if __name__ == "__main__":
    convert_masks_to_pt(
        mask_source_dir=mask_path_dir,
        split_file_path=split_txt_path,
        output_dir=save_pt_dir
    )