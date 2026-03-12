import os
import argparse
from src.image_processor import ImageProcessor

def run_augmentation(dataset_root, output_folder, dims, n_vars):
    """
    Iterates through class subfolders and generates augmented image variations.
    """
    if not os.path.exists(dataset_root):
        print(f"Error: Dataset folder '{dataset_root}' not found!")
        return

    # 1. Identify Class Folders (Subdirectories)
    class_dirs = sorted([d for d in os.listdir(dataset_root) 
                        if os.path.isdir(os.path.join(dataset_root, d))])
    
    if not class_dirs:
        print(f"No class subfolders found in {dataset_root}.")
        print("Ensure your structure is: input_folder/class_name/images.jpg")
        return

    print(f"--- Found {len(class_dirs)} classes to process ---")

    # 2. Process Each Class Separately
    for class_name in class_dirs:
        print(f"\n[+] Processing Class: {class_name}")
        
        class_input = os.path.join(dataset_root, class_name)
        class_output = os.path.join(output_folder, class_name)
        
        # Initialize and run the ImageProcessor for this specific class
        processor = ImageProcessor(class_input, class_output, dims)
        processor.process(n=n_vars)

    print("\n" + "="*40)
    print("Augmentation complete for all classes.")
    print(f"Results saved to: {output_folder}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Image Augmentation Tool")
    
    # CLI Arguments
    parser.add_argument("--data", type=str, required=True, 
                        help="Path to the raw dataset (containing class subfolders)")
    parser.add_argument("--out", type=str, default="data/augmented_output", 
                        help="Directory where augmented images will be saved")
    parser.add_argument("--vars", type=int, default=16, 
                        help="Number of variations to generate per image (max ~72)")
    parser.add_argument("--dims", nargs='+', type=int, default=[32, 48, 64, 96], 
                        help="Space-separated list of target dimensions (e.g., 32 64)")

    args = parser.parse_args()
    
    run_augmentation(
        dataset_root=args.data, 
        output_folder=args.out, 
        dims=args.dims, 
        n_vars=args.vars
    )