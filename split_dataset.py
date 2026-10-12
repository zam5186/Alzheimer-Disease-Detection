import random
import shutil
from pathlib import Path

# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(
    r"C:\Users\zmujawar\Alzheimer-Disease-Detection"
)

SOURCE_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "dataset_split"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 123

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ==================================================
# 2. SPLIT DATASET
# ==================================================

def split_dataset():

    print("=" * 55)
    print("ALZHEIMER MRI DATASET SPLITTER")
    print("=" * 55)

    # Check original dataset
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {SOURCE_DIR}"
        )

    # Validate ratios
    if abs(
        TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO - 1.0
    ) > 1e-9:
        raise ValueError(
            "Training, validation and test ratios "
            "must add up to 1.0."
        )

    # Avoid accidentally overwriting an existing split
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"\nOutput folder already exists:\n{OUTPUT_DIR}\n\n"
            "If you want to create a new split, rename or "
            "remove the existing dataset_split folder first."
        )

    random.seed(RANDOM_SEED)

    # Find class folders
    class_folders = sorted(
        folder
        for folder in SOURCE_DIR.iterdir()
        if folder.is_dir()
    )

    if len(class_folders) != 4:
        raise ValueError(
            f"Expected 4 class folders, found "
            f"{len(class_folders)}: "
            f"{[folder.name for folder in class_folders]}"
        )

    total_images = 0
    total_training = 0
    total_validation = 0
    total_testing = 0

    # ==================================================
    # 3. PROCESS EACH CLASS
    # ==================================================

    for class_folder in class_folders:

        class_name = class_folder.name

        images = sorted(
            image
            for image in class_folder.rglob("*")
            if image.is_file()
            and image.suffix.lower() in IMAGE_EXTENSIONS
        )

        if len(images) < 3:
            raise ValueError(
                f"Not enough images in class {class_name}. "
                "At least 3 images are required."
            )

        # Shuffle images reproducibly
        random.shuffle(images)

        image_count = len(images)

        # Calculate split boundaries
        train_end = int(image_count * TRAIN_RATIO)

        validation_end = int(
            image_count * (TRAIN_RATIO + VALIDATION_RATIO)
        )

        training_images = images[:train_end]

        validation_images = images[
            train_end:validation_end
        ]

        testing_images = images[validation_end:]

        split_data = {
            "train": training_images,
            "validation": validation_images,
            "test": testing_images
        }

        # ==================================================
        # 4. COPY IMAGES INTO SPLIT FOLDERS
        # ==================================================

        for split_name, split_images in split_data.items():

            for image_path in split_images:

                relative_path = image_path.relative_to(
                    class_folder
                )

                destination = (
                    OUTPUT_DIR
                    / split_name
                    / class_name
                    / relative_path
                )

                destination.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                shutil.copy2(
                    image_path,
                    destination
                )

        # Print class-wise statistics
        train_count = len(training_images)
        validation_count = len(validation_images)
        test_count = len(testing_images)

        total_images += image_count
        total_training += train_count
        total_validation += validation_count
        total_testing += test_count

        print(f"\nClass: {class_name}")
        print(f"  Total images      : {image_count}")
        print(f"  Training images   : {train_count}")
        print(f"  Validation images : {validation_count}")
        print(f"  Testing images    : {test_count}")

    # ==================================================
    # 5. DISPLAY FINAL SUMMARY
    # ==================================================

    print("\n" + "=" * 55)
    print("DATASET SPLIT COMPLETED SUCCESSFULLY")
    print("=" * 55)

    print(f"Total images : {total_images}")

    print(
        f"Training images : {total_training} "
        f"({total_training / total_images * 100:.2f}%)"
    )

    print(
        f"Validation images : {total_validation} "
        f"({total_validation / total_images * 100:.2f}%)"
    )

    print(
        f"Testing images : {total_testing} "
        f"({total_testing / total_images * 100:.2f}%)"
    )

    print(f"\nOriginal dataset : {SOURCE_DIR}")
    print(f"Split dataset    : {OUTPUT_DIR}")

    print("\nNext step:")
    print("Run notebooks/train_model.py to train the CNN.")


# ==================================================
# 6. EXECUTION
# ==================================================

if __name__ == "__main__":
    try:
        split_dataset()

    except (
        FileNotFoundError,
        FileExistsError,
        ValueError
    ) as error:
        print(f"\nERROR: {error}")

    except Exception as error:
        print(f"\nUnexpected error: {error}")
        raise