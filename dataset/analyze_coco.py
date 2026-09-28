import os
import json
import statistics
from collections import Counter

import matplotlib.pyplot as plt


class COCODatasetAnalyzer:

    def __init__(self):

        # ==========================================
        # DATASET LOCATIONS
        # ==========================================

        self.images_folder = "dataset/images/val2017"

        self.captions_file = (
            "dataset/annotations/annotations/"
            "captions_val2017.json"
        )

        self.instances_file = (
            "dataset/annotations/annotations/"
            "instances_val2017.json"
        )

        # ==========================================
        # OUTPUT FOLDER
        # ==========================================

        self.output_folder = "dataset/analysis_results"

        os.makedirs(
            self.output_folder,
            exist_ok=True
        )

        print("========================================")
        print("       COCO DATASET ANALYZER")
        print("========================================")

        # ==========================================
        # CHECK PATHS
        # ==========================================

        if not os.path.exists(self.images_folder):

            raise FileNotFoundError(
                f"Image folder not found:\n"
                f"{self.images_folder}"
            )

        if not os.path.exists(self.captions_file):

            raise FileNotFoundError(
                f"Caption file not found:\n"
                f"{self.captions_file}"
            )

        if not os.path.exists(self.instances_file):

            raise FileNotFoundError(
                f"Instance annotation file not found:\n"
                f"{self.instances_file}"
            )

        # ==========================================
        # LOAD CAPTION DATA
        # ==========================================

        print("\nLoading caption annotations...")

        with open(
            self.captions_file,
            "r",
            encoding="utf-8"
        ) as file:

            self.caption_data = json.load(file)

        self.images = self.caption_data.get(
            "images",
            []
        )

        self.annotations = self.caption_data.get(
            "annotations",
            []
        )

        # ==========================================
        # LOAD CATEGORY DATA
        # ==========================================

        print("Loading category annotations...")

        with open(
            self.instances_file,
            "r",
            encoding="utf-8"
        ) as file:

            self.instance_data = json.load(file)

        self.categories = self.instance_data.get(
            "categories",
            []
        )

        self.instance_annotations = (
            self.instance_data.get(
                "annotations",
                []
            )
        )

        # ==========================================
        # IMAGE LOOKUP
        # ==========================================

        self.image_map = {
            image["id"]: image
            for image in self.images
        }

        # ==========================================
        # CATEGORY LOOKUP
        # ==========================================

        self.category_map = {
            category["id"]: category["name"]
            for category in self.categories
        }

        print("Dataset loaded successfully.")

    # ==========================================
    # BASIC STATISTICS
    # ==========================================

    def basic_statistics(self):

        print("\n")
        print("========================================")
        print("       BASIC DATASET STATISTICS")
        print("========================================")

        print(
            f"Number of images: "
            f"{len(self.images)}"
        )

        print(
            f"Number of captions: "
            f"{len(self.annotations)}"
        )

        print(
            f"Number of categories/classes: "
            f"{len(self.categories)}"
        )

    # ==========================================
    # IMAGE RESOLUTION
    # ==========================================

    def image_resolution_statistics(self):

        widths = []
        heights = []

        for image in self.images:

            width = image.get("width")
            height = image.get("height")

            if width and height:

                widths.append(width)
                heights.append(height)

        if not widths:

            print("\nNo resolution data found.")

            return

        print("\n")
        print("========================================")
        print("       IMAGE RESOLUTION ANALYSIS")
        print("========================================")

        print(
            f"Minimum width: "
            f"{min(widths)} px"
        )

        print(
            f"Maximum width: "
            f"{max(widths)} px"
        )

        print(
            f"Average width: "
            f"{statistics.mean(widths):.2f} px"
        )

        print(
            f"Minimum height: "
            f"{min(heights)} px"
        )

        print(
            f"Maximum height: "
            f"{max(heights)} px"
        )

        print(
            f"Average height: "
            f"{statistics.mean(heights):.2f} px"
        )

        aspect_ratios = [
            width / height
            for width, height
            in zip(widths, heights)
            if height != 0
        ]

        print(
            f"Average aspect ratio: "
            f"{statistics.mean(aspect_ratios):.2f}"
        )

        # Save resolution chart

        plt.figure(figsize=(10, 6))

        plt.hist(
            widths,
            bins=30
        )

        plt.title(
            "COCO Validation Set - Image Width Distribution"
        )

        plt.xlabel("Image Width (pixels)")
        plt.ylabel("Number of Images")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_folder,
                "image_width_distribution.png"
            ),
            dpi=150
        )

        plt.close()

        # Height chart

        plt.figure(figsize=(10, 6))

        plt.hist(
            heights,
            bins=30
        )

        plt.title(
            "COCO Validation Set - Image Height Distribution"
        )

        plt.xlabel("Image Height (pixels)")
        plt.ylabel("Number of Images")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_folder,
                "image_height_distribution.png"
            ),
            dpi=150
        )

        plt.close()

    # ==========================================
    # CAPTION ANALYSIS
    # ==========================================

    def caption_statistics(self):

        caption_lengths = []

        for annotation in self.annotations:

            caption = annotation.get(
                "caption",
                ""
            ).strip()

            if caption:

                words = caption.split()

                caption_lengths.append(
                    len(words)
                )

        if not caption_lengths:

            print("\nNo captions found.")

            return

        print("\n")
        print("========================================")
        print("       CAPTION LENGTH ANALYSIS")
        print("========================================")

        print(
            f"Minimum caption length: "
            f"{min(caption_lengths)} words"
        )

        print(
            f"Maximum caption length: "
            f"{max(caption_lengths)} words"
        )

        print(
            f"Average caption length: "
            f"{statistics.mean(caption_lengths):.2f} words"
        )

        print(
            f"Median caption length: "
            f"{statistics.median(caption_lengths):.2f} words"
        )

        # ==========================================
        # CAPTION LENGTH HISTOGRAM
        # ==========================================

        plt.figure(figsize=(10, 6))

        plt.hist(
            caption_lengths,
            bins=30
        )

        plt.title(
            "COCO Validation Set - Caption Length Distribution"
        )

        plt.xlabel("Caption Length (words)")
        plt.ylabel("Number of Captions")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_folder,
                "caption_length_distribution.png"
            ),
            dpi=150
        )

        plt.close()

    # ==========================================
    # CAPTIONS PER IMAGE
    # ==========================================

    def captions_per_image(self):

        counts = Counter()

        for annotation in self.annotations:

            image_id = annotation.get(
                "image_id"
            )

            if image_id is not None:

                counts[image_id] += 1

        if not counts:

            return

        print("\n")
        print("========================================")
        print("       CAPTIONS PER IMAGE")
        print("========================================")

        print(
            f"Images with captions: "
            f"{len(counts)}"
        )

        print(
            f"Minimum captions/image: "
            f"{min(counts.values())}"
        )

        print(
            f"Maximum captions/image: "
            f"{max(counts.values())}"
        )

        print(
            f"Average captions/image: "
            f"{statistics.mean(counts.values()):.2f}"
        )

    # ==========================================
    # CATEGORY ANALYSIS
    # ==========================================

    def category_statistics(self):

        category_counts = Counter()

        for annotation in self.instance_annotations:

            category_id = annotation.get(
                "category_id"
            )

            if category_id in self.category_map:

                category_name = (
                    self.category_map[category_id]
                )

                category_counts[
                    category_name
                ] += 1

        print("\n")
        print("========================================")
        print("       CATEGORY ANALYSIS")
        print("========================================")

        print(
            f"Number of categories/classes: "
            f"{len(self.categories)}"
        )

        print("\nCategory names:")

        for category in self.categories:

            print(
                f"- {category['name']}"
            )

        # ==========================================
        # TOP CATEGORIES
        # ==========================================

        top_categories = (
            category_counts.most_common(20)
        )

        if not top_categories:

            return

        names = [
            item[0]
            for item in top_categories
        ]

        counts = [
            item[1]
            for item in top_categories
        ]

        plt.figure(figsize=(12, 7))

        plt.bar(
            names,
            counts
        )

        plt.title(
            "Top 20 COCO Categories by Annotation Count"
        )

        plt.xlabel("Category")
        plt.ylabel("Number of Object Annotations")

        plt.xticks(
            rotation=75,
            ha="right"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_folder,
                "category_distribution.png"
            ),
            dpi=150
        )

        plt.close()

    # ==========================================
    # IMAGE FILE VERIFICATION
    # ==========================================

    def verify_images(self):

        existing = 0
        missing = 0

        for image in self.images:

            filename = image.get(
                "file_name"
            )

            if not filename:

                continue

            image_path = os.path.join(
                self.images_folder,
                filename
            )

            if os.path.exists(image_path):

                existing += 1

            else:

                missing += 1

        print("\n")
        print("========================================")
        print("       IMAGE FILE VERIFICATION")
        print("========================================")

        print(
            f"Images found: "
            f"{existing}"
        )

        print(
            f"Images missing: "
            f"{missing}"
        )

    # ==========================================
    # SAMPLE IMAGE + CAPTION GRID
    # ==========================================

    def create_sample_grid(self, number=6):

        print("\n")
        print("========================================")
        print("       CREATING SAMPLE IMAGE GRID")
        print("========================================")

        samples = []

        for annotation in self.annotations:

            image_id = annotation.get(
                "image_id"
            )

            caption = annotation.get(
                "caption",
                ""
            ).strip()

            image_info = self.image_map.get(
                image_id
            )

            if not image_info:

                continue

            filename = image_info.get(
                "file_name"
            )

            image_path = os.path.join(
                self.images_folder,
                filename
            )

            if not os.path.exists(image_path):

                continue

            samples.append(
                (
                    image_path,
                    caption
                )
            )

            if len(samples) >= number:

                break

        if not samples:

            print("No sample images found.")

            return

        # ==========================================
        # CREATE GRID
        # ==========================================

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(15, 10)
        )

        axes = axes.flatten()

        for index, (image_path, caption) in enumerate(
            samples
        ):

            image = plt.imread(
                image_path
            )

            axes[index].imshow(image)

            axes[index].set_title(
                caption,
                fontsize=9
            )

            axes[index].axis("off")

        # Hide unused axes

        for index in range(
            len(samples),
            len(axes)
        ):

            axes[index].axis("off")

        fig.suptitle(
            "COCO Validation Set - Sample Images with Captions",
            fontsize=16
        )

        plt.tight_layout()

        output_path = os.path.join(
            self.output_folder,
            "sample_images_with_captions.png"
        )

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Sample grid saved to:\n"
            f"{output_path}"
        )

    # ==========================================
    # SAVE TEXT SUMMARY
    # ==========================================

    def save_summary(self):

        summary_path = os.path.join(
            self.output_folder,
            "dataset_summary.txt"
        )

        with open(
            summary_path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                "COCO 2017 VALIDATION DATASET ANALYSIS\n"
            )

            file.write(
                "========================================\n\n"
            )

            file.write(
                f"Images: {len(self.images)}\n"
            )

            file.write(
                f"Captions: {len(self.annotations)}\n"
            )

            file.write(
                f"Categories: {len(self.categories)}\n"
            )

            file.write(
                "\nThis analysis was generated "
                "from the downloaded COCO 2017 "
                "validation subset.\n"
            )

        print(
            f"\nSummary saved to:\n"
            f"{summary_path}"
        )

    # ==========================================
    # RUN COMPLETE ANALYSIS
    # ==========================================

    def run_analysis(self):

        self.basic_statistics()

        self.image_resolution_statistics()

        self.caption_statistics()

        self.captions_per_image()

        self.category_statistics()

        self.verify_images()

        self.create_sample_grid()

        self.save_summary()

        print("\n")
        print("========================================")
        print("       TASK 4 ANALYSIS COMPLETE")
        print("========================================")

        print(
            f"\nResults saved in:"
            f"\n{self.output_folder}"
        )


# ==============================================
# PROGRAM ENTRY POINT
# ==============================================

if __name__ == "__main__":

    analyzer = COCODatasetAnalyzer()

    analyzer.run_analysis()