from typing import List

import cv2
from pathlib import Path
from matplotlib import pyplot as plt
import os, random
import pandas as pd
from tqdm import tqdm
import shutil
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class SlabOutlineDetector:
    def __init__(self,
                 mode="test",
                 model_path: str = "output"  # .pkl path
                 ):
        self.mode = mode
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)

        if self.mode == "test":
            # Load model
            self.model = joblib.load(self.model_path / "model.pkl")["model"]
            self.scaler = joblib.load(self.model_path / "model.pkl")["scaler"]

        # Hyperparams
        self.max_area_thresh = 0.70
        self.kernel_sizes = [(1, 1), (3, 3)]
        self.area_check = True
        self.test_split = 0.15
        self.outline_dir = None
        self.vis_dir = None

    def train(self, X, y):
        # 3️⃣ 90-10 Split (Stratified → equal class proportion)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_split,
            random_state=42,
            stratify=y
        )

        print("\nTrain class distribution:\n", y_train.value_counts())
        print("\nTest class distribution:\n", y_test.value_counts())

        # 4️⃣ Scale AFTER split (correct practice)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 5️⃣ Train SVM
        model = SVC(
            kernel="rbf",
            C=3.0,
            gamma="scale",
            class_weight="balanced"
        )

        model.fit(X_train, y_train)

        # Evaluate on train
        print("\n===== Train PERFORMANCE =====")
        self.evaluate(model, X_train, y_train)

        print("\n===== Test PERFORMANCE =====")
        self.evaluate(model, X_test, y_test)

        # Save model
        joblib.dump({
            "model": model,
            "scaler": scaler,
            "feature_columns": X.columns.tolist()
        }, self.model_path / "model.pkl")

        print("Model saved successfully.")

        # Assign to the class variable
        self.model = model
        self.scaler = scaler

    @staticmethod
    def evaluate(model, X, y):
        y_pred = model.predict(X)
        print("Accuracy:", accuracy_score(y, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
        print(classification_report(y, y_pred))

    def predict_contour(self, image):
        # Get multi-kernel contours
        contour_1x1, contour_3x3 = self.find_slab_contour(image)

        # Get Contour features + Normalize
        features_1x1 = self.contour_feature_extractor(image, contour_1x1)
        features_3x3 = self.contour_feature_extractor(image, contour_3x3)

        consolidated_features = self.normalize_features(features_1x1, features_3x3)
        consolidated_features = np.array(list(consolidated_features.values())).reshape(1, -1)

        # 4️⃣ Predict
        consolidated_features = self.scaler.transform(consolidated_features)
        prediction = self.model.predict(consolidated_features)[0]

        return contour_1x1 if prediction == 0 else contour_3x3

    def run(self, files: List[str], save_dir: Path = None):
        self.create_folder(save_dir)

        for file in tqdm(files):
            # Read image
            image = cv2.imread(file)
            outline = self.predict_contour(image)

            # Save csv
            csv_name = Path(file).stem + ".csv"
            csv_path = self.outline_dir / csv_name
            np.savetxt(
                csv_path,
                outline.reshape(-1, 2),
                delimiter=",",
                header="x,y",
                comments=""
            )

            # Save visualization
            vis_image = image.copy()
            cv2.drawContours(vis_image, outline, -1, (0, 0, 255), 5)
            img_name = Path(file).stem + ".png"
            img_path = self.vis_dir / img_name
            cv2.imwrite(str(img_path), vis_image)

    @staticmethod
    def visualize(image, contour):
        pass

    def create_folder(self, directory: Path):
        if directory is not None:
            self.outline_dir = directory / "outline_csv"
            self.vis_dir = directory / "visualization"

            # Remove old folders if they exist
            for folder in [self.outline_dir, self.vis_dir]:
                if folder.exists():
                    shutil.rmtree(folder)
                folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def contour_feature_extractor(image, contour):
        if contour is None:
            return None

        H, W, _ = image.shape
        area = cv2.contourArea(contour)
        normalized_area = area / (H * W)

        perimeter = cv2.arcLength(contour, True)

        # Solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        # Shape complexity
        shape_complexity = (perimeter ** 2) / area if area > 0 else 0

        # Circularity
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        # point_density
        num_points = len(contour)
        point_density = num_points / perimeter if perimeter > 0 else 0

        # Shape
        epsilon = 0.01 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)

        # Aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / w

        return {
            "normalized_area": normalized_area,
            "solidity": solidity,
            "shape_complexity": shape_complexity,
            "circularity": circularity,
            "perimeter": perimeter,
            "point_density": point_density,
            "num_points": num_points,
            "area": area,
            "num_vertices": num_vertices,
            "aspect_ratio": aspect_ratio
        }

    @staticmethod
    def normalize_features(feat1, feat2):
        relative_feats = {}
        eps = 1e-6
        for key in feat1.keys():
            f1, f2 = feat1[key], feat2[key]
            r1, r2 = f1 / (f1 + f2 + eps), f2 / (f1 + f2 + eps)
            relative_feats[f"{key}_k1"] = r1
            relative_feats[f"{key}_k2"] = r2

        return relative_feats

    def find_slab_contour(self, image):
        # 1. Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 2. Binary threshold
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Thresholds for Area check
        h, w = image.shape[:2]
        image_area = h * w
        max_allowed_area = self.max_area_thresh * image_area

        # 3. Remove thin lines and structures.
        # - Erosion followed by Dilation -- Multi-kernel
        multi_kernel_largest_contour = []
        for kern in self.kernel_sizes:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kern)
            binary_ = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            # 4. Find contours at this kernel size
            contours, hierarchy = cv2.findContours(
                binary_,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_NONE
            )

            # 5. Filter contours:
            #    - remove zero-area
            #    - remove very large (>= 80% image)
            if self.area_check:
                contours = [
                    c for c in contours
                    if 0 < cv2.contourArea(c) < max_allowed_area
                ]

            # 6. Sort by area and take top 3
            largest_contour = sorted(
                contours,
                key=cv2.contourArea,
                reverse=True
            )[0]

            multi_kernel_largest_contour.append(largest_contour)

        return multi_kernel_largest_contour

    def postprocess(self):
        pass


if __name__ == "__main__":
    mode = "test"
    if mode == "train":  # Training + Testing
        # Initialize
        save_dir = "runs/SO_DET_V1"
        so_det = SlabOutlineDetector(mode="train", model_path=save_dir)

        # Load Data
        csv_path = "data_/dataset_with_features.csv"
        df = pd.read_csv(csv_path)

        # Drop non-feature columns
        drop_cols = ["filename", "image_path"]
        drop_cols = [c for c in drop_cols if c in df.columns]

        X = df.drop(columns=drop_cols + ["label"])
        y = df["label"]

        print("Total samples:", len(df))
        print("Feature dimension:", X.shape[1])
        print("Class distribution:\n", y.value_counts())

        # Train
        so_det.train(X, y)

        # Test
        folder2run = "data_/402627 - BHC Chermside McNabs/images"
        files = [os.path.join(folder2run, file) for file in os.listdir(folder2run)]
        save_dir = Path(save_dir) / folder2run.split("/")[-2]
        so_det.run(files, save_dir)

    else:  # Testing on projects
        # Initialize
        save_dir = "runs/SO_DET_V1"
        so_det = SlabOutlineDetector(mode="test", model_path=save_dir)

        # Test
        projects2run = ["data_/402627 - BHC Chermside McNabs/images",
                        "data_/402867 - Polycell - The Rochester/images",
                        "data_/402868 - Melrose Built - Ducale Luxury Residence, Teneriffe/images",
                        "data_/402886 - M8 Con Trading - Kora, 798 Pacific Pd, Currumbin/images",
                        "data_/Res-7 level-PoC/images",
                        "data_/Resi-6 lvls (402460 - One Earle Lane)/images",
                        "data_/Resi-6 lvls (402683 - Hutchinson Radcliffe Kingsford Tce)/images",
                        "data_/Resi-8 lvls (402489 - Lana Apartments)/images",
                        "data_/Residential - 5 level/images"
                        ]

        for project in projects2run:
            files = [os.path.join(project, file) for file in os.listdir(project) if ".png" in file]
            save_dir_ = Path(save_dir) / project.split("/")[-2]
            so_det.run(files, save_dir_)

