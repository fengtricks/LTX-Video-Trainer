"""Dataset Manager for LTX-Video-Trainer.

This module provides utilities for managing video datasets, including:
- Creating and organizing datasets
- Adding videos with automatic thumbnail generation
- Managing captions
- Dataset validation
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import cv2


class DatasetManager:
    """Manager for video dataset operations."""

    def __init__(self, datasets_root: Path = Path(__file__).parent.parent / "datasets"):
        """Initialize the dataset manager.

        Args:
            datasets_root: Root directory for all datasets
        """
        self.datasets_root = datasets_root
        self.datasets_root.mkdir(exist_ok=True)

    def list_datasets(self) -> List[str]:
        """Get all dataset names.

        Returns:
            List of dataset names
        """
        return [d.name for d in self.datasets_root.iterdir() if d.is_dir()]

    def create_dataset(self, name: str) -> Path:
        """Create new dataset directory structure.

        Args:
            name: Name of the dataset

        Returns:
            Path to the created dataset directory

        Raises:
            ValueError: If dataset name is invalid or already exists
        """
        if not name or not name.strip():
            raise ValueError("Dataset name cannot be empty")

        # Sanitize name
        safe_name = "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip()
        if not safe_name:
            raise ValueError("Dataset name must contain alphanumeric characters")

        dataset_dir = self.datasets_root / safe_name

        if dataset_dir.exists():
            raise ValueError(f"Dataset '{safe_name}' already exists")

        # Create directory structure
        (dataset_dir / "videos").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "thumbnails").mkdir(parents=True, exist_ok=True)

        # Initialize empty dataset.json
        dataset_json = dataset_dir / "dataset.json"
        with open(dataset_json, "w") as f:
            json.dump([], f)

        return dataset_dir

    def add_videos(self, dataset_name: str, video_files: List[str]) -> Dict:
        """Add videos to dataset and generate thumbnails.

        Args:
            dataset_name: Name of the dataset
            video_files: List of video file paths to add

        Returns:
            Dictionary with 'added' and 'failed' lists
        """
        dataset_dir = self.datasets_root / dataset_name
        if not dataset_dir.exists():
            raise ValueError(f"Dataset '{dataset_name}' does not exist")

        videos_dir = dataset_dir / "videos"
        thumbs_dir = dataset_dir / "thumbnails"

        results = {"added": [], "failed": []}

        # Load existing dataset
        dataset_json = dataset_dir / "dataset.json"
        with open(dataset_json) as f:
            items = json.load(f)

        for video_file in video_files:
            try:
                # Copy video to dataset
                video_path = Path(video_file)
                dest_path = videos_dir / video_path.name

                # Skip if already exists
                if dest_path.exists():
                    results["failed"].append({"video": video_path.name, "error": "Already exists"})
                    continue

                shutil.copy2(video_file, dest_path)

                # Generate thumbnail
                thumb_path = self._generate_thumbnail(dest_path, thumbs_dir)

                # Add to dataset.json
                items.append({"media_path": f"videos/{video_path.name}", "caption": ""})

                results["added"].append({"video": video_path.name, "thumbnail": thumb_path.name})

            except Exception as e:
                results["failed"].append({"video": Path(video_file).name, "error": str(e)})

        # Save updated dataset.json
        with open(dataset_json, "w") as f:
            json.dump(items, f, indent=2)

        return results

    def _generate_thumbnail(self, video_path: Path, output_dir: Path) -> Path:
        """Extract first frame as thumbnail.

        Args:
            video_path: Path to the video file
            output_dir: Directory to save thumbnail

        Returns:
            Path to the generated thumbnail

        Raises:
            ValueError: If video cannot be read
        """
        cap = cv2.VideoCapture(str(video_path))

        # Try to get a frame from the middle of the video for better representation
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 10:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Could not read video: {video_path}")

        thumb_path = output_dir / f"{video_path.stem}.jpg"

        # Resize to reasonable thumbnail size while maintaining aspect ratio
        height, width = frame.shape[:2]
        target_size = 256

        if width > height:
            new_width = target_size
            new_height = int(height * (target_size / width))
        else:
            new_height = target_size
            new_width = int(width * (target_size / height))

        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Add padding to make it square
        if new_width != new_height:
            max_dim = max(new_width, new_height)
            square_frame = cv2.copyMakeBorder(
                frame,
                (max_dim - new_height) // 2,
                (max_dim - new_height + 1) // 2,
                (max_dim - new_width) // 2,
                (max_dim - new_width + 1) // 2,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
            frame = square_frame

        cv2.imwrite(str(thumb_path), frame)
        return thumb_path

    def get_dataset_items(self, dataset_name: str) -> List[Dict]:
        """Get all videos and captions in dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            List of dictionaries with 'media_path', 'caption', and 'thumbnail'
        """
        dataset_dir = self.datasets_root / dataset_name
        if not dataset_dir.exists():
            return []

        dataset_json = dataset_dir / "dataset.json"

        if dataset_json.exists():
            with open(dataset_json) as f:
                items = json.load(f)
        else:
            items = []

        # Add thumbnail paths and full video paths
        for item in items:
            video_name = Path(item["media_path"]).name
            thumb_path = dataset_dir / "thumbnails" / f"{Path(video_name).stem}.jpg"
            item["thumbnail"] = str(thumb_path) if thumb_path.exists() else None
            item["full_video_path"] = str(dataset_dir / item["media_path"])

        return items

    def update_caption(self, dataset_name: str, video_name: str, caption: str):
        """Update caption for a specific video.

        Args:
            dataset_name: Name of the dataset
            video_name: Name of the video file
            caption: New caption text
        """
        dataset_dir = self.datasets_root / dataset_name
        dataset_json = dataset_dir / "dataset.json"

        with open(dataset_json) as f:
            items = json.load(f)

        # Find and update the item
        found = False
        for item in items:
            if Path(item["media_path"]).name == video_name:
                item["caption"] = caption
                found = True
                break

        if not found:
            # Add new entry if not found
            items.append({"media_path": f"videos/{video_name}", "caption": caption})

        with open(dataset_json, "w") as f:
            json.dump(items, f, indent=2)

    def delete_video(self, dataset_name: str, video_name: str):
        """Remove video from dataset.

        Args:
            dataset_name: Name of the dataset
            video_name: Name of the video file to delete
        """
        dataset_dir = self.datasets_root / dataset_name

        # Remove from JSON
        dataset_json = dataset_dir / "dataset.json"
        with open(dataset_json) as f:
            items = json.load(f)

        items = [i for i in items if Path(i["media_path"]).name != video_name]

        with open(dataset_json, "w") as f:
            json.dump(items, f, indent=2)

        # Remove files
        (dataset_dir / "videos" / video_name).unlink(missing_ok=True)
        (dataset_dir / "thumbnails" / f"{Path(video_name).stem}.jpg").unlink(missing_ok=True)

    def get_dataset_stats(self, dataset_name: str) -> Dict:
        """Get statistics about dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary with dataset statistics
        """
        dataset_dir = self.datasets_root / dataset_name
        items = self.get_dataset_items(dataset_name)

        return {
            "name": dataset_name,
            "total_videos": len(items),
            "captioned": sum(1 for i in items if i.get("caption") and i.get("caption").strip()),
            "uncaptioned": sum(1 for i in items if not i.get("caption") or not i.get("caption").strip()),
            "preprocessed": (dataset_dir / ".precomputed").exists(),
        }

    def validate_dataset(self, dataset_name: str) -> Dict:
        """Validate dataset and return issues.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary with validation results
        """
        issues = []
        warnings = []

        items = self.get_dataset_items(dataset_name)
        dataset_dir = self.datasets_root / dataset_name

        if not items:
            issues.append("Dataset is empty")
            return {"valid": False, "issues": issues, "warnings": warnings, "total_videos": 0}

        # Check for videos without captions
        uncaptioned = [i for i in items if not i.get("caption") or not i.get("caption").strip()]
        if uncaptioned:
            warnings.append(f"{len(uncaptioned)} videos without captions")

        # Check for missing video files
        for item in items:
            video_path = dataset_dir / item["media_path"]
            if not video_path.exists():
                issues.append(f"Missing video: {item['media_path']}")

        # Check if preprocessed
        if not (dataset_dir / ".precomputed").exists():
            warnings.append("Dataset not preprocessed - will be slower to train")

        # Check video formats/resolutions (sample check)
        for item in items[:5]:  # Check first 5 videos
            video_path = dataset_dir / item["media_path"]
            if video_path.exists():
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    issues.append(f"Cannot read video: {video_path.name}")
                cap.release()

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "total_videos": len(items),
        }

    def delete_dataset(self, dataset_name: str):
        """Delete entire dataset.

        Args:
            dataset_name: Name of the dataset to delete
        """
        dataset_dir = self.datasets_root / dataset_name
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)


