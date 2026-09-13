"""Read-only DenPAR structural annotation loading and validation."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
from typing import Any

from PIL import Image
import numpy as np


@dataclass(frozen=True)
class ImageInfo:
    image_id: str
    filename: str
    width: int
    height: int
    mode: str
    path: str


@dataclass(frozen=True)
class Issue:
    image_id: str
    category: str
    message: str
    source: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class ImageAnnotations:
    image: ImageInfo
    tooth_masks: list[dict[str, Any]]
    radiograph_mask: dict[str, Any] | None
    coco_annotations: list[dict[str, Any]]
    keypoints: dict[str, Any] | None
    bone_lines: dict[str, Any] | None
    characteristics: dict[str, Any] | None


@dataclass
class AuditResult:
    images: list[ImageInfo]
    records: list[ImageAnnotations]
    issues: list[Issue]
    statistics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "images": [asdict(item) for item in self.images],
            "records": [
                {
                    "image": asdict(record.image),
                    "tooth_masks": record.tooth_masks,
                    "radiograph_mask": record.radiograph_mask,
                    "coco_annotations": record.coco_annotations,
                    "keypoints": record.keypoints,
                    "bone_lines": record.bone_lines,
                    "characteristics": record.characteristics,
                }
                for record in self.records
            ],
            "issues": [issue.as_dict() for issue in self.issues],
            "statistics": self.statistics,
        }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_workbook(path: Path) -> dict[str, dict[str, str]]:
    namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main", "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        try:
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(node.itertext()) for node in shared_root.findall(".//a:si", namespace)]
        except KeyError:
            pass
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relationship_map = {node.attrib["Id"]: node.attrib["Target"] for node in relationships}
        sheet = workbook.find(".//a:sheet", namespace)
        if sheet is None:
            return {}
        target = relationship_map[sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]]
        target = target if target.startswith("xl/") else f"xl/{target}"
        worksheet = ET.fromstring(archive.read(target))
        rows: list[list[str]] = []
        for row in worksheet.findall(".//a:sheetData/a:row", namespace):
            values: list[str] = []
            for cell in row.findall("a:c", namespace):
                value = cell.find("a:v", namespace)
                text = value.text if value is not None else ""
                if cell.attrib.get("t") == "s" and text:
                    text = shared[int(text)]
                values.append(text)
            rows.append(values)
    if not rows:
        return {}
    headers = rows[0]
    result: dict[str, dict[str, str]] = {}
    for row in rows[1:]:
        values = dict(zip(headers, row))
        raw_id = values.get("id", "")
        if raw_id:
            result[str(int(float(raw_id)))] = values
    return result


def _image_info(path: Path) -> ImageInfo:
    with Image.open(path) as image:
        return ImageInfo(path.stem, path.name, image.width, image.height, image.mode, str(path))


def _mask_info(path: Path, image: ImageInfo, source: str) -> dict[str, Any]:
    with Image.open(path) as mask:
        pixels = mask.convert("L")
        foreground = np.asarray(pixels) != 0
        nonzero = int(np.count_nonzero(foreground))
        # This is descriptive geometry only.  In particular, it does not assign
        # maskN to a COCO annotation or an FDI tooth.
        bbox = None
        if nonzero:
            rows, columns = np.where(foreground)
            left, top = int(columns.min()), int(rows.min())
            bbox = [left, top, int(columns.max()) + 1, int(rows.max()) + 1]
        return {"filename": path.name, "path": str(path), "width": mask.width, "height": mask.height, "mode": mask.mode, "nonzero_pixels": nonzero, "bbox_xyxy": bbox, "source": source}


def _in_bounds(x: float, y: float, image: ImageInfo) -> bool:
    return 0 <= x < image.width and 0 <= y < image.height


def _validate_point_list(points: Any, image: ImageInfo, issues: list[Issue], image_id: str, source: str, label: str) -> None:
    if not isinstance(points, list):
        issues.append(Issue(image_id, "invalid_format", f"{label} is not a list", source))
        return
    for index, point in enumerate(points):
        if not isinstance(point, list) or len(point) != 2 or not all(isinstance(value, (int, float)) for value in point):
            issues.append(Issue(image_id, "invalid_coordinate", f"{label}[{index}] is not [x, y]", source))
        elif not _in_bounds(float(point[0]), float(point[1]), image):
            issues.append(Issue(image_id, "out_of_bounds", f"{label}[{index}] lies outside image bounds", source))


def audit_validation_dataset(validation_dir: Path) -> AuditResult:
    images_dir = validation_dir / "Images"
    tooth_dir = validation_dir / "Masks (Tooth-wise)"
    radiograph_dir = validation_dir / "Masks (Radiograph-wise)"
    keypoint_dir = validation_dir / "Key Points Annotations"
    bone_dir = validation_dir / "Bone Level Annotations"
    workbook_path = validation_dir.parent / "Characteristics of radiographs included.xlsx"
    images = [_image_info(path) for path in sorted(images_dir.glob("*.jpg"))]
    image_by_id = {image.image_id: image for image in images}
    issues: list[Issue] = []
    coco_path = tooth_dir / "coco_format_instances_valid.json"
    coco = _read_json(coco_path) if coco_path.exists() else {"images": [], "annotations": [], "categories": []}
    coco_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    coco_image_names: dict[str, str] = {}
    coco_numeric_ids: dict[str, str] = {}
    for coco_image in coco.get("images", []):
        coco_id = str(Path(coco_image.get("file_name", "")).stem)
        coco_image_names[coco_id] = coco_image.get("file_name", "")
        coco_numeric_ids[str(coco_image.get("id"))] = coco_id
    for annotation in coco.get("annotations", []):
        coco_image_id = coco_numeric_ids.get(str(annotation.get("image_id")))
        if coco_image_id:
            coco_by_id[coco_image_id].append(annotation)
    characteristics = _read_workbook(workbook_path) if workbook_path.exists() else {}
    records: list[ImageAnnotations] = []
    mask_count = 0
    empty_masks = 0
    dimensions_match = True
    duplicate_annotations = 0
    coordinate_stats = Counter()
    for image in images:
        image_id = image.image_id
        mask_folder = tooth_dir / image_id
        masks: list[dict[str, Any]] = []
        if not mask_folder.exists():
            issues.append(Issue(image_id, "missing_file", "tooth-wise mask folder is missing", "Masks (Tooth-wise)"))
        else:
            for mask_path in sorted(mask_folder.glob("*.png")):
                info = _mask_info(mask_path, image, "tooth-wise")
                masks.append(info)
                mask_count += 1
                if info["nonzero_pixels"] == 0:
                    empty_masks += 1
                    issues.append(Issue(image_id, "empty_mask", f"empty tooth mask: {mask_path.name}", "Masks (Tooth-wise)"))
                if (info["width"], info["height"]) != (image.width, image.height):
                    dimensions_match = False
                    issues.append(Issue(image_id, "dimension_mismatch", f"mask is {info['width']}x{info['height']}, image is {image.width}x{image.height}", "Masks (Tooth-wise)"))
        radiograph_path = radiograph_dir / f"{image_id}.png"
        radiograph_mask = _mask_info(radiograph_path, image, "radiograph-wise") if radiograph_path.exists() else None
        if radiograph_mask is None:
            dimensions_match = False
            issues.append(Issue(image_id, "missing_file", "radiograph-wise mask is missing", "Masks (Radiograph-wise)"))
        elif (radiograph_mask["width"], radiograph_mask["height"]) != (image.width, image.height):
            dimensions_match = False
            issues.append(Issue(image_id, "dimension_mismatch", "radiograph-wise mask dimensions do not match image", "Masks (Radiograph-wise)"))
        keypoint_path = keypoint_dir / f"{image_id}.json"
        keypoints = _read_json(keypoint_path) if keypoint_path.exists() else None
        if keypoints is None:
            issues.append(Issue(image_id, "missing_file", "key-point JSON is missing", "Key Points Annotations"))
        else:
            if keypoints.get("Image_id") != f"{image_id}.jpg":
                issues.append(Issue(image_id, "correspondence", "key-point Image_id does not match image filename", "Key Points Annotations"))
            for index, bbox in enumerate(keypoints.get("bboxes", [])):
                if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(value, (int, float)) for value in bbox) or not (bbox[0] < bbox[2] <= image.width and bbox[1] < bbox[3] <= image.height):
                    issues.append(Issue(image_id, "invalid_coordinate", f"key-point bbox {index} is not valid corner coordinates", "Key Points Annotations"))
                else:
                    coordinate_stats["keypoint_bboxes"] += 1
            _validate_point_list(keypoints.get("CEJ_Points", []), image, issues, image_id, "Key Points Annotations", "CEJ_Points")
            _validate_point_list(keypoints.get("Apex_Points", []), image, issues, image_id, "Key Points Annotations", "Apex_Points")
            if not keypoints.get("CEJ_Points"):
                issues.append(Issue(image_id, "missing_annotation", "no CEJ points", "Key Points Annotations"))
            if not keypoints.get("Apex_Points"):
                issues.append(Issue(image_id, "missing_annotation", "no apex points", "Key Points Annotations"))
        bone_path = bone_dir / f"{image_id}.json"
        bone_lines = _read_json(bone_path) if bone_path.exists() else None
        if bone_lines is None:
            issues.append(Issue(image_id, "missing_file", "bone-level JSON is missing", "Bone Level Annotations"))
        else:
            if bone_lines.get("Image_id") != f"{image_id}.jpg":
                issues.append(Issue(image_id, "correspondence", "bone Image_id does not match image filename", "Bone Level Annotations"))
            lines = bone_lines.get("Bone_Lines", [])
            if not lines:
                issues.append(Issue(image_id, "missing_annotation", "no bone lines", "Bone Level Annotations"))
            for line in lines:
                _validate_point_list(line, image, issues, image_id, "Bone Level Annotations", "Bone_Lines")
        coco_records = coco_by_id.get(image_id, [])
        if len(coco_records) != len(masks):
            issues.append(Issue(image_id, "correspondence", f"COCO instances={len(coco_records)} but tooth masks={len(masks)}", "COCO"))
        seen_geometries: set[str] = set()
        for annotation in coco_records:
            bbox = annotation.get("bbox", [])
            valid_bbox = isinstance(bbox, list) and len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0 and 0 <= bbox[0] and 0 <= bbox[1] and bbox[0] + bbox[2] <= image.width and bbox[1] + bbox[3] <= image.height
            if not valid_bbox:
                issues.append(Issue(image_id, "invalid_coordinate", "COCO bbox is invalid or out of bounds", "COCO"))
            segmentation = annotation.get("segmentation", [])
            if not isinstance(segmentation, list) or not segmentation or any(len(polygon) < 6 or len(polygon) % 2 for polygon in segmentation if isinstance(polygon, list)):
                issues.append(Issue(image_id, "invalid_polygon", "COCO polygon is empty or malformed", "COCO"))
            geometry_key = json.dumps([bbox, segmentation], sort_keys=True)
            if geometry_key in seen_geometries:
                duplicate_annotations += 1
                issues.append(Issue(image_id, "duplicate_annotation", "duplicate COCO geometry", "COCO"))
            seen_geometries.add(geometry_key)
        if image_id not in coco_image_names:
            issues.append(Issue(image_id, "missing_file", "image is missing from COCO images list", "COCO"))
        if image_id not in characteristics:
            issues.append(Issue(image_id, "missing_file", "characteristics spreadsheet row is missing", "Characteristics workbook"))
        records.append(ImageAnnotations(image, masks, radiograph_mask, coco_records, keypoints, bone_lines, characteristics.get(image_id)))
    stats = {
        "number_of_images": len(images),
        "number_of_tooth_instances": len(coco.get("annotations", [])),
        "number_of_tooth_masks": mask_count,
        "number_of_keypoint_records": sum(record.keypoints is not None for record in records),
        "number_of_bone_line_records": sum(record.bone_lines is not None for record in records),
        "number_of_radiograph_masks": sum(record.radiograph_mask is not None for record in records),
        "coco_categories": coco.get("categories", []),
        "tooth_instances_per_image": dict(Counter(len(record.coco_annotations) for record in records)),
        "tooth_masks_per_image": dict(Counter(len(record.tooth_masks) for record in records)),
        "bone_lines_per_image": dict(Counter(len(record.bone_lines.get("Bone_Lines", [])) for record in records if record.bone_lines)),
        "dimensions_consistent": dimensions_match,
        "empty_masks": empty_masks,
        "duplicate_annotations": duplicate_annotations,
        "issue_count": len(issues),
        "unresolved_correspondence_issues": sum(issue.category == "correspondence" for issue in issues),
    }
    return AuditResult(images, records, issues, stats)


def find_validation_record(validation_dir: Path, image_id: str) -> tuple[AuditResult, ImageAnnotations]:
    result = audit_validation_dataset(validation_dir)
    for record in result.records:
        if record.image.image_id == image_id:
            return result, record
    raise FileNotFoundError(image_id)
