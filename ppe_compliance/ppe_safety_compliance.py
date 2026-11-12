"""Check whether workers are wearing the required PPE"""

from dataclasses import dataclass


BoundingBox = tuple[float, float, float, float]  # (x1, y1, x2, y2)
Point = tuple[float, float] # px, py


@dataclass
class Detection:
    """One object found through detection."""

    class_name: str
    confidence: float
    bbox: BoundingBox

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def contains(self, point: Point) -> bool:
        """Will return True if the point lies inside the box."""
        x1, y1, x2, y2 = self.bbox
        px, py = point
        return x1 <= px <= x2 and y1 <= py <= y2


@dataclass
class ComplianceResult:
    """The PPE check result for one person."""

    person_bbox: BoundingBox
    detected_items: set[str]
    missing_items: set[str]

    @property
    def is_compliant(self) -> bool:
        return not self.missing_items


class PPEComplianceAnalyzer:
    """Match PPE items to each people and report what PPE is missing."""

    def __init__(self, required_items: set[str], confidence_threshold: float = 0.5) -> None:
        if not required_items:
            raise ValueError("Provide at least one required PPE item.")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1.")

        self.required_items = set(required_items)
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def _find_wearer(item: Detection, persons: list[Detection]) -> Detection | None:
        """Return the person wearing this item, or None if nobody is."""
        wearer = None

        for person in persons:
            # Skip people whose box does not contain the item.
            if not person.contains(item.center):
                continue

            # Keep this person if they are the first match,
            # or if their box is smaller than the best match so far.
            if wearer is None or person.area < wearer.area:
                wearer = person

        return wearer

    def analyze(self, detections: list[Detection]) -> list[ComplianceResult]:
        """Return one compliance result per detected person."""
        confident = [d for d in detections if d.confidence >= self.confidence_threshold]
        persons = [d for d in confident if d.class_name == "person"]
        ppe_items = [d for d in confident if d.class_name in self.required_items]

        results = []
        for person in persons:
            
            worn = {
                item.class_name
                for item in ppe_items
                if self._find_wearer(item, persons) is person
            }
            results.append(
                ComplianceResult(
                    person_bbox=person.bbox,
                    detected_items=worn,
                    missing_items=self.required_items - worn,
                )
            )
        return results


def main() -> None:
    """Demonstrate compliance analysis for two workers."""
    detections = [
        Detection("person", 0.98, (50, 50, 250, 500)),
        Detection("person", 0.96, (300, 50, 500, 500)),
        Detection("helmet", 0.94, (110, 60, 180, 120)),
        Detection("helmet", 0.91, (360, 65, 430, 125)),
        Detection("vest", 0.89, (90, 180, 210, 330)),
    ]
    analyzer = PPEComplianceAnalyzer(
        required_items={"helmet", "vest"},
        confidence_threshold=0.60,
    )

    for number, result in enumerate(analyzer.analyze(detections), start=1):
        status = "Compliant" if result.is_compliant else "Non-compliant"
        detected = ", ".join(sorted(result.detected_items)) or "None"
        missing = ", ".join(sorted(result.missing_items)) or "None"
        print(f"Worker {number}: {status}")
        print(f"  Detected PPE: {detected}")
        print(f"  Missing PPE: {missing}")


if __name__ == "__main__":
    main()
