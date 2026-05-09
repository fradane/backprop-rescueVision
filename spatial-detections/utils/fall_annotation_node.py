import depthai as dai
from depthai_nodes.utils import AnnotationHelper
from typing import List

FALL_COLOR = (1.0, 0.45, 0.0, 1.0)
FALL_FILL = (1.0, 0.45, 0.0, 0.2)


class FallAnnotationNode(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.input_detections = self.createInput()
        self.out_annotations = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgAnnotations, True)
            ]
        )
        self.labels: List[str] = []

    def build(
        self,
        input_detections: dai.Node.Output,
        labels: List[str],
    ) -> "FallAnnotationNode":
        self.labels = labels
        self.link_args(input_detections)
        return self

    def process(self, detections_msg: dai.SpatialImgDetections) -> None:
        annotation_helper = AnnotationHelper()

        for detection in detections_msg.detections:
            label_name = (
                self.labels[detection.label]
                if detection.label < len(self.labels)
                else str(detection.label)
            )
            z = detection.spatialCoordinates.z
            confidence = int(detection.confidence * 100)

            annotation_helper.draw_rectangle(
                top_left=(detection.xmin, detection.ymin),
                bottom_right=(detection.xmax, detection.ymax),
                outline_color=FALL_COLOR,
                fill_color=FALL_FILL,
                thickness=3.0,
            )
            annotation_helper.draw_text(
                text=f"{label_name} {confidence}%\ndist: {z / 1000:.1f}m",
                position=(detection.xmin + 0.01, detection.ymin + 0.2),
                size=12,
                color=FALL_COLOR,
            )

        annotations = annotation_helper.build(
            timestamp=detections_msg.getTimestamp(),
            sequence_num=detections_msg.getSequenceNum(),
        )
        self.out_annotations.send(annotations)
