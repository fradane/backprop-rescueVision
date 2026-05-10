import depthai as dai
import cv2
from datetime import datetime


class VideoRecorder(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.input_frame = self.createInput()
        self._writer = None
        self._path = None

    def build(self, input_frame: dai.Node.Output, fps: float) -> "VideoRecorder":
        self._fps = fps
        self.link_args(input_frame)
        return self

    def process(self, frame_msg: dai.ImgFrame) -> None:
        img = frame_msg.getCvFrame()
        h, w = img.shape[:2]

        if self._writer is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._path = f"recording_{ts}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self._path, fourcc, self._fps, (w, h))
            print(f"[VideoRecorder] Saving to {self._path}")

        self._writer.write(img)

    def close(self) -> None:
        if self._writer:
            self._writer.release()
            print(f"[VideoRecorder] Saved: {self._path}")
