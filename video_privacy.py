from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np


MAX_VIDEO_BYTES = 250 * 1024 * 1024
YUNET_MODEL_NAME = "face_detection_yunet_2023mar.onnx"
YUNET_MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
    "f12e12798e8314f7c074a6656816c048dcc95b7a/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
YUNET_MODEL_MIN_BYTES = 200_000
YUNET_MODEL_MAX_BYTES = 1_000_000
YUNET_SCORE_THRESHOLD = 0.35
FACE_TILE_SCALE = 0.60
FACE_NMS_THRESHOLD = 0.35
FACE_MARGIN_X = 0.20
FACE_MARGIN_TOP = 0.25
FACE_MARGIN_BOTTOM = 0.25
FACE_TRACK_HOLD_SECONDS = 0.25
FACE_TRACK_MIN_HOLD_FRAMES = 2
FACE_TRACK_IOU = 0.2
LOW_LIGHT_LUMA = 90
PIXEL_BLOCK_SIZE = 12
PIXEL_MAX_BLOCKS = 8


def create_sample_video(destination: Path, seconds: int = 8, fps: int = 12) -> None:
    width, height = 640, 360
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV n'a pas pu créer la vidéo de démonstration.")
    try:
        for frame_index in range(seconds * fps):
            frame = np.full((height, width, 3), (52, 42, 31), dtype=np.uint8)
            cv2.rectangle(frame, (32, 48), (608, 330), (75, 78, 80), -1)
            cv2.rectangle(frame, (74, 82), (566, 298), (95, 98, 100), 3)
            cv2.line(frame, (320, 82), (320, 298), (80, 210, 245), 2)
            cv2.putText(frame, "SYNTHETIC AMR TEST", (118, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)
            x = 105 + (frame_index * 5) % 410
            cv2.rectangle(frame, (x, 205), (x + 56, 247), (20, 160, 235), -1)
            cv2.circle(frame, (x + 10, 252), 7, (25, 25, 25), -1)
            cv2.circle(frame, (x + 46, 252), 7, (25, 25, 25), -1)
            writer.write(frame)
    finally:
        writer.release()


def anonymize_video(video_bytes: bytes) -> Tuple[bytes, int, int]:
    if not video_bytes:
        raise ValueError("La vidéo envoyée est vide.")
    if len(video_bytes) > MAX_VIDEO_BYTES:
        raise ValueError("Vidéo trop volumineuse (limite : 250 Mo).")

    model_path = ensure_yunet_model(Path(__file__).resolve().parent / "data" / "models")

    with tempfile.TemporaryDirectory(prefix="robotics-video-") as directory:
        input_path = Path(directory) / "input_video"
        output_path = Path(directory) / "anonymized.mp4"
        input_path.write_bytes(video_bytes)
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError("Vidéo illisible. Vérifiez le codec ou essayez MP4/AVI.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not (fps > 0 and 0 < width <= 8192 and 0 < height <= 8192):
            capture.release()
            raise ValueError("Dimensions ou cadence vidéo non prises en charge.")
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("OpenCV ne peut pas encoder le MP4 de sortie.")
        try:
            detector = cv2.FaceDetectorYN_create(
                str(model_path),
                "",
                (width, height),
                YUNET_SCORE_THRESHOLD,
                0.3,
                5000,
            )
        except cv2.error as error:
            capture.release()
            writer.release()
            raise RuntimeError("OpenCV ne peut pas charger le modèle de visages YuNet.") from error

        tracker = _FaceTracker(
            max(FACE_TRACK_MIN_HOLD_FRAMES, int(round(fps * FACE_TRACK_HOLD_SECONDS)))
        )
        frame_count = 0
        detected_count = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                faces = _detect_faces_multiscale(detector, frame)
                if faces is not None:
                    detected_count += len(faces)
                for face_box in tracker.update(faces):
                    _pixelate_face(frame, face_box, width, height)
                writer.write(frame)
                frame_count += 1
        finally:
            capture.release()
            writer.release()
        if frame_count == 0:
            raise ValueError("La vidéo ne contient aucune image lisible.")
        output = output_path.read_bytes()
        return output, frame_count, detected_count


def ensure_yunet_model(directory: Path) -> Path:
    model_path = directory / YUNET_MODEL_NAME
    if model_path.is_file():
        if _sha256(model_path) == YUNET_MODEL_SHA256:
            return model_path
        raise RuntimeError(
            "Le modèle YuNet local n'a pas l'empreinte attendue : {}.".format(model_path)
        )

    directory.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_suffix(".onnx.part")
    request = Request(YUNET_MODEL_URL, headers={"User-Agent": "MarsoRoboticsVideoPrivacy/1.0"})
    try:
        with urlopen(request, timeout=30) as response, temporary_path.open("wb") as output:
            final_host = (urlparse(response.geturl()).hostname or "").lower()
            if final_host not in ("media.githubusercontent.com", "raw.githubusercontent.com"):
                raise RuntimeError("Le téléchargement du modèle YuNet a redirigé vers un hôte inattendu.")
            total_bytes = 0
            while chunk := response.read(64 * 1024):
                total_bytes += len(chunk)
                if total_bytes > YUNET_MODEL_MAX_BYTES:
                    raise RuntimeError("Le modèle YuNet dépasse la taille maximale autorisée.")
                output.write(chunk)

        if not YUNET_MODEL_MIN_BYTES <= total_bytes <= YUNET_MODEL_MAX_BYTES:
            raise RuntimeError("Le téléchargement du modèle YuNet est incomplet ou invalide.")
        if _sha256(temporary_path) != YUNET_MODEL_SHA256:
            raise RuntimeError("L'empreinte SHA-256 du modèle YuNet ne correspond pas.")
        os.replace(temporary_path, model_path)
        return model_path
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError(
            "Impossible de télécharger le modèle YuNet. Vérifiez la connexion, "
            "ou téléchargez-le depuis le dépôt OpenCV Zoo : {}.".format(YUNET_MODEL_URL)
        ) from error
    finally:
        temporary_path.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        while chunk := model_file.read(64 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _detect_faces_multiscale(detector: Any, frame: np.ndarray) -> np.ndarray | None:
    height, width = frame.shape[:2]
    candidates: list[np.ndarray] = []

    _, full_frame_faces = detector.detect(frame)
    if full_frame_faces is not None:
        candidates.extend(full_frame_faces)

    if cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean() < LOW_LIGHT_LUMA:
        _, low_light_faces = detector.detect(_enhance_low_light(frame))
        if low_light_faces is not None:
            candidates.extend(low_light_faces)

    tile_width = max(1, int(round(width * FACE_TILE_SCALE)))
    tile_height = max(1, int(round(height * FACE_TILE_SCALE)))
    x_positions = sorted({0, max(0, width - tile_width)})
    y_positions = sorted({0, max(0, height - tile_height)})

    for y0 in y_positions:
        for x0 in x_positions:
            tile = frame[y0:y0 + tile_height, x0:x0 + tile_width]
            scaled_tile = cv2.resize(tile, (width, height), interpolation=cv2.INTER_CUBIC)
            _, tile_faces = detector.detect(scaled_tile)
            if tile_faces is None:
                continue

            scale_x = tile_width / width
            scale_y = tile_height / height
            for face in tile_faces:
                mapped_face = face.astype(np.float32, copy=True)
                mapped_face[0] = x0 + mapped_face[0] * scale_x
                mapped_face[1] = y0 + mapped_face[1] * scale_y
                mapped_face[2] *= scale_x
                mapped_face[3] *= scale_y
                mapped_face[4:14:2] = x0 + mapped_face[4:14:2] * scale_x
                mapped_face[5:14:2] = y0 + mapped_face[5:14:2] * scale_y
                candidates.append(mapped_face)

    if not candidates:
        return None

    boxes = [
        [int(round(face[0])), int(round(face[1])),
         max(1, int(round(face[2]))), max(1, int(round(face[3])))]
        for face in candidates
    ]
    scores = [float(face[14]) for face in candidates]
    kept_indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        YUNET_SCORE_THRESHOLD,
        FACE_NMS_THRESHOLD,
    )
    if len(kept_indices) == 0:
        return None
    return np.asarray([candidates[int(index)] for index in np.asarray(kept_indices).reshape(-1)])


def _enhance_low_light(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR)


def _box_iou(first: np.ndarray, second: np.ndarray) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[0] + first[2], second[0] + second[2])
    y1 = min(first[1] + first[3], second[1] + second[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return float(intersection / union) if union > 0 else 0.0


@dataclass
class _FaceTrack:
    box: np.ndarray
    velocity: np.ndarray
    missed_frames: int = 0


class _FaceTracker:
    def __init__(self, hold_frames: int) -> None:
        self.hold_frames = hold_frames
        self._tracks: list[_FaceTrack] = []

    def update(self, faces: np.ndarray | None) -> list[np.ndarray]:
        unmatched = set(range(len(self._tracks)))
        detections = [] if faces is None else [np.asarray(face[:4], dtype=np.float32) for face in faces]

        for detection in detections:
            best_index = max(
                unmatched,
                key=lambda index: _box_iou(
                    self._tracks[index].box + self._tracks[index].velocity,
                    detection,
                ),
                default=None,
            )
            if best_index is not None:
                track = self._tracks[best_index]
                predicted_box = track.box + track.velocity
                if _box_iou(predicted_box, detection) >= FACE_TRACK_IOU:
                    track.velocity = 0.5 * track.velocity + 0.5 * (detection - track.box)
                    track.box = detection
                    track.missed_frames = 0
                    unmatched.remove(best_index)
                    continue
            self._tracks.append(_FaceTrack(detection, np.zeros(4, dtype=np.float32)))

        retained: list[_FaceTrack] = []
        for index, track in enumerate(self._tracks):
            if index in unmatched:
                track.box = track.box + track.velocity
                track.missed_frames += 1
            if track.missed_frames <= self.hold_frames:
                retained.append(track)
        self._tracks = retained
        return [track.box for track in self._tracks]


def _pixelate_face(frame: np.ndarray, face: np.ndarray, width: int, height: int) -> None:
    x, y, face_width, face_height = (int(round(value)) for value in face[:4])
    margin_x = int(round(face_width * FACE_MARGIN_X))
    margin_top = int(round(face_height * FACE_MARGIN_TOP))
    margin_bottom = int(round(face_height * FACE_MARGIN_BOTTOM))
    x0 = max(0, x - margin_x)
    y0 = max(0, y - margin_top)
    x1 = min(width, x + face_width + margin_x)
    y1 = min(height, y + face_height + margin_bottom)
    region = frame[y0:y1, x0:x1]
    if region.size == 0:
        return

    reduced_size = (
        max(1, min(PIXEL_MAX_BLOCKS, region.shape[1] // PIXEL_BLOCK_SIZE)),
        max(1, min(PIXEL_MAX_BLOCKS, region.shape[0] // PIXEL_BLOCK_SIZE)),
    )
    pixelated = cv2.resize(region, reduced_size, interpolation=cv2.INTER_AREA)
    frame[y0:y1, x0:x1] = cv2.resize(
        pixelated,
        (region.shape[1], region.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
