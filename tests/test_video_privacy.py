from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

import video_privacy


class FakeResponse:
    def __init__(self, payload: bytes, url: str = video_privacy.YUNET_MODEL_URL) -> None:
        self.payload = payload
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def geturl(self) -> str:
        return self.url

    def read(self, size: int = -1) -> bytes:
        chunk, self.payload = self.payload[:size], self.payload[size:]
        return chunk


def test_yunet_model_is_downloaded_once_and_hash_verified(tmp_path: Path, monkeypatch):
    payload = b"x" * video_privacy.YUNET_MODEL_MIN_BYTES
    monkeypatch.setattr(
        video_privacy,
        "YUNET_MODEL_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    calls = []

    def fake_urlopen(*_args, **_kwargs):
        calls.append(True)
        return FakeResponse(payload)

    monkeypatch.setattr(video_privacy, "urlopen", fake_urlopen)

    first_path = video_privacy.ensure_yunet_model(tmp_path)
    second_path = video_privacy.ensure_yunet_model(tmp_path)

    assert first_path == second_path
    assert first_path.read_bytes() == payload
    assert len(calls) == 1
    assert not first_path.with_suffix(".onnx.part").exists()


def test_yunet_model_rejects_redirect_to_untrusted_host(tmp_path: Path, monkeypatch):
    payload = b"x" * video_privacy.YUNET_MODEL_MIN_BYTES
    monkeypatch.setattr(video_privacy, "YUNET_MODEL_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(
        video_privacy,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(payload, "https://example.com/model.onnx"),
    )

    with pytest.raises(RuntimeError, match="hôte inattendu"):
        video_privacy.ensure_yunet_model(tmp_path)

    assert not (tmp_path / video_privacy.YUNET_MODEL_NAME).exists()
    assert not (tmp_path / (video_privacy.YUNET_MODEL_NAME + ".part")).exists()


def test_pixelation_includes_margin_but_leaves_distant_regions_unchanged():
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    frame[:] = np.random.default_rng(0).integers(0, 255, frame.shape, dtype=np.uint8)
    original = frame.copy()

    video_privacy._pixelate_face(
        frame,
        np.array([40.0, 25.0, 20.0, 20.0]),
        width=100,
        height=80,
    )

    assert not np.array_equal(frame[20:50, 36:64], original[20:50, 36:64])
    assert np.array_equal(frame[:20], original[:20])
    assert np.array_equal(frame[50:], original[50:])
    assert np.array_equal(frame[:, :36], original[:, :36])
    assert np.array_equal(frame[:, 64:], original[:, 64:])


def test_face_mode_uses_yunet_and_anonymizes_each_frame(tmp_path: Path, monkeypatch):
    source_path = tmp_path / "synthetic.avi"
    video_privacy.create_sample_video(source_path, seconds=1, fps=2)
    source_bytes = source_path.read_bytes()
    model_path = tmp_path / video_privacy.YUNET_MODEL_NAME
    model_path.write_bytes(b"verified test model")
    monkeypatch.setattr(video_privacy, "ensure_yunet_model", lambda _directory: model_path)

    class FakeDetector:
        def __init__(self):
            self.calls = 0

        def detect(self, _frame):
            self.calls += 1
            if self.calls % 6 == 1:
                return 1, np.array([[100, 100, 30, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9]])
            return 1, None

    detector = FakeDetector()
    monkeypatch.setattr(
        cv2,
        "FaceDetectorYN_create",
        lambda *_args: detector,
    )

    output, frame_count, detections = video_privacy.anonymize_video(source_bytes)

    assert frame_count == 2
    assert detections == 2
    # The synthetic scene is dark: full frame + low-light pass + four tiles.
    assert detector.calls == frame_count * 6
    assert output


def test_multiscale_detection_finds_small_faces_and_suppresses_duplicates():
    class FakeDetector:
        def __init__(self):
            self.calls = 0

        def detect(self, _frame):
            self.calls += 1
            if self.calls == 1:
                return 1, np.array([[60, 60, 24, 24, *([0] * 10), 0.9]])
            if self.calls == 2:
                return 1, np.array([[100, 100, 40, 40, *([0] * 10), 0.95]])
            return 1, None

    detector = FakeDetector()
    faces = video_privacy._detect_faces_multiscale(
        detector,
        np.full((360, 640, 3), 128, dtype=np.uint8),
    )

    assert detector.calls == 5
    assert faces is not None
    assert len(faces) == 1
    assert np.allclose(faces[0][:4], [60, 60, 24, 24])


def test_dark_frames_get_an_extra_low_light_detection_pass():
    class FakeDetector:
        def __init__(self):
            self.frames = []

        def detect(self, frame):
            self.frames.append(frame)
            return 1, None

    detector = FakeDetector()
    dark_frame = np.full((360, 640, 3), 20, dtype=np.uint8)
    dark_frame[100:200, 100:200] = 40

    video_privacy._detect_faces_multiscale(detector, dark_frame)

    assert len(detector.frames) == 6
    assert detector.frames[1].std() > detector.frames[0].std()


def test_tracker_masks_face_between_missed_frames_and_extrapolates_motion():
    tracker = video_privacy._FaceTracker(hold_frames=2)
    first_face = np.array([[100, 100, 40, 40]], dtype=np.float32)
    next_face = np.array([[110, 100, 40, 40]], dtype=np.float32)

    assert np.allclose(tracker.update(first_face)[0], [100, 100, 40, 40])
    assert np.allclose(tracker.update(next_face)[0], [110, 100, 40, 40])
    held = tracker.update(None)
    assert len(held) == 1
    assert held[0][0] > 110
    assert len(tracker.update(None)) == 1
    assert tracker.update(None) == []


def test_tracker_keeps_two_faces_separate():
    tracker = video_privacy._FaceTracker(hold_frames=2)
    faces = np.array([[40, 50, 30, 30], [200, 50, 30, 30]], dtype=np.float32)

    boxes = tracker.update(faces)

    assert len(boxes) == 2
    assert np.allclose(boxes[0], faces[0])
    assert np.allclose(boxes[1], faces[1])


def test_pixelation_is_coarse_on_large_faces():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)

    video_privacy._pixelate_face(frame, np.array([400.0, 200.0, 300.0, 300.0]), 1280, 720)

    region = frame[125:575, 340:760]
    distinct_rows = len(np.unique(region[:, 0], axis=0))
    assert distinct_rows <= video_privacy.PIXEL_MAX_BLOCKS
