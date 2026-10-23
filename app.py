from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from observability import RequestMetrics, configure_logging, read_recent_logs
from mlops import MLOpsService
from operations import PipelineOrchestrator
from robot_data import analyze_text, make_sample_bundle
from video_privacy import anonymize_video


ROOT = Path(__file__).resolve().parent
MAX_REQUEST_BYTES = 250 * 1024 * 1024
MAX_ANALYSIS_REQUEST_BYTES = 100 * 1024 * 1024
MAX_ANALYSIS_FILE_BYTES = 25 * 1024 * 1024
HOST = os.environ.get("ROBOTICS_DEMO_HOST", "127.0.0.1")
PORT = int(os.environ.get("ROBOTICS_DEMO_PORT", "8765"))
LOG = logging.getLogger("robotics-demo")
METRICS = RequestMetrics()
ORCHESTRATOR: PipelineOrchestrator | None = None
MLOPS: MLOpsService | None = None
LOG_PATH = ROOT / "logs" / "app.jsonl"


class RequestError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


class Handler(BaseHTTPRequestHandler):
    server_version = "RoboticsDataDemo/0.1"

    def log_message(self, format_string: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        self._begin_request()
        path = urlparse(self.path).path
        try:
            if path == "/":
                self._send_file(ROOT / "static" / "index.html", "text/html; charset=utf-8")
            elif path == "/api/health":
                running = bool(ORCHESTRATOR and ORCHESTRATOR.snapshot()["scheduler_running"])
                status = "ok" if running else "degraded"
                self._send_json({"status": status, "local_only": True, "scheduler_running": running})
            elif path == "/api/operations":
                if ORCHESTRATOR is None:
                    raise RequestError(HTTPStatus.SERVICE_UNAVAILABLE, "Orchestrateur non démarré.")
                snapshot = ORCHESTRATOR.snapshot()
                snapshot["logs"] = read_recent_logs(LOG_PATH)
                self._send_json(snapshot)
            elif path == "/api/ml/status":
                if MLOPS is None:
                    raise RequestError(HTTPStatus.SERVICE_UNAVAILABLE, "Service MLOps non démarré.")
                self._send_json(MLOPS.status())
            elif path == "/metrics":
                body = METRICS.prometheus_metrics()
                if ORCHESTRATOR is not None:
                    body += ORCHESTRATOR.prometheus_metrics()
                if MLOPS is not None:
                    body += MLOPS.prometheus_metrics()
                self._send_bytes(body.encode("utf-8"), "text/plain; version=0.0.4; charset=utf-8")
            else:
                self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except RequestError as error:
            self._send_json({"error": str(error)}, error.status)
        except Exception:
            LOG.exception("http_request_failed", extra={"request_id": self._request_id})
            self._send_json({"error": "Erreur interne. Consultez les journaux locaux."}, HTTPStatus.INTERNAL_SERVER_ERROR)
        finally:
            self._finish_request(path)

    def do_POST(self) -> None:
        self._begin_request()
        path = urlparse(self.path).path
        try:
            self._validate_local_origin()
            if path in ("/api/sample", "/api/jobs/run", "/api/jobs/stop") or (
                path == "/api/ml/train"
                and not self.headers.get("Content-Length")
                and not self.headers.get("Transfer-Encoding")
            ):
                body = b""
            else:
                limit = MAX_ANALYSIS_REQUEST_BYTES if path in (
                    "/api/analyze", "/api/jobs/schedule", "/api/ml/predict"
                ) else MAX_REQUEST_BYTES
                body = self._read_body(limit)
            if path == "/api/sample":
                self._send_sample()
            elif path == "/api/analyze":
                self._analyze(body)
            elif path == "/api/anonymize":
                self._anonymize(body)
            elif path == "/api/jobs/schedule":
                self._schedule(body)
            elif path == "/api/jobs/run":
                self._run_job()
            elif path == "/api/jobs/stop":
                self._stop_job()
            elif path == "/api/ml/train":
                self._train_model(body)
            elif path == "/api/ml/predict":
                self._predict(body)
            else:
                self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except RequestError as error:
            self._send_json({"error": str(error)}, error.status)
        except (ValueError, UnicodeDecodeError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception:
            LOG.exception("http_request_failed", extra={"request_id": self._request_id})
            self._send_json(
                {"error": "Traitement impossible. Consultez les journaux locaux pour le détail."},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
        finally:
            self._finish_request(path)

    def _begin_request(self) -> None:
        self._request_id = uuid.uuid4().hex
        self._request_started = time.monotonic()
        self._response_status = int(HTTPStatus.INTERNAL_SERVER_ERROR)

    def _finish_request(self, path: str) -> None:
        duration = time.monotonic() - self._request_started
        route = _route_label(path)
        status = self._response_status
        METRICS.observe(self.command, route, status, duration)
        if self.command == "GET" and route == "/api/operations":
            return
        LOG.info(
            "http_request",
            extra={
                "request_id": self._request_id,
                "method": self.command,
                "route": route,
                "status": status,
                "duration_seconds": round(duration, 6),
            },
        )

    def send_response(self, code: int, message: str | None = None) -> None:
        self._response_status = code
        super().send_response(code, message)

    def end_headers(self) -> None:
        self.send_header("X-Request-ID", self._request_id)
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def _validate_local_origin(self) -> None:
        allowed_hosts = {"127.0.0.1:{}".format(PORT), "localhost:{}".format(PORT)}
        if self.headers.get("Host", "") not in allowed_hosts:
            raise RequestError(HTTPStatus.FORBIDDEN, "Cet endpoint accepte uniquement les hôtes locaux.")
        origin = self.headers.get("Origin")
        allowed_origins = {"http://127.0.0.1:{}".format(PORT), "http://localhost:{}".format(PORT)}
        if origin and origin not in allowed_origins:
            raise RequestError(HTTPStatus.FORBIDDEN, "Origine non autorisée.")

    def _read_body(self, limit: int) -> bytes:
        raw_length = self.headers.get("Content-Length", "")
        if not raw_length.isdigit():
            raise RequestError(HTTPStatus.LENGTH_REQUIRED, "Content-Length est requis.")
        length = int(raw_length)
        if length > limit:
            raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Requête trop volumineuse (limite : {} Mo).".format(limit // (1024 * 1024)))
        return self.rfile.read(length)

    def _analyze(self, body: bytes) -> None:
        try:
            request = json.loads(body.decode("utf-8"))
            filename = str(request["filename"])
            content = str(request["content"])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("Envoyez un fichier CSV ou JSONL valide.") from error
        if len(content.encode("utf-8")) > MAX_ANALYSIS_FILE_BYTES:
            raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Fichier de données trop volumineux (limite : 25 Mo).")
        suffix = Path(filename).suffix.lower()
        if suffix not in (".csv", ".jsonl", ".ndjson"):
            raise ValueError("Formats acceptés : CSV et JSONL.")
        self._send_json(analyze_text(content, suffix))

    def _schedule(self, body: bytes) -> None:
        if ORCHESTRATOR is None:
            raise RequestError(HTTPStatus.SERVICE_UNAVAILABLE, "Orchestrateur non démarré.")
        try:
            request = json.loads(body.decode("utf-8"))
            interval = request["interval_seconds"]
            if isinstance(interval, bool) or not isinstance(interval, int):
                raise ValueError("La fréquence doit être un nombre entier de secondes.")
            result = ORCHESTRATOR.configure_schedule(interval)
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("Envoyez interval_seconds dans le corps JSON.") from error
        except ValueError as error:
            raise RequestError(HTTPStatus.BAD_REQUEST, str(error)) from error
        self._send_json(result)

    def _run_job(self) -> None:
        if ORCHESTRATOR is None:
            raise RequestError(HTTPStatus.SERVICE_UNAVAILABLE, "Orchestrateur non démarré.")
        succeeded = ORCHESTRATOR.run_now()
        if succeeded is None:
            raise RequestError(HTTPStatus.CONFLICT, "Ce pipeline est déjà en cours.")
        if not succeeded:
            raise RequestError(HTTPStatus.INTERNAL_SERVER_ERROR, "Le pipeline a échoué. Consultez l'historique et les logs locaux.")
        self._send_json(ORCHESTRATOR.snapshot())

    def _stop_job(self) -> None:
        if ORCHESTRATOR is None:
            raise RequestError(HTTPStatus.SERVICE_UNAVAILABLE, "Orchestrateur non démarré.")
        self._send_json(ORCHESTRATOR.disable_schedule())

    def _train_model(self, body: bytes) -> None:
        if MLOPS is None:
            raise RequestError(HTTPStatus.SERVICE_UNAVAILABLE, "Service MLOps non démarré.")
        content = None
        if body:
            try:
                request = json.loads(body.decode("utf-8"))
                content = request["content"]
                filename = str(request["filename"])
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError("Envoyez un CSV avec filename et content.") from error
            if not isinstance(content, str):
                raise ValueError("Le contenu CSV doit être du texte.")
            if Path(filename).suffix.lower() != ".csv":
                raise ValueError("L'entraînement MLOps accepte uniquement les fichiers CSV.")
            if len(content.encode("utf-8")) > MAX_ANALYSIS_FILE_BYTES:
                raise RequestError(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    "Fichier de données trop volumineux (limite : 25 Mo).",
                )
        self._send_json(MLOPS.train(content))

    def _predict(self, body: bytes) -> None:
        if MLOPS is None:
            raise RequestError(HTTPStatus.SERVICE_UNAVAILABLE, "Service MLOps non démarré.")
        try:
            request = json.loads(body.decode("utf-8"))
            content = request["content"]
            filename = str(request["filename"])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("Envoyez un CSV avec filename et content.") from error
        if not isinstance(content, str):
            raise ValueError("Le contenu CSV doit être du texte.")
        if Path(filename).suffix.lower() != ".csv":
            raise ValueError("La prédiction MLOps accepte uniquement les fichiers CSV.")
        if len(content.encode("utf-8")) > MAX_ANALYSIS_FILE_BYTES:
            raise RequestError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "Fichier de données trop volumineux (limite : 25 Mo).",
            )
        self._send_json(MLOPS.predict(content))

    def _anonymize(self, body: bytes) -> None:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Envoyez une vidéo au format multipart/form-data.")
        message = BytesParser(policy=policy.default).parsebytes(
            b"Content-Type: " + content_type.encode("ascii", "strict")
            + b"\r\nMIME-Version: 1.0\r\n\r\n"
            + body
        )
        upload = next((part for part in message.iter_parts() if part.get_filename()), None)
        if upload is None:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Aucun fichier vidéo reçu.")
        filename = Path(upload.get_filename() or "video.avi").name
        if Path(filename).suffix.lower() not in (".mp4", ".avi", ".mov", ".mkv"):
            raise RequestError(HTTPStatus.BAD_REQUEST, "Formats vidéo acceptés : MP4, AVI, MOV, MKV.")
        output, frames, detections = anonymize_video(upload.get_payload(decode=True) or b"")
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(filename).stem)[:80] or "video"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Disposition", 'attachment; filename="{}_anonymisee.mp4"'.format(safe_name))
        self.send_header("X-Processed-Frames", str(frames))
        self.send_header("X-Detected-Faces", str(detections))
        self.send_header("Content-Length", str(len(output)))
        self.end_headers()
        self.wfile.write(output)

    def _send_sample(self) -> None:
        archive = make_sample_bundle()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition", 'attachment; filename="robotics-demo-samples.zip"')
        self.send_header("Content-Length", str(len(archive)))
        self.end_headers()
        self.wfile.write(archive)

    def _send_json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self._send_bytes(encoded, "application/json; charset=utf-8", status)

    def _send_bytes(self, encoded: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_file(self, path: Path, content_type: str) -> None:
        self._send_bytes(path.read_bytes(), content_type)


def main() -> None:
    global MLOPS, ORCHESTRATOR, LOG_PATH
    LOG_PATH = configure_logging(ROOT / "logs")
    MLOPS = MLOpsService(
        ROOT / "data" / "mlops.sqlite3",
        ROOT / "data" / "models",
        ROOT / "data" / "evidently",
    )
    ORCHESTRATOR = PipelineOrchestrator(
        ROOT / "data" / "operations.sqlite3",
        model_trainer=MLOPS.train,
    )
    ORCHESTRATOR.start()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print("Robotics Data Demo : http://{}:{}".format(HOST, PORT))
    print("Serveur local uniquement. Métriques : http://{}:{}/metrics".format(HOST, PORT))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt du serveur.")
    finally:
        server.server_close()
        ORCHESTRATOR.stop()


def _route_label(path: str) -> str:
    known_routes = {
        "/", "/api/health", "/api/operations", "/api/analyze", "/api/sample",
        "/api/anonymize", "/api/jobs/schedule", "/api/jobs/run", "/api/jobs/stop", "/metrics",
        "/api/ml/status", "/api/ml/train", "/api/ml/predict",
    }
    return path if path in known_routes else "/other"


if __name__ == "__main__":
    main()
