"""
Face detection + 512-d embedding extraction for a single in-memory image,
per spec Section 4/8, Phase 7 §4.

Uses `insightface`'s "buffalo_l" model pack: SCRFD for detection, an
ArcFace-class model for the 512-d recognition embedding. Replaces the
Phase 1-5 pipeline (dlib HOG/CNN detector + `face_recognition`'s 128-d
embedding) per PHASE7_SCALE_ARCHITECTURE.md §4 -- SCRFD is meaningfully
better than dlib's HOG detector at small/angled/partially-occluded faces
(e.g. someone further back in a group photo, or not looking straight at
the camera), and ArcFace-class embeddings give better-separated same-
person/different-person distributions than dlib's older embedding model,
which is the whole point of a v1 spent on `face_recognition` in the first
place -- see spec Section 4's rationale for planning to swap it out once
the rest of the pipeline was proven out.

**Embedding space changed, not just the dimension.** `MATCH_THRESHOLD`
(matching.py) and `CLUSTER_EPS`/`CLUSTER_MIN_SAMPLES` (clustering.py)
were tuned by hand against the *old* 128-d dlib embedding space -- those
numbers do not carry over to ArcFace's 512-d space and need to be
re-tuned from scratch against real photos before this is trusted for
matching (per README's §13, "labeled eval set / threshold tuning", and
`EMBEDDING_DIM` in `.env` needs to move from `128` to `512` before the
Postgres/pgvector path in §6 can be re-migrated with real vectors -- see
`app/database_pg.py`). `embedding_to_blob`/`blob_to_embedding` are
dimension-agnostic (they just serialize whatever-length float64 array
they're given), so nothing downstream needs a code change for the new
dimension -- only the tuning constants and `EMBEDDING_DIM` do.

Deliberately stateless / single-image-in, list-of-faces-out: no DB or
Drive knowledge here so it stays independently testable.
"""
import io
import logging
import os
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# Which insightface model pack to load. "buffalo_l" (SCRFD-10GF detector +
# ArcFace-class "w600k_r50" recognition model, both ONNX) is insightface's
# standard general-purpose pack and what spec Section 4's "SCRFD/ArcFace-
# class" language points at. Configurable in case a smaller/faster pack
# (e.g. "buffalo_s") is ever worth trading accuracy for ingestion speed.
FACE_MODEL_PACK = os.environ.get("FACE_MODEL_PACK", "buffalo_l")

# Square size (pixels) the detector internally resizes to before running
# SCRFD -- insightface's own default. Larger finds smaller/more distant
# faces (like the old FACE_UPSAMPLE knob did for dlib) at the cost of
# detection time; must be a multiple of 32.
FACE_DET_SIZE = int(os.environ.get("FACE_DET_SIZE", "640"))

# Minimum detector confidence (0-1) for a candidate to count as a face at
# all. insightface's own default; lower catches more marginal/occluded
# faces at the cost of more false positives.
FACE_DET_THRESH = float(os.environ.get("FACE_DET_THRESH", "0.5"))

# -1 = CPU. Set to a GPU device index (0, 1, ...) if this ever runs
# somewhere with a CUDA-capable GPU and onnxruntime-gpu installed --
# ingestion of a large photo library is the case that would benefit most,
# same tradeoff the old FACE_DETECTION_MODEL=cnn knob offered for dlib.
FACE_CTX_ID = int(os.environ.get("FACE_CTX_ID", "-1"))

# buffalo_l's recognition model ("w600k_r50") always outputs 512-d
# embeddings; not independently configurable, just documented here since
# EMBEDDING_DIM (database_pg.py / .env) needs to match it.
EMBEDDING_DIM = 512

MAX_DETECTION_DIMENSION = int(os.environ.get("MAX_DETECTION_DIMENSION", "1600"))


class ImageDecodeError(ValueError):
    """Raised when the bytes can't be decoded as an image at all."""


@dataclass
class DetectedFace:
    embedding: np.ndarray  # shape (512,), float64, L2-normalized
    bounding_box: tuple[int, int, int, int]  # (top, right, bottom, left)


def _face_quality_metrics(rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> dict:
    """
    Cheap, dependency-free (no OpenCV) quality signals for the *cropped
    face region only* -- not the whole frame -- since that's the part that
    actually feeds the recognition model. Used purely for diagnostic
    logging right now (see detect_faces), to find out whether webcam
    selfie captures are systematically lower quality than the library
    photos they're compared against, before deciding on a fix (upscaling,
    a min-quality capture gate, retake guidance, etc).

    - resolution: crop's (width, height) in pixels post-downscale. A tiny
      crop (e.g. a distant/small face) has less real information for the
      encoder to work with.
    - blur_variance: variance of a simple discrete Laplacian over the
      grayscale crop. Low variance ~= few sharp edges ~= blurry/out of
      focus. No fixed "good" cutoff yet -- the point of logging this is
      to compare library-photo crops against selfie crops for the same
      person and see if there's a real, consistent gap.
    - brightness: mean grayscale pixel value (0-255). Very low means
      underexposed/dark, very high means blown out/overexposed -- both
      lose the texture detail the encoder relies on.
    """
    top, right, bottom, left = bbox
    top, left = max(top, 0), max(left, 0)
    crop = rgb[top:bottom, left:right]
    if crop.size == 0:
        return {"resolution": (0, 0), "blur_variance": 0.0, "brightness": 0.0}

    gray = crop.astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])  # luma

    # Simple discrete Laplacian (4*center - up - down - left - right),
    # computed on the interior only (no padding) to keep this dependency-free.
    if gray.shape[0] > 2 and gray.shape[1] > 2:
        center = gray[1:-1, 1:-1]
        up = gray[:-2, 1:-1]
        down = gray[2:, 1:-1]
        left_ = gray[1:-1, :-2]
        right_ = gray[1:-1, 2:]
        laplacian = 4 * center - up - down - left_ - right_
        blur_variance = float(laplacian.var())
    else:
        blur_variance = 0.0

    return {
        "resolution": (crop.shape[1], crop.shape[0]),
        "blur_variance": round(blur_variance, 1),
        "brightness": round(float(gray.mean()), 1),
    }


def _load_rgb_array(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)  # correct for EXIF orientation
        img = img.convert("RGB")
        # Detection cost scales with pixel count, and modern phone photos
        # routinely run 3000-4000px on the long edge -- several times more
        # pixels than detection actually needs to find faces reliably.
        # Downscaling first (rather than detecting at full resolution) is
        # the standard fix and cuts detection time dramatically with no
        # meaningful accuracy loss for clustering purposes. bounding_box
        # is only ever stored, never used to crop a display image (see
        # database.py/ingestion.py), so working in downscaled coordinates
        # throughout is safe.
        if max(img.size) > MAX_DETECTION_DIMENSION:
            img.thumbnail((MAX_DETECTION_DIMENSION, MAX_DETECTION_DIMENSION), Image.LANCZOS)
    except Exception as e:
        raise ImageDecodeError(f"Could not decode image: {e}") from e
    return np.array(img)


_face_app = None  # lazily built insightface.app.FaceAnalysis, cached per process


def _get_face_app():
    global _face_app
    if _face_app is not None:
        return _face_app

    from insightface.app import FaceAnalysis
    from insightface.model_zoo import model_zoo

    logger.info("Loading insightface FaceAnalysis (pack=%s, det+rec only)...", FACE_MODEL_PACK)
    t0 = time.monotonic()

    # Load only the two models we actually need — detection and recognition.
    # The full pack also loads 1k3d68 (3D landmark) and 2d106det (2D landmark)
    # which are only needed for face alignment visualisation, not for
    # embedding/clustering/matching. Skipping them saves ~100-150MB RAM.
    app = FaceAnalysis(
        name=FACE_MODEL_PACK,
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=FACE_CTX_ID, det_size=(FACE_DET_SIZE, FACE_DET_SIZE), det_thresh=FACE_DET_THRESH)
    logger.info("FaceAnalysis ready in %.1fs", time.monotonic() - t0)

    _face_app = app
    return _face_app


def detect_faces(image_bytes: bytes) -> list[DetectedFace]:
    """
    Detect faces in an image and return one DetectedFace per face found.
    Raises ImageDecodeError if the bytes aren't a decodable image; returns
    an empty list (not an error) if the image decodes fine but has no faces.

    Builds/imports insightface lazily (via _get_face_app(), rather than at
    module load) so the rest of the app -- including Phases 1 and 3, and
    even just booting the server -- doesn't require the onnxruntime native
    build or the downloaded model files. It's only needed once ingestion
    (or a selfie capture) actually runs.
    """
    logger.info("Detecting faces (pack=%s)...", FACE_MODEL_PACK)
    app = _get_face_app()

    rgb = _load_rgb_array(image_bytes)
    logger.info("Image size fed to detector: %dx%d pixels", rgb.shape[1], rgb.shape[0])

    # insightface's ONNX models (like OpenCV, which most insightface
    # examples are built around) expect BGR channel order, not RGB.
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])

    t0 = time.monotonic()
    detected = app.get(bgr)
    logger.info("FaceAnalysis.get() took %.1fs", time.monotonic() - t0)
    logger.info("Face detection finished: %d face(s) found", len(detected))

    faces = []
    for i, f in enumerate(detected):
        # insightface's native bbox order is (x1, y1, x2, y2) i.e.
        # (left, top, right, bottom). Convert to the (top, right, bottom,
        # left) convention the rest of the app already relies on
        # (database.py's stored JSON, main.py's crop-to-bbox, ingestion.py)
        # so nothing downstream needs to change for this swap.
        x1, y1, x2, y2 = (int(round(v)) for v in f.bbox)
        bbox = (y1, x2, y2, x1)

        # normed_embedding is ArcFace's L2-normalized 512-d embedding
        # (unit norm) -- the form insightface recommends comparing with
        # cosine similarity / Euclidean distance. Cast to float64 to match
        # embedding_to_blob/blob_to_embedding's serialization format.
        embedding = np.asarray(f.normed_embedding, dtype=np.float64)

        metrics = _face_quality_metrics(rgb, bbox)
        logger.info(
            "Face %d quality: resolution=%dx%d, blur_variance=%.1f, brightness=%.1f, "
            "det_score=%.3f",
            i,
            metrics["resolution"][0],
            metrics["resolution"][1],
            metrics["blur_variance"],
            metrics["brightness"],
            float(f.det_score),
        )
        faces.append(DetectedFace(embedding=embedding, bounding_box=bbox))
    return faces


def embedding_to_blob(embedding: np.ndarray) -> bytes:
    return np.asarray(embedding, dtype=np.float64).tobytes()


def blob_to_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float64)
