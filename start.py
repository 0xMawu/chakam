import os
from app.database import init_db
from app.face_processing import _get_face_app

if __name__ == "__main__":
    import uvicorn

    init_db()
    print("Preloading model...")
    _get_face_app()
    print("Model ready.")

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
    )