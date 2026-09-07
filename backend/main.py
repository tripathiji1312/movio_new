import uvicorn
from .tts_engine import engine
from .server import app
from .config import PORT


def main():
    engine.load()
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
