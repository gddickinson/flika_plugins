"""Start the FLIKA plugins web catalogue on http://127.0.0.1:8198 (local only). Needs requirements-web.txt."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("web.app:app", host="127.0.0.1", port=8198)
