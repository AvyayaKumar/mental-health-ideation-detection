import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from celery.result import AsyncResult

from worker import analyze_text, celery_app

ENV = os.getenv("ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

app = FastAPI(title="Suicide Detection Bot API", version="0.1.0")

# CORS for local testing; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*" if ENV == "development" else "http://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")


class EssayRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/v1/analyze")
def submit_essay(req: EssayRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    # Queue background task
    async_result = analyze_text.delay(text)
    return {"task_id": async_result.id}


@app.get("/api/v1/result/{task_id}")
def get_result(task_id: str):
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is required")
    result: AsyncResult = celery_app.AsyncResult(task_id)
    resp = {"task_id": task_id, "status": result.status}
    if result.successful():
        resp["result"] = result.get()  # result is a JSON-serializable dict
    elif result.failed():
        # Avoid leaking internal errors
        return JSONResponse(status_code=500, content={"task_id": task_id, "status": "FAILURE"})
    return resp
