"""
DeskMate Backend — FastAPI application.
WebSocket endpoint for real-time screen analysis.
REST endpoints for answer evaluation and session summary.
"""

import json
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent.question_generator import generate_questions, clear_session_hashes
from agent.deskmate_agent import evaluate_answer
from agent.session_manager import SessionManager

app = FastAPI(
    title="DeskMate API",
    description="Proactive AI Study Agent — Backend",
    version="1.0.0",
)

# CORS — allow all origins (Tauri desktop app connects directly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared session manager instance
session_manager = SessionManager()


# ──────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "running",
        "platform": "Google Cloud Run",
        "model": "gemini-2.0-flash",
        "vertex_ai_project": os.environ.get("GCP_PROJECT_ID", "YOUR_PROJECT_ID"),
        "vertex_ai_location": os.environ.get("GCP_LOCATION", "us-central1"),
    }


# ──────────────────────────────────────────────
# WebSocket — Study Session (screenshot stream)
# ──────────────────────────────────────────────

@app.websocket("/ws/study-session")
async def study_session(websocket: WebSocket):
    await websocket.accept()

    # Get session_id from query params or generate one
    session_id = websocket.query_params.get("session_id", "default")
    session_manager.create_session(session_id)

    try:
        while True:
            # Receive screenshot data from Tauri desktop client
            raw = await websocket.receive_text()
            data = json.loads(raw)

            screenshot_b64 = data.get("screenshot", "")
            sid = data.get("session_id", session_id)

            if not screenshot_b64:
                await websocket.send_json({"error": "No screenshot data provided"})
                continue

            # Generate questions from screenshot
            result = generate_questions(screenshot_b64, sid)
            # generate_questions is async
            result = await result

            # Record questions in session if study material found
            if result.get("is_study_material") and result.get("questions"):
                topic = result.get("topic", "")
                for q in result["questions"]:
                    session_manager.record_question(sid, q, topic)

            await websocket.send_json(result)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


# ──────────────────────────────────────────────
# REST — Answer Evaluation
# ──────────────────────────────────────────────

class AnswerRequest(BaseModel):
    session_id: str
    question_id: str
    question: str
    correct_answer: str
    user_answer: str
    topic: str = ""


class AnswerResponse(BaseModel):
    is_correct: bool
    score: int
    feedback: str
    hint_for_improvement: str = ""


@app.post("/api/answer", response_model=AnswerResponse)
async def submit_answer(req: AnswerRequest):
    # Call Gemini to evaluate
    evaluation = await evaluate_answer(
        question=req.question,
        correct_answer=req.correct_answer,
        user_answer=req.user_answer,
        topic=req.topic,
    )

    # Record in session
    session_manager.record_answer(
        session_id=req.session_id,
        question_id=req.question_id,
        user_answer=req.user_answer,
        is_correct=evaluation.get("is_correct", False),
        score=evaluation.get("score", 0),
        feedback=evaluation.get("feedback", ""),
        topic=req.topic,
    )

    return AnswerResponse(
        is_correct=evaluation.get("is_correct", False),
        score=evaluation.get("score", 0),
        feedback=evaluation.get("feedback", ""),
        hint_for_improvement=evaluation.get("hint_for_improvement", ""),
    )


# ──────────────────────────────────────────────
# REST — Session Summary
# ──────────────────────────────────────────────

@app.get("/api/summary/{session_id}")
async def get_summary(session_id: str):
    summary = session_manager.get_summary(session_id)
    if "error" in summary:
        return {"error": summary["error"]}, 404
    return summary


# ──────────────────────────────────────────────
# REST — End Session (cleanup)
# ──────────────────────────────────────────────

@app.post("/api/end-session/{session_id}")
async def end_session(session_id: str):
    summary = session_manager.get_summary(session_id)
    # Clean up
    clear_session_hashes(session_id)
    session_manager.delete_session(session_id)
    return {"message": "Session ended", "summary": summary}
