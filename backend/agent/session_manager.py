"""
Session Manager — In-memory session tracking for DeskMate.
Tracks questions asked, answers given, scores, and topics per session.
"""

import time
from collections import defaultdict
from typing import Optional


class SessionData:
    """Data container for a single study session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = time.time()
        self.topics: list[str] = []
        self.questions: list[dict] = []
        self.answers: list[dict] = []
        self.total_questions = 0
        self.correct_answers = 0

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "topics": list(set(self.topics)),
            "total_questions": self.total_questions,
            "correct_answers": self.correct_answers,
            "score_percent": round(
                (self.correct_answers / self.total_questions * 100)
                if self.total_questions > 0
                else 0,
                1,
            ),
        }


class SessionManager:
    """Manages in-memory study sessions."""

    def __init__(self):
        self._sessions: dict[str, SessionData] = {}

    def create_session(self, session_id: str) -> SessionData:
        """Create a new session or return existing one."""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionData(session_id)
        return self._sessions[session_id]

    def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def record_question(self, session_id: str, question_data: dict, topic: str) -> None:
        """Record that a question was asked in a session."""
        session = self.create_session(session_id)
        session.questions.append(question_data)
        session.total_questions += 1
        if topic and topic not in session.topics:
            session.topics.append(topic)

    def record_answer(
        self,
        session_id: str,
        question_id: str,
        user_answer: str,
        is_correct: bool,
        score: int,
        feedback: str,
        topic: str = "",
    ) -> None:
        """Record a student's answer and evaluation result."""
        session = self.create_session(session_id)
        session.answers.append(
            {
                "question_id": question_id,
                "user_answer": user_answer,
                "is_correct": is_correct,
                "score": score,
                "feedback": feedback,
                "topic": topic,
            }
        )
        if is_correct:
            session.correct_answers += 1

    def get_summary(self, session_id: str) -> dict:
        """
        Get a full session summary including topics, score, and weak areas.
        """
        session = self.get_session(session_id)
        if not session:
            return {"error": "Session not found"}

        # Calculate weak areas — topics where the student got answers wrong
        topic_scores: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
        for answer in session.answers:
            t = answer.get("topic", "General")
            topic_scores[t]["total"] += 1
            if answer.get("is_correct"):
                topic_scores[t]["correct"] += 1

        weak_areas = []
        for topic, scores in topic_scores.items():
            if scores["total"] > 0:
                pct = scores["correct"] / scores["total"] * 100
                if pct < 70:
                    weak_areas.append(
                        {
                            "topic": topic,
                            "score_percent": round(pct, 1),
                            "questions_attempted": scores["total"],
                        }
                    )

        # Sort weak areas by score ascending (weakest first)
        weak_areas.sort(key=lambda x: x["score_percent"])

        summary = session.to_dict()
        summary["weak_areas"] = weak_areas
        summary["duration_seconds"] = round(time.time() - session.created_at)

        return summary

    def delete_session(self, session_id: str) -> bool:
        """Remove a session from memory."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
