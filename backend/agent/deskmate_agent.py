"""
DeskMate Agent — Core logic for screen analysis and answer evaluation.
Uses Google GenAI SDK with Vertex AI (gemini-2.0-flash).
"""

import json
import os
import base64
from google import genai
from google.genai import types

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "YOUR_PROJECT_ID")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION
)

MODEL = "gemini-2.0-flash"

SYSTEM_PROMPT = """You are DeskMate, a proactive AI study coach. You observe a student's screen and help them learn actively through retrieval practice.

When given a screenshot:
1. Determine if screen shows study material (slides, PDFs, textbooks, notes, code tutorials, educational videos). If NOT, return {"is_study_material": false}.
2. If IS study material, identify: main topic, key concepts, difficulty (beginner/intermediate/advanced).
3. Generate 1-3 practice questions based ONLY on what's visually on screen. Test understanding, not just recall.
   Types: multiple_choice (4 options), short_answer, explanation ("explain in your own words")

Always respond ONLY in valid JSON:
{
  "is_study_material": true,
  "topic": "string",
  "concepts": ["string"],
  "difficulty": "beginner|intermediate|advanced",
  "questions": [
    {
      "id": "q1",
      "type": "multiple_choice|short_answer|explanation",
      "question": "string",
      "choices": ["A. ...", "B. ...", "C. ...", "D. ..."],
      "correct_answer": "string",
      "explanation": "string",
      "focus_area": {
        "description": "brief text describing what to highlight",
        "region": "top-left|top-center|top-right|middle-left|middle-center|middle-right|bottom-left|bottom-center|bottom-right"
      }
    }
  ]
}

For short_answer and explanation types, omit "choices".
If the screen does NOT show study material, respond with:
{"is_study_material": false}
"""


async def analyze_screen_content(image_base64: str) -> dict:
    """
    Analyze a screenshot and generate study questions if study material is detected.
    
    Args:
        image_base64: Base64-encoded JPEG screenshot string.
        
    Returns:
        dict with question data or {"is_study_material": false}.
    """
    try:
        image_bytes = base64.b64decode(image_base64)

        response = client.models.generate_content(
            model=MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        types.Part.from_text("Analyze this screenshot and generate practice questions if it contains study material."),
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.7,
                max_output_tokens=2048,
            ),
        )

        # Parse the JSON response
        response_text = response.text.strip()
        # Handle markdown code blocks if returned
        if response_text.startswith("```"):
            response_text = response_text.split("\n", 1)[1]
            response_text = response_text.rsplit("```", 1)[0].strip()

        return json.loads(response_text)

    except json.JSONDecodeError:
        return {"is_study_material": False, "error": "Failed to parse AI response"}
    except Exception as e:
        return {"is_study_material": False, "error": str(e)}


async def evaluate_answer(question: str, correct_answer: str, user_answer: str, topic: str) -> dict:
    """
    Use Gemini to evaluate a student's answer.
    
    Returns:
        dict with is_correct, score, feedback, hint_for_improvement.
    """
    prompt = f"""
Topic: {topic}
Question: {question}
Correct Answer: {correct_answer}
Student Answer: {user_answer}

Evaluate encouragingly but honestly. Return ONLY valid JSON:
{{"is_correct": bool, "score": 0-100, "feedback": "2-3 sentences", "hint_for_improvement": "specific tip if wrong"}}
"""

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.5,
                max_output_tokens=512,
            ),
        )

        response_text = response.text.strip()
        if response_text.startswith("```"):
            response_text = response_text.split("\n", 1)[1]
            response_text = response_text.rsplit("```", 1)[0].strip()

        return json.loads(response_text)

    except json.JSONDecodeError:
        return {
            "is_correct": False,
            "score": 0,
            "feedback": "Sorry, I couldn't evaluate your answer. Please try again.",
            "hint_for_improvement": ""
        }
    except Exception as e:
        return {
            "is_correct": False,
            "score": 0,
            "feedback": f"Evaluation error: {str(e)}",
            "hint_for_improvement": ""
        }
