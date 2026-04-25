"""Prompt templates for the TalentScout AI Hiring Assistant."""

from textwrap import dedent
from typing import Dict, List


def build_info_extraction_prompt(field_name: str, user_message: str) -> str:
    """Build a strict prompt to extract a single field value from user input."""
    return dedent(
        f"""
        You are an information extraction engine for an AI hiring assistant.
        Extract only the value for the requested field.

        Requested field: {field_name}
        User message: {user_message}

        Rules:
        - Return only the extracted value.
        - Do not add explanations.
        - If no value can be extracted, return: UNKNOWN
        """
    ).strip()


def build_tech_parsing_prompt(raw_tech_input: str) -> str:
    """Build a prompt to normalize raw tech stack text into CSV values."""
    return dedent(
        f"""
        You are parsing technical skills from candidate input.
        Convert the input into a clean comma-separated list of technologies.

        Input: {raw_tech_input}

        Rules:
        - Keep only technology names.
        - Remove duplicates.
        - Preserve common capitalization (for example: Python, React, PostgreSQL).
        - Return only CSV text.
        - If unclear, return: UNKNOWN
        """
    ).strip()


def build_question_generation_prompt(
    technology: str,
    question_count: int = 4,
    candidate_context: Dict[str, object] | None = None,
) -> str:
    """Build a prompt to generate interview questions for one technology."""
    context_lines = ""
    if candidate_context:
        desired_positions = candidate_context.get("desired_positions", [])
        if isinstance(desired_positions, list):
            desired_positions_text = ", ".join(desired_positions) if desired_positions else "Not provided"
        else:
            desired_positions_text = str(desired_positions) or "Not provided"

        context_lines = dedent(
            f"""
            Candidate profile context:
            - Desired positions: {desired_positions_text}
            - Experience: {candidate_context.get('years_of_experience', 'Not provided')} years
            - Location: {candidate_context.get('current_location', 'Not provided')}
            """
        ).strip()

    return dedent(
        f"""
        You are a senior technical interviewer.
        Generate {question_count} intermediate-level interview questions for the technology below.

        Technology: {technology}
        {context_lines}

        Rules:
        - Questions should evaluate practical understanding.
        - Avoid yes/no questions.
        - Keep each question concise.
        - Return as a numbered list only.
        - Every line must be a direct question.
        - Keep difficulty at intermediate level.
        - Tailor questions to the specific technology and candidate context.
        """
    ).strip()


def build_post_screening_prompt(
    user_message: str,
    profile: Dict[str, object],
    technologies: List[str],
) -> str:
    """Build a prompt for conversational follow-ups after screening is complete."""
    return dedent(
        f"""
        You are TalentScout, an AI hiring assistant.
        The screening form is already complete, and now the candidate is chatting.

        Candidate profile:
        - Name: {profile.get('full_name', 'Not provided')}
        - Positions: {profile.get('desired_positions', [])}
        - Experience: {profile.get('years_of_experience', 'Not provided')} years
        - Location: {profile.get('current_location', 'Not provided')}
        - Tech stack: {', '.join(technologies) if technologies else 'Not provided'}

        Candidate message:
        {user_message}

        Response rules:
        - Reply in 2-4 lines.
        - Sound human, warm, and professional.
        - If candidate asks for more practice, offer one practical follow-up question.
        - Do not ask for personal sensitive data.
        - Stay within hiring, screening, interview preparation, or role-fit topics.
        - If user asks unrelated questions, politely refuse and redirect to screening support.
        """
    ).strip()


def build_assessment_summary_prompt(
    profile: Dict[str, object],
    qa_pairs: List[Dict[str, str]],
) -> str:
    """Build a prompt that generates a recruiter-friendly summary from Q/A pairs."""
    joined_pairs = "\n".join(
        [
            f"- [{item.get('technology', 'General')}] Q: {item.get('question', '')} | A: {item.get('answer', '')}"
            for item in qa_pairs
        ]
    )

    return dedent(
        f"""
        You are an AI hiring analyst preparing a concise screening summary.

        Candidate profile:
        - Name: {profile.get('full_name', 'Not provided')}
        - Desired positions: {profile.get('desired_positions', [])}
        - Experience: {profile.get('years_of_experience', 'Not provided')} years
        - Tech stack: {profile.get('tech_stack', [])}

        Candidate technical responses:
        {joined_pairs if joined_pairs else 'No answers provided.'}

        Output format (exact headings):
        Strengths:
        - ...

        Risks:
        - ...

        Recommended Next Round Focus:
        - ...

        Confidence Score: <0-100>

        Rules:
        - Keep it factual and concise.
        - Do not invent credentials not present in responses.
        - If data is insufficient, clearly say so.
        """
    ).strip()
