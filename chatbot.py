"""Conversation logic for the TalentScout AI Hiring Assistant."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Dict, List, Optional, Tuple

from llm import GroqLLM
from prompts import (
    build_assessment_summary_prompt,
    build_info_extraction_prompt,
    build_post_screening_prompt,
    build_question_generation_prompt,
    build_tech_parsing_prompt,
)
from utils import (
    has_alpha_content,
    is_exit_intent,
    normalize_text,
    parse_desired_positions,
    parse_tech_stack,
    validate_email,
    validate_experience,
    validate_phone,
)


FALLBACK_MESSAGE = "Sorry, I didn't understand that. Could you rephrase?"
EXIT_MESSAGE = "Thank you! Our team will review your responses."

FIELD_SEQUENCE = [
    "full_name",
    "email",
    "phone_number",
    "years_of_experience",
    "desired_positions",
    "current_location",
    "tech_stack",
]

FIELD_QUESTIONS = {
    "full_name": "Please share your full name.",
    "email": "What is your email address?",
    "phone_number": "What is your phone number?",
    "years_of_experience": "How many years of professional experience do you have?",
    "desired_positions": "Which desired position(s) are you applying for? (You can list multiple, separated by commas)",
    "current_location": "What is your current location?",
    "tech_stack": "Please list your tech stack (for example: Python, Django, React).",
}

SMALL_TALK_INPUTS = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
}

QUESTION_PREFIXES = (
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "can",
    "could",
    "would",
    "should",
    "do",
    "does",
    "is",
    "are",
)

START_ASSESSMENT_KEYWORDS = {
    "start assessment",
    "begin assessment",
    "start technical round",
    "start questions",
}

SKIP_ASSESSMENT_KEYWORDS = {
    "skip assessment",
    "not now",
    "later",
}

SKIP_QUESTION_KEYWORDS = {
    "skip",
    "pass",
    "next",
}


@dataclass
class CandidateProfile:
    """Represents candidate screening information collected during chat."""

    full_name: Optional[str] = None
    email: Optional[str] = None
    phone_number: Optional[str] = None
    years_of_experience: Optional[float] = None
    desired_positions: List[str] = field(default_factory=list)
    current_location: Optional[str] = None
    tech_stack: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        """Serialize profile information to a dictionary."""
        return asdict(self)


class TalentScoutChatbot:
    """Stateful chatbot that collects candidate info and generates interview questions."""

    def __init__(self, llm: Optional[GroqLLM] = None) -> None:
        self.llm = llm or GroqLLM()
        self.profile = CandidateProfile()
        self.current_field_index = 0
        self.started = False
        self.finished = False
        self.technical_questions: Dict[str, List[str]] = {}
        self.assessment_queue: List[Dict[str, str]] = []
        self.assessment_answers: List[Dict[str, str]] = []
        self.assessment_active = False
        self.assessment_index = 0
        self.assessment_summary: Optional[str] = None

    def start_conversation(self) -> str:
        """Start the chat flow and ask the first screening question."""
        self.started = True
        greeting = (
            "Hello! I am TalentScout, your AI Hiring Assistant. "
            "I will collect your details and then generate technical screening questions "
            "based on your tech stack."
        )
        return f"{greeting}\n\n{self._current_question()}"

    def handle_message(self, user_message: str) -> str:
        """Handle a user message while maintaining a deterministic flow."""
        normalized = normalize_text(user_message)

        if not normalized:
            return FALLBACK_MESSAGE

        if is_exit_intent(normalized):
            self.finished = True
            return EXIT_MESSAGE

        if not self.started:
            return self.start_conversation()

        if self.finished:
            return self._handle_post_screening(normalized)

        # Always treat question-style input as a clarification request, not field data.
        if self._is_user_question(normalized):
            answer = self._answer_candidate_question(normalized)
            return f"{answer}\n\n{self._current_question()}"

        field_name = FIELD_SEQUENCE[self.current_field_index]
        extracted_value = self._extract_field_value(field_name=field_name, user_message=normalized)
        is_valid, normalized_value, error_message = self._validate_field(field_name, extracted_value)

        if not is_valid:
            if self._is_small_talk(normalized):
                return f"Hello! Let us continue.\n\n{self._current_question()}"

            if error_message:
                return f"{error_message}\n\n{self._current_question()}"
            return FALLBACK_MESSAGE

        self._store_field(field_name, normalized_value)

        self.current_field_index += 1

        if self.current_field_index < len(FIELD_SEQUENCE):
            return self._current_question()

        self.finished = True
        return self._complete_screening()

    def get_profile_data(self) -> Dict[str, object]:
        """Return current in-memory profile data."""
        return self.profile.to_dict()

    def get_question_bank(self) -> Dict[str, List[str]]:
        """Return generated technical questions by technology."""
        return self.technical_questions

    def get_assessment_report(self) -> Dict[str, object]:
        """Return structured technical assessment information."""
        return {
            "queued": len(self.assessment_queue),
            "answered": len(self.assessment_answers),
            "active": self.assessment_active,
            "summary": self.assessment_summary,
            "answers": self.assessment_answers,
        }

    def get_progress(self) -> Dict[str, object]:
        """Return progress details for UI status panels."""
        total = len(FIELD_SEQUENCE)
        completed = total if self.finished else min(self.current_field_index, total)
        percent = int((completed / total) * 100)
        next_field = None if self.finished else FIELD_SEQUENCE[self.current_field_index]
        return {
            "completed": completed,
            "total": total,
            "percent": percent,
            "next_field": next_field,
            "finished": self.finished,
        }

    def _current_question(self) -> str:
        field_name = FIELD_SEQUENCE[self.current_field_index]
        return FIELD_QUESTIONS[field_name]

    def _extract_field_value(self, field_name: str, user_message: str) -> str:
        """Extract the field value with LLM support and fallback to raw input."""
        prompt = build_info_extraction_prompt(field_name=field_name, user_message=user_message)

        if not self.llm.is_available:
            return user_message

        extracted = self.llm.safe_generate_response(
            prompt=prompt,
            default_response=user_message,
            system_prompt="You extract one field value from candidate responses.",
        )
        cleaned = normalize_text(extracted).strip("\"'")

        if not cleaned or cleaned.upper() == "UNKNOWN":
            return user_message

        return cleaned

    def _validate_field(self, field_name: str, value: str) -> Tuple[bool, object, Optional[str]]:
        """Validate and normalize each field value before storing it."""
        if field_name == "full_name":
            words = [part for part in re.split(r"\s+", value) if part]
            if len(value) < 3 or len(words) < 2:
                return False, None, "Please provide a valid full name."
            if any(token.casefold() in SMALL_TALK_INPUTS for token in words):
                return False, None, "Please provide your full name (first and last name)."
            return True, value, None

        if field_name == "email":
            if not validate_email(value):
                return False, None, "Please enter a valid email address."
            return True, value, None

        if field_name == "phone_number":
            if not validate_phone(value):
                return False, None, "Please enter a valid phone number with country code if available."
            return True, value, None

        if field_name == "years_of_experience":
            valid, years = validate_experience(value)
            if not valid or years is None:
                return False, None, "Please provide years of experience as a number (for example: 4 or 6.5)."
            return True, years, None

        if field_name == "desired_positions":
            positions = parse_desired_positions(value)
            if not positions:
                return False, None, "Please provide at least one valid desired position."
            return True, positions, None

        if field_name == "current_location":
            if len(value) < 2 or not has_alpha_content(value):
                return False, None, "Please provide your current location."
            return True, value, None

        if field_name == "tech_stack":
            technologies = self._parse_technologies(value)
            if not technologies:
                return False, None, "Please provide at least one valid technology (for example: Python, Django, React)."
            return True, technologies, None

        return False, None, FALLBACK_MESSAGE

    def _store_field(self, field_name: str, value: object) -> None:
        """Store normalized values in the in-memory profile."""
        setattr(self.profile, field_name, value)

    def _parse_technologies(self, raw_input: str) -> List[str]:
        """Parse and normalize technology names using LLM-first parsing."""
        if self.llm.is_available:
            parsed_text = self.llm.safe_generate_response(
                prompt=build_tech_parsing_prompt(raw_input),
                default_response=raw_input,
                system_prompt="You normalize technical skills into CSV text.",
            )
            parsed_list = parse_tech_stack(parsed_text)
            if parsed_list and parsed_list[0].upper() != "UNKNOWN":
                return parsed_list

        return parse_tech_stack(raw_input)

    def _is_small_talk(self, user_input: str) -> bool:
        """Check if the user input is likely greeting/small talk rather than a field value."""
        return normalize_text(user_input).casefold() in SMALL_TALK_INPUTS

    def _is_user_question(self, user_input: str) -> bool:
        """Check if the user message appears to be a question."""
        normalized = normalize_text(user_input).casefold()
        if "?" in normalized:
            return True

        # Reduce false positives by requiring question-prefix and enough words.
        words = normalized.split()
        if len(words) < 3:
            return False

        return any(normalized.startswith(f"{prefix} ") for prefix in QUESTION_PREFIXES)

    def _answer_candidate_question(self, user_question: str) -> str:
        """Answer candidate questions while preserving the current screening step."""
        pending_prompt = self._current_question()
        default_answer = (
            "I am here to run a quick pre-screening and tailor technical questions to your skills. "
            "Your responses help us prepare a focused interview."
        )

        if not self.llm.is_available:
            return default_answer

        prompt = (
            "You are TalentScout, an AI hiring assistant. "
            "Answer the candidate question in 1-2 concise, friendly sentences. "
            "Do not ask additional questions.\n\n"
            f"Candidate question: {user_question}\n"
            f"Current pending detail to collect next: {pending_prompt}"
        )
        return self.llm.safe_generate_response(
            prompt=prompt,
            default_response=default_answer,
            system_prompt="You are a concise, professional hiring assistant.",
        )

    def _complete_screening(self) -> str:
        """Generate interview questions for each technology and return a final response."""
        self.technical_questions = self._generate_questions_for_stack(self.profile.tech_stack)
        self.assessment_queue = self._build_assessment_queue(self.technical_questions)
        self.assessment_answers = []
        self.assessment_active = False
        self.assessment_index = 0
        self.assessment_summary = None

        lines = [
            (
                "Great, your profile is complete. I created tailored screening questions "
                "based on your experience and tech stack:"
            ),
            "",
        ]

        for tech, questions in self.technical_questions.items():
            lines.append(f"{tech}:")
            for index, question in enumerate(questions, start=1):
                lines.append(f"{index}. {question}")
            lines.append("")

        lines.append(
            "If you want to continue, type 'start assessment' and I will collect your answers "
            "to a focused technical round."
        )
        lines.append("You can type 'exit', 'quit', or 'bye' whenever you want to end the conversation.")
        lines.append("You can also ask for deeper practice questions on any technology.")
        return "\n".join(lines).strip()

    def _generate_questions_for_stack(self, technologies: List[str]) -> Dict[str, List[str]]:
        """Generate 3 to 5 technical questions per technology."""
        question_bank: Dict[str, List[str]] = {}
        profile_context = self.profile.to_dict()
        target_count = self._target_question_count()

        for technology in technologies:
            prompt = build_question_generation_prompt(
                technology=technology,
                question_count=target_count,
                candidate_context=profile_context,
            )
            default_text = self._default_questions_for_technology(technology)

            if self.llm.is_available:
                response = self.llm.safe_generate_response(
                    prompt=prompt,
                    default_response=default_text,
                    system_prompt="You are a technical interviewer generating concise, practical questions.",
                )
            else:
                response = default_text

            parsed_questions = self._parse_numbered_questions(response)
            if len(parsed_questions) < 3:
                parsed_questions = self._parse_numbered_questions(default_text)

            completed_questions = self._ensure_question_count(
                technology=technology,
                questions=parsed_questions,
                target_count=target_count,
            )

            question_bank[technology] = completed_questions[:5]

        return question_bank

    def _target_question_count(self) -> int:
        """Choose question count between 3 and 5 based on candidate experience."""
        years = self.profile.years_of_experience or 0
        if years >= 6:
            return 5
        if years >= 2:
            return 4
        return 3

    def _ensure_question_count(self, technology: str, questions: List[str], target_count: int) -> List[str]:
        """Ensure each technology has exactly 3-5 high-quality questions."""
        unique: List[str] = []
        seen = set()

        for question in questions:
            key = question.casefold().strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(question)

        if len(unique) >= target_count:
            return unique[:target_count]

        for fallback in self._fallback_question_pool_for_technology(technology):
            key = fallback.casefold().strip()
            if key in seen:
                continue
            seen.add(key)
            unique.append(fallback)
            if len(unique) >= target_count:
                break

        return unique[:target_count]

    @staticmethod
    def _parse_numbered_questions(text: str) -> List[str]:
        """Extract question lines from numbered or bulleted LLM output."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        questions: List[str] = []

        for line in lines:
            cleaned = line
            if ". " in line[:4]:
                cleaned = line.split(". ", 1)[1].strip()
            elif line.startswith("-"):
                cleaned = line.lstrip("- ").strip()
            elif line.startswith("*"):
                cleaned = line.lstrip("* ").strip()

            if not cleaned:
                continue

            cleaned = cleaned.strip("- ").strip()
            cleaned = re.sub(r"\s+", " ", cleaned)

            if len(cleaned) < 12:
                continue

            if cleaned.endswith("?"):
                questions.append(cleaned)
                continue

            if re.match(r"^(How|What|Why|When|Where|Which|Explain|Describe|Walk me through)\b", cleaned, re.I):
                cleaned = cleaned.rstrip(".")
                if not cleaned.endswith("?"):
                    cleaned = f"{cleaned}?"
                questions.append(cleaned)

        # Preserve unique order.
        unique_questions: List[str] = []
        seen = set()
        for question in questions:
            key = question.casefold()
            if key not in seen:
                seen.add(key)
                unique_questions.append(question)

        return unique_questions

    @staticmethod
    def _default_questions_for_technology(technology: str) -> str:
        """Return a deterministic fallback set when the LLM is unavailable."""
        return (
            f"1. How have you used {technology} in a real project, and what trade-offs did you face?\n"
            f"2. What are common performance or scalability issues in {technology}, and how do you address them?\n"
            f"3. How do you test and debug applications built with {technology}?"
        )

    @staticmethod
    def _fallback_question_pool_for_technology(technology: str) -> List[str]:
        """Return a diversified fallback pool to keep questions tailored and non-repetitive."""
        return [
            f"How have you used {technology} in a real project, and what trade-offs did you face?",
            f"What are common performance or scalability issues in {technology}, and how do you address them?",
            f"How do you test and debug applications built with {technology}?",
            f"Explain an architecture decision you made using {technology} and why you chose that approach?",
            f"How would you secure a production workload built with {technology}?",
            f"Describe a difficult bug you resolved in {technology} and the diagnostic steps you followed?",
            f"What coding or design patterns in {technology} do you rely on for maintainability?",
            f"How would you evaluate and improve reliability for a {technology}-based system under high load?",
        ]

    def _build_assessment_queue(self, question_bank: Dict[str, List[str]]) -> List[Dict[str, str]]:
        """Create a focused queue of questions for answer collection."""
        queue: List[Dict[str, str]] = []
        for technology, questions in question_bank.items():
            for question in questions[:2]:
                queue.append({"technology": technology, "question": question})
        return queue

    def _handle_post_screening(self, user_message: str) -> str:
        """Handle post-screening behavior including optional assessment mode."""
        normalized = user_message.casefold()

        if self.assessment_active:
            return self._capture_assessment_answer(user_message)

        if normalized in START_ASSESSMENT_KEYWORDS:
            if not self.assessment_queue:
                return "No assessment questions are queued yet. Please request technical questions first."

            self.assessment_active = True
            self.assessment_index = 0
            return self._next_assessment_prompt(opening=True)

        if normalized in SKIP_ASSESSMENT_KEYWORDS:
            return "No problem. You can type 'start assessment' anytime to continue with answer collection."

        return self._handle_post_screening_message(user_message)

    def _next_assessment_prompt(self, opening: bool = False) -> str:
        """Return the next question prompt during assessment mode."""
        total = len(self.assessment_queue)
        if self.assessment_index >= total:
            return self._finish_assessment_round()

        current = self.assessment_queue[self.assessment_index]
        prefix = "Let us start the focused technical round.\n\n" if opening else ""
        return (
            f"{prefix}Question {self.assessment_index + 1} of {total} "
            f"[{current['technology']}]:\n{current['question']}\n\n"
            "You can answer directly, or type 'skip' to move to the next question."
        )

    def _capture_assessment_answer(self, user_message: str) -> str:
        """Capture candidate answers during assessment mode and move to next question."""
        if self.assessment_index >= len(self.assessment_queue):
            return self._finish_assessment_round()

        current = self.assessment_queue[self.assessment_index]
        normalized = normalize_text(user_message).casefold()
        answer = "Skipped by candidate." if normalized in SKIP_QUESTION_KEYWORDS else user_message

        self.assessment_answers.append(
            {
                "technology": current["technology"],
                "question": current["question"],
                "answer": answer,
            }
        )

        self.assessment_index += 1

        if self.assessment_index < len(self.assessment_queue):
            return self._next_assessment_prompt(opening=False)

        return self._finish_assessment_round()

    def _finish_assessment_round(self) -> str:
        """Close assessment mode and return recruiter-style summary."""
        self.assessment_active = False
        self.assessment_summary = self._generate_assessment_summary()
        return (
            "Technical round completed. Here is your screening summary:\n\n"
            f"{self.assessment_summary}\n\n"
            "You can still ask for more role-specific practice questions."
        )

    def _generate_assessment_summary(self) -> str:
        """Generate a concise recruiter-ready summary from captured answers."""
        if not self.assessment_answers:
            return (
                "Strengths:\n- Candidate completed profile details.\n\n"
                "Risks:\n- No technical answers captured yet.\n\n"
                "Recommended Next Round Focus:\n- Conduct a guided coding discussion.\n\n"
                "Confidence Score: 40"
            )

        fallback = self._heuristic_assessment_summary()
        if not self.llm.is_available:
            return fallback

        prompt = build_assessment_summary_prompt(
            profile=self.profile.to_dict(),
            qa_pairs=self.assessment_answers,
        )
        return self.llm.safe_generate_response(
            prompt=prompt,
            default_response=fallback,
            system_prompt="You are a concise technical hiring evaluator.",
        )

    def _heuristic_assessment_summary(self) -> str:
        """Fallback summary when LLM is not available."""
        meaningful_answers = [
            item for item in self.assessment_answers if len(normalize_text(item["answer"])) >= 20
        ]
        score = min(90, 45 + (len(meaningful_answers) * 8))

        return (
            "Strengths:\n"
            f"- Answered {len(self.assessment_answers)} technical questions.\n"
            f"- Demonstrated depth in {len(set(item['technology'] for item in meaningful_answers))} technology areas.\n\n"
            "Risks:\n"
            "- Some answers may need deeper implementation-level detail.\n\n"
            "Recommended Next Round Focus:\n"
            "- Scenario-based coding and architecture follow-up for declared stack.\n\n"
            f"Confidence Score: {score}"
        )

    def _handle_post_screening_message(self, user_message: str) -> str:
        """Handle post-screening conversational replies instead of forcing exit."""
        if self._is_out_of_scope(user_message):
            return (
                "I can help with hiring-related support only, such as interview preparation, "
                "role-fit guidance, or additional technical practice questions."
            )

        default_response = (
            "Glad to hear that. If you want, I can generate deeper follow-up questions "
            "for any technology in your stack."
        )

        if not self.llm.is_available:
            return default_response

        prompt = build_post_screening_prompt(
            user_message=user_message,
            profile=self.profile.to_dict(),
            technologies=self.profile.tech_stack,
        )

        return self.llm.safe_generate_response(
            prompt=prompt,
            default_response=default_response,
            system_prompt="You are a thoughtful technical hiring assistant.",
        )

    def _is_out_of_scope(self, user_message: str) -> bool:
        """Identify prompts unrelated to hiring flow to keep conversation on-purpose."""
        text = normalize_text(user_message).casefold()
        if not text:
            return False

        hiring_keywords = {
            "interview",
            "role",
            "job",
            "position",
            "question",
            "assessment",
            "tech",
            "stack",
            "screening",
            "candidate",
        }
        out_of_scope_keywords = {
            "poem",
            "song",
            "movie",
            "weather",
            "recipe",
            "travel",
            "joke",
            "politics",
            "news",
        }

        has_hiring_term = any(word in text for word in hiring_keywords)
        has_out_of_scope_term = any(word in text for word in out_of_scope_keywords)
        return has_out_of_scope_term and not has_hiring_term
