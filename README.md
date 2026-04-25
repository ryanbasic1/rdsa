# TalentScout AI Hiring Assistant

TalentScout is an AI-powered hiring assistant built with Streamlit and Groq LLM APIs. It performs an initial screening conversation, collects structured candidate details, generates tailored technical questions from the declared tech stack, and supports a focused answer-collection round with recruiter-style summary output.

## Project Overview

- Domain: Technology recruitment pre-screening
- UI: Streamlit chat interface with progress dashboard
- LLM Provider: Groq
- Default Model: openai/gpt-oss-120b
- Data Policy: Session-only in-memory storage with explicit user consent

## Features

- Clean, interactive chat interface with progress indicators
- Consent gate before data collection begins
- Structured collection of:
   - Full Name
   - Email Address
   - Phone Number
   - Years of Experience
   - Desired Position(s)
   - Current Location
   - Tech Stack
- Input validation for contact and profile fields
- Tech stack normalization and de-duplication
- AI-generated technical questions (3-5 per technology)
- Optional focused assessment mode to capture candidate answers
- Recruiter-style summary (strengths, risks, next-round focus, confidence score)
- Fallback responses for unexpected input
- Exit intent handling with graceful close
- Masked PII preview in sidebar
- Session clear with confirmation
- Basic sentiment indicator (optional enhancement)

## Architecture and Technical Decisions

- app.py
   - Streamlit UI layer
   - Consent handling, state management, rendering, and action controls
- chatbot.py
   - Stateful conversation engine
   - Field-wise flow control, validation routing, context retention
   - Technical question generation and answer collection
- llm.py
   - Groq API wrapper
   - Model configuration via environment variables
   - Decommissioned-model fallback handling and status diagnostics
- prompts.py
   - Prompt templates for extraction, parsing, question generation, post-screening chat, and summary generation
- utils.py
   - Validation, parsing, masking, sentiment helper utilities

Design rationale:

- Separation of concerns keeps UI, logic, and model integration independent and testable.
- Chatbot logic is deterministic for data collection, then adaptive for follow-up interaction.
- Safe fallback behavior ensures graceful degradation if API calls fail.

## Project Structure

- app.py
- chatbot.py
- llm.py
- prompts.py
- utils.py
- requirements.txt
- .env.example

## Installation Instructions

1. Create and activate a Python environment.

2. Install dependencies:

pip install -r requirements.txt

3. Create a .env file in project root:

GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
GROQ_MODEL_FALLBACK=openai/gpt-oss-120b

Important:

- Never hardcode API keys.
- Keep .env out of version control.

## Usage Guide

1. Start the app:

streamlit run app.py

2. In the UI:

- Confirm consent to begin screening.
- Provide requested profile fields step-by-step.
- Review generated technical questions.
- Type start assessment to enter focused answer collection mode.
- Optionally continue with follow-up interview-prep chat.
- Use exit, quit, or bye to end the session.

## Chatbot Functional Flow

1. Greeting and purpose clarification
2. Consent confirmation
3. Candidate profile collection and validation
4. Tech stack normalization
5. Technical question generation (3-5 per technology)
6. Optional assessment answer collection
7. Recruiter-ready summary generation
8. Graceful exit and session controls

## Prompt Design Strategy

Prompt families in prompts.py:

1. Info extraction prompt
- Extracts one target field from noisy user message
- Returns only field value or UNKNOWN

2. Tech parsing prompt
- Converts free text stack into clean comma-separated technologies
- Removes duplicates and noise

3. Question generation prompt
- Generates intermediate, practical questions
- Uses profile context (role, experience, location) for relevance

4. Post-screening prompt
- Keeps interactions context-aware after form completion
- Constrains assistant behavior to hiring and interview-prep domain

5. Assessment summary prompt
- Produces concise recruiter digest from captured answers

## Data Handling and Privacy

- Consent is required before collecting candidate details.
- Candidate data is kept only in Streamlit session memory.
- No disk/database persistence is used in this implementation.
- Sidebar snapshot masks sensitive fields (email, phone, name).
- Clear Session Data action requires explicit confirmation.
- This is a simulated screening tool and uses anonymized/demo operation by default.

## Challenges and Solutions

- Model decommissioning issue:
   - Challenge: legacy model IDs can fail at runtime.
   - Solution: environment-configurable model + fallback support in llm.py.

- Static behavior after question generation:
   - Challenge: one-shot output felt rigid.
   - Solution: added optional assessment mode and post-screening guided conversation.

- Invalid noisy inputs:
   - Challenge: random tokens could pass as fields.
   - Solution: stricter validators and tech-token filters.

## Testing

Suggested local checks:

- Run automated tests:

pytest -q

- Run app with and without API key to validate fallback behavior.
- Validate email/phone/experience input edge cases.
- Test multi-position and mixed tech-stack parsing.
- Execute start assessment flow and verify summary output.
- Verify clear-session and consent behavior.

## Deployment Notes

- Local deployment is fully supported.
- Optional cloud deployment can be done on Streamlit Community Cloud, AWS, or GCP.

## Submission Checklist

- Source code repository (public link) or zip package
- README with setup, architecture, and prompt design
- Demo video or walkthrough link
- Screenshots of key flows (optional but recommended)

## Troubleshooting

- If API is not used, verify GROQ_API_KEY in .env.
- If model fails, update GROQ_MODEL to an active Groq model.
- If Streamlit does not start, verify environment and dependencies.
