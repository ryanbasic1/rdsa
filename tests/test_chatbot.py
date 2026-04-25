from chatbot import TalentScoutChatbot


class FakeLLM:
    is_available = False

    def safe_generate_response(self, prompt, default_response, system_prompt=None):
        return default_response



def _complete_profile(bot: TalentScoutChatbot):
    bot.start_conversation()
    bot.handle_message("Aryan Vishwakarma")
    bot.handle_message("aryan@example.com")
    bot.handle_message("7972250722")
    bot.handle_message("4")
    bot.handle_message("Python Developer, Backend Engineer")
    bot.handle_message("Nashik")
    final_response = bot.handle_message("Python, Django")
    return final_response



def test_chatbot_collects_profile_and_generates_questions():
    bot = TalentScoutChatbot(llm=FakeLLM())
    response = _complete_profile(bot)

    profile = bot.get_profile_data()
    question_bank = bot.get_question_bank()

    assert bot.finished is True
    assert profile["desired_positions"] == ["Python Developer", "Backend Engineer"]
    assert "Python" in question_bank
    assert "Django" in question_bank
    assert "start assessment" in response.lower()



def test_chatbot_assessment_round_captures_answers_and_generates_summary():
    bot = TalentScoutChatbot(llm=FakeLLM())
    _complete_profile(bot)

    start_response = bot.handle_message("start assessment")
    assert "Question 1 of" in start_response

    queued = bot.get_assessment_report()["queued"]
    final_response = ""
    for _ in range(queued):
        final_response = bot.handle_message(
            "I solved this by designing modular services and validating edge cases."
        )

    report = bot.get_assessment_report()
    assert report["answered"] == queued
    assert report["summary"] is not None
    assert "Technical round completed" in final_response


def test_question_input_does_not_get_stored_as_full_name():
    bot = TalentScoutChatbot(llm=FakeLLM())
    bot.start_conversation()

    response = bot.handle_message("Why do you need this detail?")

    profile = bot.get_profile_data()
    assert profile["full_name"] is None
    assert bot.get_progress()["next_field"] == "full_name"
    assert "Please share your full name" in response
