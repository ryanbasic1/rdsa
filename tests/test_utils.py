from utils import mask_profile_data, parse_desired_positions, parse_tech_stack


def test_parse_desired_positions_deduplicates_and_filters_noise():
    result = parse_desired_positions("Python Developer, backend engineer, 123, python developer")
    assert result == ["Python Developer", "backend engineer"]


def test_parse_tech_stack_filters_invalid_tokens():
    result = parse_tech_stack("Python, Django, 23r, React")
    assert result == ["Python", "Django", "React"]


def test_mask_profile_data_masks_sensitive_values():
    profile = {
        "full_name": "Aryan Vishwakarma",
        "email": "aryan@example.com",
        "phone_number": "7972250722",
        "years_of_experience": 0,
    }

    masked = mask_profile_data(profile)

    assert masked["full_name"].startswith("Aryan")
    assert "@example.com" in masked["email"]
    assert masked["phone_number"].endswith("0722")
    assert masked["phone_number"].count("*") >= 4
