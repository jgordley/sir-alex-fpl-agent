"""Pytest configuration and fixtures."""

import pytest

# Mock FPL API responses
MOCK_PLAYERS = [
    {
        "id": 1,
        "first_name": "Mohamed",
        "second_name": "Salah",
        "web_name": "Salah",
        "team": 11,
        "element_type": 3,  # Midfielder
        "total_points": 180,
        "points_per_game": "7.5",
        "now_cost": 130,
        "selected_by_percent": "45.2",
        "form": "8.5",
        "goals_scored": 15,
        "assists": 10,
        "clean_sheets": 0,
        "minutes": 2100,
        "status": "a",
        "news": "",
    },
    {
        "id": 2,
        "first_name": "Erling",
        "second_name": "Haaland",
        "web_name": "Haaland",
        "team": 13,
        "element_type": 4,  # Forward
        "total_points": 200,
        "points_per_game": "8.0",
        "now_cost": 145,
        "selected_by_percent": "60.1",
        "form": "9.0",
        "goals_scored": 25,
        "assists": 5,
        "clean_sheets": 0,
        "minutes": 2200,
        "status": "a",
        "news": "",
    },
    {
        "id": 3,
        "first_name": "Virgil",
        "second_name": "van Dijk",
        "web_name": "Van Dijk",
        "team": 11,
        "element_type": 2,  # Defender
        "total_points": 140,
        "points_per_game": "5.5",
        "now_cost": 65,
        "selected_by_percent": "25.0",
        "form": "6.0",
        "goals_scored": 3,
        "assists": 2,
        "clean_sheets": 12,
        "minutes": 2300,
        "status": "a",
        "news": "",
    },
    {
        "id": 4,
        "first_name": "Alisson",
        "second_name": "Becker",
        "web_name": "Alisson",
        "team": 11,
        "element_type": 1,  # Goalkeeper
        "total_points": 120,
        "points_per_game": "5.0",
        "now_cost": 55,
        "selected_by_percent": "20.0",
        "form": "5.5",
        "goals_scored": 0,
        "assists": 0,
        "clean_sheets": 14,
        "minutes": 2400,
        "status": "a",
        "news": "",
    },
]

MOCK_TEAMS = [
    {
        "id": 11,
        "name": "Liverpool",
        "short_name": "LIV",
        "strength": 5,
        "strength_overall_home": 1300,
        "strength_overall_away": 1280,
        "strength_attack_home": 1320,
        "strength_attack_away": 1300,
        "strength_defence_home": 1280,
        "strength_defence_away": 1260,
    },
    {
        "id": 13,
        "name": "Man City",
        "short_name": "MCI",
        "strength": 5,
        "strength_overall_home": 1350,
        "strength_overall_away": 1320,
        "strength_attack_home": 1380,
        "strength_attack_away": 1350,
        "strength_defence_home": 1300,
        "strength_defence_away": 1280,
    },
]

MOCK_GAMEWEEKS = [
    {
        "id": 20,
        "name": "Gameweek 20",
        "deadline_time": "2025-01-14T18:30:00Z",
        "is_current": True,
        "finished": False,
        "average_entry_score": 52,
        "highest_score": 120,
    },
    {
        "id": 21,
        "name": "Gameweek 21",
        "deadline_time": "2025-01-18T11:00:00Z",
        "is_current": False,
        "finished": False,
        "average_entry_score": None,
        "highest_score": None,
    },
]

MOCK_FIXTURES = [
    {
        "id": 1,
        "event": 20,
        "team_h": 11,
        "team_a": 13,
        "team_h_difficulty": 4,
        "team_a_difficulty": 4,
        "kickoff_time": "2025-01-15T20:00:00Z",
        "finished": False,
        "team_h_score": None,
        "team_a_score": None,
    },
]

MOCK_USER_INFO = {
    "id": 12345,
    "name": "My FPL Team",
    "player_first_name": "John",
    "player_last_name": "Doe",
    "summary_overall_rank": 50000,
    "summary_overall_points": 1200,
    "summary_event_points": 65,
}

MOCK_USER_TEAM = [
    {
        "element": 1,
        "position": 1,
        "is_captain": False,
        "is_vice_captain": False,
        "multiplier": 1,
    },
    {
        "element": 2,
        "position": 2,
        "is_captain": True,
        "is_vice_captain": False,
        "multiplier": 2,
    },
]


@pytest.fixture
def mock_players():
    """Return mock player data."""
    return MOCK_PLAYERS


@pytest.fixture
def mock_teams():
    """Return mock team data."""
    return MOCK_TEAMS


@pytest.fixture
def mock_gameweeks():
    """Return mock gameweek data."""
    return MOCK_GAMEWEEKS


@pytest.fixture
def mock_fixtures():
    """Return mock fixture data."""
    return MOCK_FIXTURES


@pytest.fixture
def mock_user_info():
    """Return mock user info."""
    return MOCK_USER_INFO


@pytest.fixture
def mock_user_team():
    """Return mock user team."""
    return MOCK_USER_TEAM
