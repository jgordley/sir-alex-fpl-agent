"""FPL API service for fetching Fantasy Premier League data."""

import os
import asyncio
import aiohttp
from fpl import FPL


FPL_EMAIL = os.getenv("FPL_EMAIL")
FPL_PASSWORD = os.getenv("FPL_PASSWORD")


async def _get_fpl_client(session: aiohttp.ClientSession, login: bool = False) -> FPL:
    """Create an FPL client, optionally logging in."""
    fpl = FPL(session)
    if login and FPL_EMAIL and FPL_PASSWORD:
        await fpl.login(email=FPL_EMAIL, password=FPL_PASSWORD)
    return fpl


async def get_player(player_id: int) -> dict:
    """Get a player by their ID."""
    async with aiohttp.ClientSession() as session:
        fpl = await _get_fpl_client(session)
        player = await fpl.get_player(player_id, return_json=True)
        return player


async def get_player_by_name(name: str) -> dict | None:
    """Search for a player by name."""
    async with aiohttp.ClientSession() as session:
        fpl = await _get_fpl_client(session)
        players = await fpl.get_players(return_json=True)
        name_lower = name.lower()
        for player in players:
            full_name = f"{player.get('first_name', '')} {player.get('second_name', '')}".lower()
            web_name = player.get('web_name', '').lower()
            if name_lower in full_name or name_lower in web_name:
                return player
        return None


async def get_top_players(position: str | None = None, limit: int = 10) -> list[dict]:
    """Get top players by total points, optionally filtered by position."""
    position_map = {
        "goalkeeper": 1, "gk": 1,
        "defender": 2, "def": 2,
        "midfielder": 3, "mid": 3,
        "forward": 4, "fwd": 4, "striker": 4,
    }

    async with aiohttp.ClientSession() as session:
        fpl = await _get_fpl_client(session)
        players = await fpl.get_players(return_json=True)

        if position:
            pos_id = position_map.get(position.lower())
            if pos_id:
                players = [p for p in players if p.get("element_type") == pos_id]

        sorted_players = sorted(players, key=lambda x: x.get("total_points", 0), reverse=True)
        return sorted_players[:limit]


async def get_user_team(team_id: int) -> dict:
    """Get a user's current team (requires authentication)."""
    async with aiohttp.ClientSession() as session:
        fpl = await _get_fpl_client(session, login=True)
        user = await fpl.get_user(team_id)
        team = await user.get_team()

        # Get player details for each team member
        players = await fpl.get_players(return_json=True)
        player_map = {p["id"]: p for p in players}

        enriched_team = []
        for entry in team:
            player_id = entry.get("element")
            player_info = player_map.get(player_id, {})
            enriched_team.append({
                **entry,
                "player_name": player_info.get("web_name", "Unknown"),
                "team_name": player_info.get("team", "Unknown"),
                "total_points": player_info.get("total_points", 0),
                "now_cost": player_info.get("now_cost", 0),
            })

        return {"team": enriched_team}


async def get_user_info(team_id: int) -> dict:
    """Get basic user/team info."""
    async with aiohttp.ClientSession() as session:
        fpl = await _get_fpl_client(session)
        user = await fpl.get_user(team_id, return_json=True)
        return user


async def get_current_gameweek() -> dict:
    """Get the current gameweek information."""
    async with aiohttp.ClientSession() as session:
        fpl = await _get_fpl_client(session)
        gameweeks = await fpl.get_gameweeks(return_json=True)
        for gw in gameweeks:
            if gw.get("is_current"):
                return gw
        return gameweeks[-1] if gameweeks else {}


async def get_fixtures(gameweek: int | None = None) -> list[dict]:
    """Get fixtures, optionally for a specific gameweek."""
    async with aiohttp.ClientSession() as session:
        fpl = await _get_fpl_client(session)
        fixtures = await fpl.get_fixtures(return_json=True)
        if gameweek:
            fixtures = [f for f in fixtures if f.get("event") == gameweek]
        return fixtures


async def get_teams() -> list[dict]:
    """Get all Premier League teams."""
    async with aiohttp.ClientSession() as session:
        fpl = await _get_fpl_client(session)
        teams = await fpl.get_teams(return_json=True)
        return teams


# Synchronous wrappers for use in Streamlit
def sync_get_player(player_id: int) -> dict:
    """Synchronous wrapper for get_player."""
    return asyncio.run(get_player(player_id))


def sync_get_player_by_name(name: str) -> dict | None:
    """Synchronous wrapper for get_player_by_name."""
    return asyncio.run(get_player_by_name(name))


def sync_get_top_players(position: str | None = None, limit: int = 10) -> list[dict]:
    """Synchronous wrapper for get_top_players."""
    return asyncio.run(get_top_players(position, limit))


def sync_get_user_team(team_id: int) -> dict:
    """Synchronous wrapper for get_user_team."""
    return asyncio.run(get_user_team(team_id))


def sync_get_user_info(team_id: int) -> dict:
    """Synchronous wrapper for get_user_info."""
    return asyncio.run(get_user_info(team_id))


def sync_get_current_gameweek() -> dict:
    """Synchronous wrapper for get_current_gameweek."""
    return asyncio.run(get_current_gameweek())


def sync_get_fixtures(gameweek: int | None = None) -> list[dict]:
    """Synchronous wrapper for get_fixtures."""
    return asyncio.run(get_fixtures(gameweek))


def sync_get_teams() -> list[dict]:
    """Synchronous wrapper for get_teams."""
    return asyncio.run(get_teams())
