"""FPL API service for fetching Fantasy Premier League data."""

import os
import asyncio
import aiohttp
from fpl import FPL


async def _get_fpl_client(session: aiohttp.ClientSession, login: bool = False) -> FPL:
    """Create an FPL client, optionally logging in."""
    fpl = FPL(session)
    if login:
        email = os.getenv("FPL_EMAIL")
        password = os.getenv("FPL_PASSWORD")
        if email and password:
            await fpl.login(email=email, password=password)
    return fpl


async def get_player(player_id: int) -> dict:
    """Get a player by their ID."""
    async with aiohttp.ClientSession() as session:
        fpl = await _get_fpl_client(session)
        player = await fpl.get_player(player_id, return_json=True)
        return player


async def get_player_by_name(name: str, season: str | None = None) -> dict | None:
    """Search for a player by name, optionally for a specific season.

    Args:
        name: Player name to search for.
        season: Optional season in format "YY/YY" (e.g., "24/25", "23/24").
                If provided, returns stats from that historical season.
                If None, returns current season stats.

    Returns:
        Player data dict or None if not found.
    """
    async with aiohttp.ClientSession() as session:
        fpl = await _get_fpl_client(session)
        players = await fpl.get_players(return_json=True)
        name_lower = name.lower()

        found_player = None
        for player in players:
            full_name = f"{player.get('first_name', '')} {player.get('second_name', '')}".lower()
            web_name = player.get("web_name", "").lower()
            if name_lower in full_name or name_lower in web_name:
                found_player = player
                break

        if not found_player:
            return None

        if season:
            # Fetch historical data and filter to requested season
            player_summary = await fpl.get_player_summary(
                found_player["id"], return_json=True
            )
            history_past = player_summary.get("history_past", [])

            # Convert "YY/YY" to "20YY/YY" format used by API (e.g., "24/25" -> "2024/25")
            season_full = f"20{season}"

            for past_season in history_past:
                if past_season.get("season_name") == season_full:
                    found_player["season_data"] = past_season
                    found_player["is_historical"] = True
                    break
            else:
                # Season not found in history
                found_player["season_data"] = None
                found_player["is_historical"] = True

        return found_player


async def get_top_players(position: str | None = None, limit: int = 10) -> list[dict]:
    """Get top players by total points, optionally filtered by position."""
    position_map = {
        "goalkeeper": 1,
        "gk": 1,
        "defender": 2,
        "def": 2,
        "midfielder": 3,
        "mid": 3,
        "forward": 4,
        "fwd": 4,
        "striker": 4,
    }

    async with aiohttp.ClientSession() as session:
        fpl = await _get_fpl_client(session)
        players = await fpl.get_players(return_json=True)

        if position:
            pos_id = position_map.get(position.lower())
            if pos_id:
                players = [p for p in players if p.get("element_type") == pos_id]

        sorted_players = sorted(
            players, key=lambda x: x.get("total_points", 0), reverse=True
        )
        return sorted_players[:limit]


async def get_user_team(team_id: int, gameweek: int | None = None) -> dict:
    """Get a user's team for a specific gameweek.

    Uses public API endpoint - no authentication required.

    Args:
        team_id: The FPL team ID.
        gameweek: Gameweek number. If None, uses current gameweek.

    Returns:
        Dict with team picks and player details.
    """
    async with aiohttp.ClientSession() as session:
        fpl = await _get_fpl_client(session)

        # Get current gameweek if not specified
        if gameweek is None:
            gameweeks = await fpl.get_gameweeks(return_json=True)
            for gw in gameweeks:
                if gw.get("is_current"):
                    gameweek = gw["id"]
                    break
            if gameweek is None:
                gameweek = 1

        # Fetch picks directly from public API
        url = f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{gameweek}/picks/"
        async with session.get(url) as response:
            if response.status != 200:
                return {"team": [], "error": f"Failed to fetch team: {response.status}"}
            picks_data = await response.json()

        team = picks_data.get("picks", [])

        # Get player details for each team member
        players = await fpl.get_players(return_json=True)
        player_map = {p["id"]: p for p in players}

        enriched_team = []
        for entry in team:
            player_id = entry.get("element")
            player_info = player_map.get(player_id, {})
            enriched_team.append(
                {
                    **entry,
                    "player_name": player_info.get("web_name", "Unknown"),
                    "team_name": player_info.get("team", "Unknown"),
                    "total_points": player_info.get("total_points", 0),
                    "now_cost": player_info.get("now_cost", 0),
                }
            )

        return {
            "team": enriched_team,
            "gameweek": gameweek,
            "active_chip": picks_data.get("active_chip"),
            "points": picks_data.get("entry_history", {}).get("points"),
        }


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


def sync_get_player_by_name(name: str, season: str | None = None) -> dict | None:
    """Synchronous wrapper for get_player_by_name."""
    return asyncio.run(get_player_by_name(name, season))


def sync_get_top_players(position: str | None = None, limit: int = 10) -> list[dict]:
    """Synchronous wrapper for get_top_players."""
    return asyncio.run(get_top_players(position, limit))


def sync_get_user_team(team_id: int, gameweek: int | None = None) -> dict:
    """Synchronous wrapper for get_user_team."""
    return asyncio.run(get_user_team(team_id, gameweek))


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
