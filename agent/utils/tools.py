"""Tools for Sir Alex FPL Agent."""

from langchain_core.tools import tool

from app.fpl_service import (
    sync_get_player_by_name,
    sync_get_user_team,
    sync_get_league_leaders,
)

POSITION_MAP = {1: "Goalkeeper", 2: "Defender", 3: "Midfielder", 4: "Forward"}


@tool
def math_add(x: float, y: float) -> float:
    """Add two numbers together.

    Args:
        x: The first number to add.
        y: The second number to add.

    Returns:
        The sum of x and y.
    """
    return x + y


@tool
def get_player_stats(player_name: str, season: str | None = None) -> str:
    """Look up a Premier League player's FPL stats by their name.

    Use this tool when the user asks about a specific player's statistics,
    form, price, points, or any other FPL-related information.

    Args:
        player_name: The name of the player to look up (e.g., 'Salah', 'Haaland', 'Palmer')
        season: Optional season in "YY/YY" format (e.g., "23/24", "22/23").
                If not provided, returns current season stats.

    Returns:
        A formatted string with the player's FPL statistics, or an error message if not found.
    """
    player = sync_get_player_by_name(player_name, season)

    if not player:
        return f"Player '{player_name}' not found. Try using their common name (e.g., 'Salah' instead of 'Mohamed Salah')."

    position = POSITION_MAP.get(player.get("element_type", 0), "Unknown")

    # Check if this is historical data
    if player.get("is_historical"):
        season_data = player.get("season_data")
        if not season_data:
            return f"No data found for {player.get('web_name')} in the {season} season. They may not have been in the Premier League."

        return f"""
Player: {player.get('first_name')} {player.get('second_name')} ({player.get('web_name')})
Season: {season_data.get('season_name')}
Position: {position}
Total Points: {season_data.get('total_points', 0)}
Goals: {season_data.get('goals_scored', 0)}
Assists: {season_data.get('assists', 0)}
Clean Sheets: {season_data.get('clean_sheets', 0)}
Minutes Played: {season_data.get('minutes', 0)}
Start Price: £{season_data.get('start_cost', 0) / 10:.1f}m
End Price: £{season_data.get('end_cost', 0) / 10:.1f}m
""".strip()

    # Current season data
    cost = player.get("now_cost", 0) / 10

    return f"""
Player: {player.get('first_name')} {player.get('second_name')} ({player.get('web_name')})
Season: Current (25/26)
Position: {position}
Team ID: {player.get('team')}
Price: £{cost:.1f}m
Total Points: {player.get('total_points', 0)}
Points Per Game: {player.get('points_per_game', 'N/A')}
Form: {player.get('form', 'N/A')}
Selected By: {player.get('selected_by_percent', 'N/A')}%
Goals: {player.get('goals_scored', 0)}
Assists: {player.get('assists', 0)}
Clean Sheets: {player.get('clean_sheets', 0)}
Minutes Played: {player.get('minutes', 0)}
xG: {player.get('expected_goals', 'N/A')}
xA: {player.get('expected_assists', 'N/A')}
ICT Index: {player.get('ict_index', 'N/A')}
Status: {player.get('status', 'Unknown')}
News: {player.get('news') or 'None'}
""".strip()


@tool
def get_fpl_team(team_id: int, gameweek: int | None = None) -> str:
    """Look up an FPL manager's team squad for a specific gameweek.

    Use this tool when the user asks about their team, someone's FPL team,
    or wants to see a squad lineup for a specific gameweek.

    Args:
        team_id: The FPL Team ID (found in the URL of an FPL team page).
        gameweek: Optional gameweek number (1-38). If not provided, uses current gameweek.

    Returns:
        A formatted string with the team's squad, captain, and points.
    """
    team_data = sync_get_user_team(team_id, gameweek)

    if team_data.get("error"):
        return f"Error fetching team: {team_data['error']}"

    squad = team_data.get("team", [])
    if not squad:
        return f"No squad data found for team {team_id}."

    gw = team_data.get("gameweek", "Unknown")
    active_chip = team_data.get("active_chip")
    points = team_data.get("points")

    lines = [f"Team {team_id} - Gameweek {gw}"]
    if active_chip:
        lines.append(f"Active Chip: {active_chip}")
    if points:
        lines.append(f"Gameweek Points: {points}")
    lines.append("")

    starters = [p for p in squad if p.get("multiplier", 0) > 0]
    bench = [p for p in squad if p.get("multiplier", 0) == 0]

    lines.append("Starting XI:")
    for player in starters:
        captain = " (C)" if player.get("is_captain") else ""
        vice = " (VC)" if player.get("is_vice_captain") else ""
        lines.append(
            f"  {player.get('player_name', 'Unknown')}{captain}{vice} - {player.get('total_points', 0)} pts"
        )

    lines.append("")
    lines.append("Bench:")
    for player in bench:
        lines.append(
            f"  {player.get('player_name', 'Unknown')} - {player.get('total_points', 0)} pts"
        )

    return "\n".join(lines)


@tool
def get_league_leaders(category: str, limit: int = 10) -> str:
    """Get the top players in the Premier League for a specific statistical category.

    Use this tool when the user asks about the best players, top scorers, most assists,
    highest points, or any ranking of players by a specific stat.

    Args:
        category: The stat category to rank by. Valid options:
            - goals_scored: Top goal scorers
            - assists: Most assists
            - goals_and_assists: Combined goals + assists
            - total_points: Most FPL points
            - minutes: Most minutes played
            - clean_sheets: Most clean sheets (best for GK/DEF)
            - saves: Most saves (GK only)
            - bonus: Most bonus points
            - bps: Highest BPS score
            - ict_index: Highest ICT index
            - expected_goals: Highest xG
            - expected_assists: Highest xA
            - expected_goal_involvements: Highest xGI
            - points_per_game: Best points per game
            - form: Best current form
            - selected_by_percent: Most owned players
            - now_cost: Most expensive players
        limit: Number of players to return (default 10, max 20).

    Returns:
        A formatted table of the top players in that category.
    """
    # Cap limit at 20
    limit = min(limit, 20)

    # Map user-friendly names to API field names
    category_map = {
        "goals": "goals_scored",
        "goal": "goals_scored",
        "scorers": "goals_scored",
        "points": "total_points",
        "fpl_points": "total_points",
        "xg": "expected_goals",
        "xa": "expected_assists",
        "xgi": "expected_goal_involvements",
        "ict": "ict_index",
        "ownership": "selected_by_percent",
        "price": "now_cost",
        "cost": "now_cost",
        "ppg": "points_per_game",
    }

    # Normalize category
    category_lower = category.lower().replace(" ", "_")
    category_key = category_map.get(category_lower, category_lower)

    players = sync_get_league_leaders(category_key, limit)

    if not players:
        return f"No data found for category '{category}'."

    # Build the table header based on category
    lines = [f"Top {limit} Players by {category.replace('_', ' ').title()}:", ""]

    # Column headers
    if category_key == "goals_and_assists":
        lines.append(
            f"{'Rank':<5} {'Player':<15} {'Price':<7} {'G':<4} {'A':<4} {'G+A':<5}"
        )
        lines.append("-" * 45)
    elif category_key == "goals_scored":
        lines.append(f"{'Rank':<5} {'Player':<15} {'Price':<7} {'Goals':<6}")
        lines.append("-" * 40)
    elif category_key == "assists":
        lines.append(f"{'Rank':<5} {'Player':<15} {'Price':<7} {'Assists':<8}")
        lines.append("-" * 42)
    elif category_key == "total_points":
        lines.append(f"{'Rank':<5} {'Player':<15} {'Price':<7} {'Points':<7}")
        lines.append("-" * 42)
    elif category_key in (
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
    ):
        lines.append(
            f"{'Rank':<5} {'Player':<15} {'Price':<7} {category_key.upper():<8}"
        )
        lines.append("-" * 42)
    else:
        lines.append(f"{'Rank':<5} {'Player':<15} {'Price':<7} {'Value':<10}")
        lines.append("-" * 45)

    for i, player in enumerate(players, 1):
        name = player.get("web_name", "Unknown")[:14]
        price = f"£{player.get('now_cost', 0) / 10:.1f}m"

        if category_key == "goals_and_assists":
            goals = player.get("goals_scored", 0)
            assists = player.get("assists", 0)
            lines.append(
                f"{i:<5} {name:<15} {price:<7} {goals:<4} {assists:<4} {goals + assists:<5}"
            )
        elif category_key == "goals_scored":
            lines.append(
                f"{i:<5} {name:<15} {price:<7} {player.get('goals_scored', 0):<6}"
            )
        elif category_key == "assists":
            lines.append(f"{i:<5} {name:<15} {price:<7} {player.get('assists', 0):<8}")
        elif category_key == "total_points":
            lines.append(
                f"{i:<5} {name:<15} {price:<7} {player.get('total_points', 0):<7}"
            )
        elif category_key in (
            "points_per_game",
            "form",
            "selected_by_percent",
            "ict_index",
        ):
            val = player.get(category_key, "0")
            lines.append(f"{i:<5} {name:<15} {price:<7} {val:<10}")
        elif category_key == "now_cost":
            lines.append(f"{i:<5} {name:<15} {price:<7}")
        else:
            val = player.get(category_key, 0)
            lines.append(f"{i:<5} {name:<15} {price:<7} {val:<10}")

    return "\n".join(lines)


# List of all available tools
ALL_TOOLS = [math_add, get_player_stats, get_fpl_team, get_league_leaders]
