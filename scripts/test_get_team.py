#!/usr/bin/env python
"""Test script to verify getting an FPL team by ID works."""

import argparse
import asyncio
import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.fpl_service import get_user_team, get_user_info


async def main(team_id: int, debug: bool = False) -> None:
    """Fetch and display team information."""
    print(f"Fetching team {team_id}...\n")

    # Get basic team info
    print("=== Team Info ===")
    try:
        user_info = await get_user_info(team_id)
        print(f"Team Name: {user_info.get('name', 'N/A')}")
        print(
            f"Manager: {user_info.get('player_first_name', '')} {user_info.get('player_last_name', '')}"
        )
        print(f"Overall Rank: {user_info.get('summary_overall_rank', 'N/A'):,}")
        print(f"Total Points: {user_info.get('summary_overall_points', 'N/A')}")
        print(f"Gameweek Points: {user_info.get('summary_event_points', 'N/A')}")
    except Exception as e:
        print(f"Error fetching team info: {e}")
        if debug:
            traceback.print_exc()
        return

    # Get team squad
    print("\n=== Squad ===")
    try:
        team_data = await get_user_team(team_id)

        if team_data.get("error"):
            print(f"Error: {team_data['error']}")
            return

        squad = team_data.get("team", [])
        print(f"Gameweek: {team_data.get('gameweek')}")
        if team_data.get("active_chip"):
            print(f"Active Chip: {team_data['active_chip']}")
        print()

        if not squad:
            print("No squad data available")
            return

        for player in squad:
            captain = " (C)" if player.get("is_captain") else ""
            vice = " (VC)" if player.get("is_vice_captain") else ""
            sub = " [SUB]" if player.get("multiplier") == 0 else ""
            print(
                f"  {player.get('player_name', 'Unknown')}{captain}{vice}{sub} - "
                f"{player.get('total_points', 0)} pts"
            )
    except Exception as e:
        print(f"Error fetching squad: {e}")
        if debug:
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test fetching an FPL team by ID")
    parser.add_argument("team_id", type=int, help="FPL Team ID to fetch")
    parser.add_argument(
        "--debug", "-d", action="store_true", help="Show debug info and full tracebacks"
    )
    args = parser.parse_args()

    asyncio.run(main(args.team_id, args.debug))
