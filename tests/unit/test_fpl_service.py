"""Unit tests for FPL service."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from tests.conftest import (
    MOCK_PLAYERS,
    MOCK_TEAMS,
    MOCK_GAMEWEEKS,
    MOCK_FIXTURES,
    MOCK_USER_INFO,
    MOCK_USER_TEAM,
)


class TestGetPlayerByName:
    """Tests for get_player_by_name function."""

    @pytest.mark.asyncio
    async def test_find_player_by_web_name(self):
        """Test finding a player by their web name."""
        from app.fpl_service import get_player_by_name

        with patch("app.fpl_service.aiohttp.ClientSession") as mock_session:
            mock_fpl = AsyncMock()
            mock_fpl.get_players = AsyncMock(return_value=MOCK_PLAYERS)

            with patch("app.fpl_service.FPL", return_value=mock_fpl):
                mock_session.return_value.__aenter__ = AsyncMock(
                    return_value=MagicMock()
                )
                mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await get_player_by_name("Salah")

                assert result is not None
                assert result["web_name"] == "Salah"
                assert result["total_points"] == 180

    @pytest.mark.asyncio
    async def test_find_player_by_full_name(self):
        """Test finding a player by their full name."""
        from app.fpl_service import get_player_by_name

        with patch("app.fpl_service.aiohttp.ClientSession") as mock_session:
            mock_fpl = AsyncMock()
            mock_fpl.get_players = AsyncMock(return_value=MOCK_PLAYERS)

            with patch("app.fpl_service.FPL", return_value=mock_fpl):
                mock_session.return_value.__aenter__ = AsyncMock(
                    return_value=MagicMock()
                )
                mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await get_player_by_name("Mohamed Salah")

                assert result is not None
                assert result["web_name"] == "Salah"

    @pytest.mark.asyncio
    async def test_player_not_found(self):
        """Test that None is returned when player is not found."""
        from app.fpl_service import get_player_by_name

        with patch("app.fpl_service.aiohttp.ClientSession") as mock_session:
            mock_fpl = AsyncMock()
            mock_fpl.get_players = AsyncMock(return_value=MOCK_PLAYERS)

            with patch("app.fpl_service.FPL", return_value=mock_fpl):
                mock_session.return_value.__aenter__ = AsyncMock(
                    return_value=MagicMock()
                )
                mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await get_player_by_name("Nonexistent Player")

                assert result is None


class TestGetTopPlayers:
    """Tests for get_top_players function."""

    @pytest.mark.asyncio
    async def test_get_top_players_all_positions(self):
        """Test getting top players across all positions."""
        from app.fpl_service import get_top_players

        with patch("app.fpl_service.aiohttp.ClientSession") as mock_session:
            mock_fpl = AsyncMock()
            mock_fpl.get_players = AsyncMock(return_value=MOCK_PLAYERS)

            with patch("app.fpl_service.FPL", return_value=mock_fpl):
                mock_session.return_value.__aenter__ = AsyncMock(
                    return_value=MagicMock()
                )
                mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await get_top_players(limit=10)

                assert len(result) == 4  # All mock players
                # Should be sorted by total_points descending
                assert result[0]["web_name"] == "Haaland"  # 200 points
                assert result[1]["web_name"] == "Salah"  # 180 points

    @pytest.mark.asyncio
    async def test_get_top_players_filtered_by_position(self):
        """Test getting top players filtered by position."""
        from app.fpl_service import get_top_players

        with patch("app.fpl_service.aiohttp.ClientSession") as mock_session:
            mock_fpl = AsyncMock()
            mock_fpl.get_players = AsyncMock(return_value=MOCK_PLAYERS)

            with patch("app.fpl_service.FPL", return_value=mock_fpl):
                mock_session.return_value.__aenter__ = AsyncMock(
                    return_value=MagicMock()
                )
                mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await get_top_players(position="forward", limit=10)

                assert len(result) == 1
                assert result[0]["web_name"] == "Haaland"

    @pytest.mark.asyncio
    async def test_get_top_players_with_limit(self):
        """Test limiting the number of returned players."""
        from app.fpl_service import get_top_players

        with patch("app.fpl_service.aiohttp.ClientSession") as mock_session:
            mock_fpl = AsyncMock()
            mock_fpl.get_players = AsyncMock(return_value=MOCK_PLAYERS)

            with patch("app.fpl_service.FPL", return_value=mock_fpl):
                mock_session.return_value.__aenter__ = AsyncMock(
                    return_value=MagicMock()
                )
                mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await get_top_players(limit=2)

                assert len(result) == 2


class TestGetCurrentGameweek:
    """Tests for get_current_gameweek function."""

    @pytest.mark.asyncio
    async def test_get_current_gameweek(self):
        """Test getting the current gameweek."""
        from app.fpl_service import get_current_gameweek

        with patch("app.fpl_service.aiohttp.ClientSession") as mock_session:
            mock_fpl = AsyncMock()
            mock_fpl.get_gameweeks = AsyncMock(return_value=MOCK_GAMEWEEKS)

            with patch("app.fpl_service.FPL", return_value=mock_fpl):
                mock_session.return_value.__aenter__ = AsyncMock(
                    return_value=MagicMock()
                )
                mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await get_current_gameweek()

                assert result["id"] == 20
                assert result["is_current"] is True


class TestGetTeams:
    """Tests for get_teams function."""

    @pytest.mark.asyncio
    async def test_get_all_teams(self):
        """Test getting all Premier League teams."""
        from app.fpl_service import get_teams

        with patch("app.fpl_service.aiohttp.ClientSession") as mock_session:
            mock_fpl = AsyncMock()
            mock_fpl.get_teams = AsyncMock(return_value=MOCK_TEAMS)

            with patch("app.fpl_service.FPL", return_value=mock_fpl):
                mock_session.return_value.__aenter__ = AsyncMock(
                    return_value=MagicMock()
                )
                mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await get_teams()

                assert len(result) == 2
                assert result[0]["name"] == "Liverpool"
                assert result[1]["name"] == "Man City"


class TestGetFixtures:
    """Tests for get_fixtures function."""

    @pytest.mark.asyncio
    async def test_get_fixtures(self):
        """Test getting fixtures."""
        from app.fpl_service import get_fixtures

        with patch("app.fpl_service.aiohttp.ClientSession") as mock_session:
            mock_fpl = AsyncMock()
            mock_fpl.get_fixtures = AsyncMock(return_value=MOCK_FIXTURES)

            with patch("app.fpl_service.FPL", return_value=mock_fpl):
                mock_session.return_value.__aenter__ = AsyncMock(
                    return_value=MagicMock()
                )
                mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await get_fixtures()

                assert len(result) == 1
                assert result[0]["team_h"] == 11

    @pytest.mark.asyncio
    async def test_get_fixtures_by_gameweek(self):
        """Test getting fixtures filtered by gameweek."""
        from app.fpl_service import get_fixtures

        with patch("app.fpl_service.aiohttp.ClientSession") as mock_session:
            mock_fpl = AsyncMock()
            mock_fpl.get_fixtures = AsyncMock(return_value=MOCK_FIXTURES)

            with patch("app.fpl_service.FPL", return_value=mock_fpl):
                mock_session.return_value.__aenter__ = AsyncMock(
                    return_value=MagicMock()
                )
                mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await get_fixtures(gameweek=20)

                assert len(result) == 1
                assert result[0]["event"] == 20
