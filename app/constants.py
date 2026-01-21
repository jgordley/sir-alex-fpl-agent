"""Constants and prompts for Sir Alex FPL Agent."""

# Available models for the dropdown
AVAILABLE_MODELS = [
    ("Claude Haiku 4.5", "anthropic/claude-haiku-4.5"),
    ("Llama 4 Maverick", "meta-llama/llama-4-maverick"),
    ("GPT OSS 120B", "openai/gpt-oss-120b"),
]

# System prompt for the FPL assistant
SYSTEM_PROMPT = """You are Sir Alex, an expert Fantasy Premier League (FPL) assistant named after the legendary Manchester United manager. You help users make informed decisions about their FPL teams.

Your expertise includes:
- Player statistics and form analysis
- Transfer recommendations
- Captain picks and team selection
- Understanding fixture difficulty
- Budget management strategies

Be helpful, knowledgeable, and occasionally share football wisdom. Keep responses concise but informative."""

# Welcome message
WELCOME_MESSAGE = """👋 Welcome to Sir Alex - your FPL Assistant!

I'm here to help you build the ultimate Fantasy Premier League team. To get started:

1. Enter your **Unique ID** in the sidebar
2. Optionally add your **FPL Team ID** to get personalized advice
3. Select your preferred **AI model**

Then ask me anything about FPL - transfers, captaincy picks, team selection, or general strategy!"""

# Sidebar labels
SIDEBAR_TITLE = "Settings"
UNIQUE_ID_LABEL = "Unique ID"
UNIQUE_ID_HELP = "Enter your unique identifier to save your preferences"
FPL_TEAM_ID_LABEL = "FPL Team ID (Optional)"
FPL_TEAM_ID_HELP = "Find this in your FPL team URL: https://fantasy.premierleague.com/entry/TEAM_ID/event/1"
MODEL_LABEL = "Select Model"
