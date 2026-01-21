# Sir Alex FPL Agent - Development Context

## Project Goal
Build a Fantasy Premier League assistant agent that helps users manage their FPL teams through natural conversation. Named after Sir Alex Ferguson.

## Architecture
- **UI**: Streamlit (`app/main.py`)
- **Agent**: LangGraph with `tools_condition` from prebuilt (`agent/sir_alex.py`)
- **LLM**: OpenRouter (supports multiple models via `OPENROUTER_API_KEY`)
- **FPL Data**: `fpl` library (async, uses aiohttp)
- **Deployment**: Digital Ocean App Platform

## Key Files
```
app/main.py          # Streamlit entry point
app/constants.py     # Prompts, model list, UI strings
app/config.py        # Settings (Pydantic BaseModel)
app/fpl_service.py   # FPL API wrappers (async + sync versions)
agent/sir_alex.py    # LangGraph agent definition
agent/tools.py       # LangChain @tool definitions
```

## Running Locally
```bash
PYTHONPATH=. streamlit run app/main.py
```
The `PYTHONPATH=.` is required for absolute imports (`from app.x import y`). On Digital Ocean, set `PYTHONPATH=/workspace` as env var.

## FPL API Key Points

### Library
Using `fpl` package (https://github.com/amosbastian/fpl). Async-only, requires `aiohttp.ClientSession`.

### No Server-Side Filtering
All filtering (position, team, price, etc.) must be done client-side after fetching data.

### Season Data
- Current season stats are in the base player response
- Historical seasons require `get_player_summary()` which returns `history_past` array
- Season format in API: `"2024/25"` - we accept `"24/25"` and convert to `"2024/25"`

### Position IDs
```python
1 = Goalkeeper, 2 = Defender, 3 = Midfielder, 4 = Forward
```

### Player Cost
Stored as integer in tenths of millions. Divide by 10 for display (e.g., `130` = `£13.0m`).

### Key Player Fields
- `web_name`: Display name (e.g., "Salah")
- `total_points`, `points_per_game`, `form`
- `expected_goals`, `expected_assists`, `ict_index`
- `status`: 'a' (available), 'd' (doubt), 'i' (injured), 'u' (unavailable)
- `news`: Injury/availability updates
- `selected_by_percent`: Ownership percentage

### Auth-Required Endpoints
User team data requires FPL login via `FPL_EMAIL` and `FPL_PASSWORD` env vars.

## Agent Implementation

### State
```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
```

### Graph Structure
```
START -> agent -> tools_condition -> tools -> agent (loop) or END
```

### Tool Response Format
Tools return `AgentResponse` (Pydantic) with `content` and `tool_calls` list for UI display.

### Adding New Tools
1. Add to `agent/tools.py` with `@tool` decorator
2. Add sync wrapper in `app/fpl_service.py` if calling FPL API
3. Include in `ALL_TOOLS` list

## Testing

### Structure
```
tests/unit/          # Mocked tests, run in CI
tests/integration/   # Real LLM calls, require credentials
```

### Commands
```bash
pytest tests/unit -v           # Unit only (CI)
pytest tests/integration -v    # Integration only (local)
pytest -v                      # All tests
```

### Integration Tests
- Use AgentEvals for trajectory matching
- Model defined in `tests/integration/conftest.py` as `TEST_MODEL`
- Require `OPENROUTER_API_KEY` env var

## Environment Variables
```
OPENROUTER_API_KEY    # Required - LLM access
PYTHONPATH            # Required for imports (/workspace on DO)
FPL_EMAIL             # Optional - for user team data
FPL_PASSWORD          # Optional - for user team data

# AWS/AgentCore Memory
AWS_ACCESS_KEY_ID     # Required for AgentCore Memory
AWS_SECRET_ACCESS_KEY # Required for AgentCore Memory
AWS_DEFAULT_REGION    # us-west-2
AGENTCORE_MEMORY_ID   # AgentCore Memory resource ID
```

## AgentCore Memory Integration
Uses `langgraph-checkpoint-aws` for conversation persistence via AWS Bedrock AgentCore Memory.

### Key Concepts
- **Memory ID**: Top-level AgentCore Memory resource (from `AGENTCORE_MEMORY_ID`)
- **Actor ID**: User identifier (unique_id from Streamlit sidebar)
- **Thread ID**: Session identifier (session_id from Streamlit sidebar)

### How It Works
1. `AgentCoreMemorySaver` checkpointer is initialized with Memory ID and region
2. Each `agent.invoke()` call passes config with `actor_id` and `thread_id`
3. Conversation state is persisted to AgentCore Memory
4. Same actor_id + thread_id = continues previous conversation
5. Same actor_id + new thread_id = starts fresh conversation

### Code Structure
```python
from langgraph_checkpoint_aws import AgentCoreMemorySaver

checkpointer = AgentCoreMemorySaver(memory_id, region_name="us-west-2")
agent = create_agent(model_name, checkpointer=checkpointer)
agent.invoke(inputs, config={"configurable": {"thread_id": session_id, "actor_id": actor_id}})
```

## Conventions
- Pydantic `BaseModel` for all data classes (not dataclass)
- Async FPL functions with `sync_*` wrappers for Streamlit/tools
- Tools return formatted strings, not raw dicts
- Season format: "YY/YY" (e.g., "24/25")

## Current Tools
1. `math_add(x, y)` - Dummy tool for testing
2. `get_player_stats(player_name, season?)` - Player lookup with optional historical season
3. `get_fpl_team(team_id, gameweek?)` - FPL team squad lookup

## Next Steps (Planned Features)
- More FPL tools: top players, fixtures
- Transfer news monitoring
- User preference storage (via AgentCore Memory strategies)
