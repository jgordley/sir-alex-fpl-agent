"""Main Streamlit application for Sir Alex FPL Agent."""

import streamlit as st

from agent import run_agent, get_conversation_history
from app.constants import (
    AVAILABLE_MODELS,
    FPL_TEAM_ID_HELP,
    FPL_TEAM_ID_LABEL,
    MODEL_LABEL,
    SESSION_ID_HELP,
    SESSION_ID_LABEL,
    SIDEBAR_TITLE,
    UNIQUE_ID_HELP,
    UNIQUE_ID_LABEL,
    WELCOME_MESSAGE,
)

# Page configuration
st.set_page_config(
    page_title="Sir Alex - FPL Agent",
    page_icon="⚽",
    layout="centered",
)


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "unique_id" not in st.session_state:
        st.session_state.unique_id = ""
    if "session_id" not in st.session_state:
        st.session_state.session_id = ""
    if "fpl_team_id" not in st.session_state:
        st.session_state.fpl_team_id = ""
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = AVAILABLE_MODELS[0][1]
    if "_last_session_key" not in st.session_state:
        st.session_state._last_session_key = ""


def load_conversation_from_memory():
    """Load conversation history from AgentCore memory if session changed."""
    current_key = f"{st.session_state.unique_id}:{st.session_state.session_id}"

    # Check if session changed
    if current_key != st.session_state._last_session_key:
        st.session_state._last_session_key = current_key

        # Load history from memory if we have both IDs
        if st.session_state.unique_id and st.session_state.session_id:
            history = get_conversation_history(
                model_name=st.session_state.selected_model,
                actor_id=st.session_state.unique_id,
                session_id=st.session_state.session_id,
            )
            st.session_state.messages = history
        else:
            st.session_state.messages = []


def render_sidebar():
    """Render the sidebar with settings."""
    with st.sidebar:
        st.title(SIDEBAR_TITLE)

        # Unique ID input
        unique_id = st.text_input(
            UNIQUE_ID_LABEL,
            value=st.session_state.unique_id,
            help=UNIQUE_ID_HELP,
            placeholder="Enter your unique ID",
        )
        st.session_state.unique_id = unique_id

        # Session ID input
        session_id = st.text_input(
            SESSION_ID_LABEL,
            value=st.session_state.session_id,
            help=SESSION_ID_HELP,
            placeholder="e.g., session-1",
        )
        st.session_state.session_id = session_id

        # Model selection dropdown
        model_names = [model[0] for model in AVAILABLE_MODELS]
        model_values = [model[1] for model in AVAILABLE_MODELS]

        current_index = 0
        if st.session_state.selected_model in model_values:
            current_index = model_values.index(st.session_state.selected_model)

        selected_name = st.selectbox(
            MODEL_LABEL,
            options=model_names,
            index=current_index,
        )
        st.session_state.selected_model = model_values[model_names.index(selected_name)]

        # FPL Team ID input (optional)
        fpl_team_id = st.text_input(
            FPL_TEAM_ID_LABEL,
            value=st.session_state.fpl_team_id,
            help=FPL_TEAM_ID_HELP,
            placeholder="e.g., 1234567",
        )
        st.session_state.fpl_team_id = fpl_team_id

        st.divider()

        # Display current settings
        if unique_id:
            st.caption(f"Logged in as: **{unique_id}**")
        if session_id:
            st.caption(f"Session: **{session_id}**")
        if fpl_team_id:
            st.caption(f"FPL Team: **{fpl_team_id}**")
        st.caption(f"Model: **{selected_name}**")


def render_chat():
    """Render the main chat interface."""
    st.title("⚽ Sir Alex - FPL Agent")

    # Show welcome message if no messages yet
    if not st.session_state.messages:
        st.markdown(WELCOME_MESSAGE)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Display tool calls if present (for assistant messages)
            if message.get("tool_calls"):
                with st.expander("Tool Calls", expanded=False):
                    for tc in message["tool_calls"]:
                        st.markdown(f"**{tc['name']}**")
                        st.code(f"Arguments: {tc['args']}\nResult: {tc['result']}")
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask Sir Alex about FPL..."):
        # Check if unique ID is set
        if not st.session_state.unique_id:
            st.warning("Please enter your Unique ID in the sidebar to continue.")
            return

        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = run_agent(
                        user_message=prompt,
                        model_name=st.session_state.selected_model,
                        actor_id=st.session_state.unique_id or None,
                        session_id=st.session_state.session_id or None,
                        fpl_team_id=st.session_state.fpl_team_id or None,
                    )

                    # Display tool calls if any
                    if response.tool_calls:
                        with st.expander("Tool Calls", expanded=True):
                            for tc in response.tool_calls:
                                st.markdown(f"**{tc.name}**")
                                st.code(f"Arguments: {tc.args}\nResult: {tc.result}")

                    st.markdown(response.content)

                    # Add assistant response to history (including tool calls)
                    assistant_msg = {
                        "role": "assistant",
                        "content": response.content,
                    }
                    if response.tool_calls:
                        assistant_msg["tool_calls"] = [
                            {"name": tc.name, "args": tc.args, "result": tc.result}
                            for tc in response.tool_calls
                        ]
                    st.session_state.messages.append(assistant_msg)
                except Exception as e:
                    st.error(f"Error getting response: {str(e)}")


def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    load_conversation_from_memory()
    render_chat()


if __name__ == "__main__":
    main()
