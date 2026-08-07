"""Lab 11 — Helper utilities shared by attack and testing flows."""


async def chat_with_agent(agent, runner, user_message: str, session_id=None):
    """Send a message through the configured OpenAI-compatible runner.

    Args:
        agent: Agent definition returned by create_*_agent()
        runner: Runner exposing an async ``chat`` method
        user_message: Plain text message to send
        session_id: Optional session ID to continue a conversation

    Returns:
        Tuple of (response_text, session)
    """
    if not hasattr(runner, "chat"):
        raise TypeError("Runner must expose async chat()")
    return await runner.chat(
        user_message,
        session_id=session_id,
        user_id="student",
    )
