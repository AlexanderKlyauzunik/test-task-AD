from aiocache import Cache

# Instantiate the in-memory cache for conversation history.
in_memory_cache = Cache(Cache.MEMORY)
DEFAULT_USER_ID = "test_user"

async def fetch_conversation_history(user_identifier: str) -> list[dict[str, str]]:
    """Retrieves the conversation memory (a list of messages) for a given user identifier."""
    history_list = await in_memory_cache.get(user_identifier)
    # Return the history if found, otherwise an empty list.
    return history_list if history_list is not None else []

async def append_message_to_history(user_identifier: str, sender_role: str, message_content: str):
    """
    Adds a new message to a user's conversation memory in the cache.
    This executes the full "Read-Modify-Write" cycle to ensure memory integrity.
    """
    # READ the current history list
    history_list = await fetch_conversation_history(user_identifier)

    # MODIFY by appending the new message
    history_list.append({"role": sender_role, "content": message_content})

    # WRITE the entire updated history list back to the cache
    await in_memory_cache.set(user_identifier, history_list)
