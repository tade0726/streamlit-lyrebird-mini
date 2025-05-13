import json
import os
import tempfile

import streamlit as st
from langsmith import Client
from langsmith.wrappers import wrap_openai
from openai import OpenAI

MODEL = "gpt-4o"


def _meta_llm_function(client: OpenAI, prompt_name: str, json_mode=False, **kwargs):

    langsmith_client = Client()
    prompt = langsmith_client.pull_prompt(prompt_name)

    openai_client = wrap_openai(OpenAI())

    # Base parameters for the API call
    params = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt.format(**kwargs)}],
    }

    # Add response_format for JSON mode
    if json_mode:
        params["response_format"] = {"type": "json_object"}

    # Make the API call
    response = openai_client.chat.completions.create(**params)

    return response.choices[0].message.content


def create_memory(
    client: OpenAI, llm_version: str, user_version: str, memory: list
) -> dict:
    """Create a memory based on differences between original and edited versions.

    Args:
        client: OpenAI client
        llm_version: The AI-generated formatted text
        user_version: The user-edited version of the text
        memory: List of existing user memories/preferences

    Returns:
        Dictionary with memory_to_write field (or empty if no new memory)
    """
    # Format the memory list to a string
    memory_str = "\n".join([f"• {m}" for m in memory]) if memory else "No memories yet."

    # Call the model with JSON mode enabled
    response = _meta_llm_function(
        client,
        "create-memory",
        json_mode=True,  # Enable JSON mode for structured response
        llm_version=llm_version,
        user_version=user_version,
        user_memory=memory_str,  # This parameter name must match exactly with {user_memory} in the prompt
    )

    try:
        # Parse the JSON response
        return json.loads(response)
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        st.error("Failed to parse memory creation response")
        return {
            "memory_to_write": False
        }  # Python False, when returned as JSON, will be converted to lowercase false


def text_to_format(client: OpenAI, transcript: str, memories: list = None) -> str:
    """Format a transcript using the clinical documentation specialist prompt.

    Args:
        client: OpenAI client
        transcript: The transcript text to format
        memories: List of user formatting preferences/memories

    Returns:
        Formatted transcript text
    """
    # Format the memories as a bulleted list or show none available
    formatted_memories = (
        "\n".join([f"- {memory}" for memory in memories])
        if memories
        else "No specific preferences recorded yet."
    )

    return _meta_llm_function(
        client, "format-transcript", transcript=transcript, memories=formatted_memories
    )


def audio_to_text(client: OpenAI, audio: bytes) -> str:
    """Convert audio bytes to text using OpenAI's transcription API.

    Args:
        client: OpenAI client
        audio: Audio bytes to transcribe

    Returns:
        Transcribed text
    """
    # Save audio to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio)
        temp_filename = f.name

    try:
        # Open the file in binary mode to get a proper file object
        with open(temp_filename, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",  # Use whisper-1 which has better format support
                file=f,
            )

        return transcription.text
    finally:
        # Clean up the temporary file, even if an error occurs
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)
