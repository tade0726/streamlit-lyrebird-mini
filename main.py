import asyncio
import difflib
import os
import tempfile
from datetime import datetime

import pyperclip
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder
from supabase import Client, create_client

from app.openai_functions import audio_to_text
from app.openai_functions import create_memory as create_memory_prompt
from app.openai_functions import text_to_format
from app.orm import Memory, init_db
from app.prompts import init_prompts

load_dotenv()
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)


# init tables
init_db()


# init openai client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def sign_up(email, password):
    try:
        user = supabase.auth.sign_up({"email": email, "password": password})
        return user
    except Exception as e:
        st.error(f"Registration failed: {e}")


def sign_in(email, password):
    try:
        user = supabase.auth.sign_in_with_password(
            {"email": email, "password": password}
        )
        return user
    except Exception as e:
        st.error(f"Login failed: {e}")


def sign_out():
    try:
        supabase.auth.sign_out()
        st.session_state.user_email = None
        st.rerun()
    except Exception as e:
        st.error(f"Logout failed: {e}")


def save_memory(user_email, memory_text):
    """Save a user memory to the database.

    Args:
        user_email (str): The email of the user
        memory_text (str): The memory text to save

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Insert using Supabase data API
        result = (
            supabase.table("memories")
            .insert(
                {
                    "user_email": user_email,
                    "memory": memory_text,
                    "created_at": datetime.now().isoformat(),
                }
            )
            .execute()
        )

        return True if result else False
    except Exception as e:
        st.error(f"Failed to save memory: {e}")
        return False


def get_memories(user_email):
    """Get all memories for a specific user.

    Args:
        user_email (str): The email of the user

    Returns:
        list: List of memory strings
    """
    try:
        # Query memories for the specific user, ordered by creation date (newest first)
        result = (
            supabase.table("memories")
            .select("*")
            .eq("user_email", user_email)
            .order("created_at", desc=True)
            .execute()
        )

        # Extract just the memory text from each record
        memories = [record["memory"] for record in result.data] if result.data else []

        return memories
    except Exception as e:
        st.error(f"Failed to retrieve memories: {e}")
        return []


def get_user_memories(user_email):
    """Helper function to get user memories from session state or database.

    Args:
        user_email: The email of the user

    Returns:
        List of user memory strings
    """
    # Initialize user memories if they don't exist in session state
    if "user_memories" not in st.session_state:
        st.session_state.user_memories = get_memories(user_email)

    # Return the memories (empty list if none exist)
    return st.session_state.user_memories if "user_memories" in st.session_state else []


def main_app(user_email):

    st.title("🎙️ Medical Audio Transcription App")
    st.success(f"Welcome, {user_email}! 👋")

    # Initialize session state variables if they don't exist
    if "transcript" not in st.session_state:
        st.session_state.transcript = ""
    if "direct_input" not in st.session_state:
        st.session_state.direct_input = ""
    if "tab_selection" not in st.session_state:
        st.session_state.tab_selection = 0  # Default to first tab

    # Get user memories
    user_memories = get_user_memories(user_email)

    # Display user memories in the sidebar
    if user_memories:
        with st.sidebar.expander(
            f"Your Formatting Preferences ({len(user_memories)})", expanded=False
        ):
            for i, mem in enumerate(user_memories):
                st.write(f"{i+1}. {mem}")
    else:
        st.sidebar.info(
            "No formatting preferences saved yet. Edit formatted text to create preferences."
        )

    # Create tabs with the selected tab active
    tab_names = ["🎙️ Audio Transcription", "✏️ Text Formatting"]
    tabs = st.tabs(tab_names)
    audio_tab = tabs[0]
    text_tab = tabs[1]

    # Tab 1: Audio Recording and Transcription Only
    with audio_tab:
        st.subheader("Record Audio for Transcription")

        # Audio recorder component
        audio_dict = mic_recorder(
            start_prompt="Start recording",
            stop_prompt="Stop & transcribe",
            key="recorder",
            just_once=True,  # return once, then reset
        )

        # Process audio recording if available
        if audio_dict:
            audio_value = audio_dict["bytes"]
            # transcribe audio only (no formatting)
            with st.spinner("Transcribing audio..."):
                transcript = audio_to_text(client, audio_value)
                st.session_state.transcript = transcript

        # Format transcript
        if st.session_state.transcript:
            with st.spinner("Formatting transcript..."):
                # Format the transcript with user memories
                formatted_transcript = text_to_format(
                    client, st.session_state.transcript, memories=user_memories
                )
                st.session_state.formatted_transcript = formatted_transcript

        # Always display the text area for transcript
        st.subheader("Transcription Result")

        # Display the raw transcript (always show the text area)
        raw_transcript = st.text_area(
            "Raw transcript:",
            value=(
                st.session_state.transcript if "transcript" in st.session_state else ""
            ),
            height=200,
            key="raw_transcript_area",
            placeholder="Your transcription will appear here after recording. You can also type or paste text here directly.",
        )

        # Store any edits to the raw transcript
        st.session_state.transcript = raw_transcript

        # Add buttons in columns for better layout
        col1, col2 = st.columns(2)

        # Only show action buttons if there's text in the transcript
        if st.session_state.transcript.strip():
            # Copy button for raw transcript
            with col1:
                if st.button("Copy to Clipboard", key="audio_copy"):
                    pyperclip.copy(st.session_state.transcript)
                    st.success("Transcript copied to clipboard!")

            # Send to formatting button
            with col2:
                if st.button("Send to Formatting Tab", key="send_to_format"):
                    # Store transcript in the direct input field for the formatting tab
                    st.session_state.direct_input = st.session_state.transcript
                    # Set the tab selection to the formatting tab (index 1)
                    st.session_state.tab_selection = 1
                    st.rerun()
        else:
            st.info("Record audio or type directly in the text area above.")

    # Tab 2: Text Formatting Only
    with text_tab:
        st.subheader("Direct Text Formatting")
        st.write("Enter your text below to format it without recording audio.")

        # Input text area for direct text input
        direct_input = st.text_area(
            "Enter your text:",
            value=st.session_state.direct_input,
            height=250,  # Increased height for better writing experience
            key="direct_input_area",
            placeholder="Enter or paste your text here for formatting...",
        )

        # Format button
        if st.button("Format Text"):
            if direct_input.strip():
                with st.spinner("Formatting text..."):
                    st.session_state.direct_input = direct_input

                    # Format the text with user memories
                    formatted_text = text_to_format(
                        client, direct_input, memories=user_memories
                    )
                    st.session_state.formatted_direct_text = formatted_text
            else:
                st.warning("Please enter some text to format.")

        # Display formatted result
        if "formatted_direct_text" in st.session_state:
            st.subheader("Formatted Result")

            # Display the original AI-formatted text (read-only)
            st.text_area(
                "AI-Formatted text (original):",
                value=st.session_state.formatted_direct_text,
                height=500,  # Increased height
                key="original_formatted_text",
                disabled=False,
            )

            # Initialize the user-editable version if it doesn't exist
            if "user_edited_direct_text" not in st.session_state:
                st.session_state.user_edited_direct_text = (
                    st.session_state.formatted_direct_text
                )

            # Editable version for user modifications
            user_edited_text = st.text_area(
                "Edit formatted text:",
                value=st.session_state.user_edited_direct_text,
                height=1200,  # Substantially increased height for better editing
                key="user_edited_direct_text_area",
            )

            # Store the user's edits
            st.session_state.user_edited_direct_text = user_edited_text

            # Copy button for user-edited text
            if st.button("Copy to Clipboard", key="text_copy"):
                pyperclip.copy(st.session_state.user_edited_direct_text)
                st.success("Edited text copied to clipboard!")

                # create memory, and save to db

                if "formatted_direct_text" in st.session_state:
                    original = st.session_state.formatted_direct_text
                    edited = st.session_state.user_edited_direct_text

                    if original == edited:
                        st.warning(
                            "Original and edited text are the same. No memory created."
                        )

                    else:
                        # Get existing memories through the helper function
                        user_memories = get_user_memories(user_email)

                        # Create a new memory using the AI
                        memory_result = create_memory_prompt(
                            client, original, edited, user_memories
                        )

                        # Check if there's a new memory to save
                        if (
                            memory_result
                            and "memory_to_write" in memory_result
                            and memory_result["memory_to_write"]
                        ):
                            # Save the new memory to the database
                            memory_text = memory_result["memory_to_write"]
                            save_success = save_memory(user_email, memory_text)

                            if save_success:
                                st.success(
                                    "New formatting preference saved: " + memory_text
                                )

                            # Add the new memory to the session state (at the beginning)
                            if "user_memories" in st.session_state:
                                st.session_state.user_memories.insert(0, memory_text)
                            else:
                                st.session_state.user_memories = [memory_text]

                            # Force refresh to update the sidebar
                            st.rerun()

            # Option to reset to original formatted text
            if st.button("Reset to Original", key="reset_text"):
                st.session_state.user_edited_direct_text = (
                    st.session_state.formatted_direct_text
                )
                st.rerun()

    # Logout button at the bottom of the page (outside the tabs)
    if st.button("Logout"):
        sign_out()


def auth_screen():
    st.title("🔐 Streamlit & Supabase Auth App")
    option = st.selectbox("Choose an action:", ["Login", "Sign Up"])
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if option == "Sign Up" and st.button("Register"):
        user = sign_up(email, password)
        if user and user.user:
            st.success("Registration successful. Please log in.")

    if option == "Login" and st.button("Login"):
        user = sign_in(email, password)
        if user and user.user:
            st.session_state.user_email = user.user.email
            st.success(f"Welcome back, {email}!")
            st.rerun()


if __name__ == "__main__":

    # if "user_email" not in st.session_state:
    #     st.session_state.user_email = None

    # if st.session_state.user_email:
    #     main_app(st.session_state.user_email)
    # else:
    #     auth_screen()

    main_app("zp4work@gmail.com")
