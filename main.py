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


def get_user_memories(user_email, force_refresh=False):
    """Helper function to get user memories from session state or database.

    Args:
        user_email: The email of the user
        force_refresh: Force a refresh from the database

    Returns:
        List of user memory strings
    """
    # Force refresh from database if requested or if memories don't exist in session state
    if force_refresh or "user_memories" not in st.session_state:
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
    if "memory_refreshed" not in st.session_state:
        st.session_state.memory_refreshed = False

    # Only load memories from the database once on page load or when explicitly refreshed
    if not st.session_state.memory_refreshed:
        # Get user memories once
        user_memories = get_user_memories(user_email)
        st.session_state.memory_refreshed = True
    else:
        # Use cached memories from session state
        user_memories = (
            st.session_state.user_memories
            if "user_memories" in st.session_state
            else []
        )

    # Display user memories in the sidebar
    if user_memories:
        with st.sidebar.expander(
            f"Your Formatting Preferences ({len(user_memories)})", expanded=True
        ):
            for i, mem in enumerate(user_memories):
                st.write(f"{i+1}. {mem}")
    else:
        st.sidebar.info(
            "No formatting preferences saved yet. Use the Memory tab to create preferences."
        )

    # Create tabs
    tab_names = ["🎙️ Transcribe", "✏️ Format", "🧠 Memory"]
    tabs = st.tabs(tab_names)
    transcribe_tab = tabs[0]
    format_tab = tabs[1]
    memory_tab = tabs[2]

    # Tab 1: Audio Recording and Transcription Only
    with transcribe_tab:
        st.subheader("Audio Transcription")
        st.write(
            "Record audio to transcribe it. The transcript can be copied but not formatted in this tab."
        )

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

        # Display transcript
        if st.session_state.transcript:
            st.subheader("Transcript")
            st.text_area(
                "Raw Transcript:",
                value=st.session_state.transcript,
                height=300,
                key="transcript_text",
            )

            col1, col2 = st.columns(2)
            # Copy button for raw transcript
            with col1:
                if st.button("Copy to Clipboard", key="audio_copy"):
                    pyperclip.copy(st.session_state.transcript)
                    st.success("Transcript copied to clipboard!")

            # Clear button
            with col2:
                if st.button("Clear Transcript", key="clear_transcript"):
                    st.session_state.transcript = ""
                    st.rerun()
        else:
            st.info("Record audio to see the transcript here.")

    # Tab 2: Text Formatting Only
    with format_tab:
        st.subheader("Text Formatting")
        st.write("Enter medical text to format it according to your saved preferences.")

        # Input text area for text to format
        text_to_format_input = st.text_area(
            "Enter text to format:",
            height=250,
            key="format_input_area",
            placeholder="Paste your medical transcript here for formatting...",
        )

        # Format button
        if st.button("Format Text", key="format_button"):
            if text_to_format_input.strip():
                with st.spinner("Formatting text..."):
                    # Use the memories from session state to avoid unnecessary database calls
                    formatted_result = text_to_format(
                        client,
                        text_to_format_input,
                        memories=(
                            st.session_state.user_memories
                            if "user_memories" in st.session_state
                            else []
                        ),
                    )
                    st.session_state.formatted_result = formatted_result
            else:
                st.warning("Please enter some text to format.")

        # Display formatted result
        if "formatted_result" in st.session_state and st.session_state.formatted_result:
            st.subheader("Formatted Result")

            formatted_text = st.text_area(
                "Formatted text:",
                value=st.session_state.formatted_result,
                height=500,
                key="formatted_result_area",
            )

            # Copy button for formatted text
            if st.button("Copy Formatted Text", key="copy_formatted"):
                pyperclip.copy(st.session_state.formatted_result)
                st.success("Formatted text copied to clipboard!")

            # Clear button
            if st.button("Clear All", key="clear_format"):
                if "formatted_result" in st.session_state:
                    st.session_state.formatted_result = ""
                st.rerun()

    # Tab 3: Memory Creation
    with memory_tab:
        st.subheader("Create Formatting Preferences")
        st.write(
            "This tab allows you to create memories (formatting preferences) from differences between original and edited text."
        )

        with st.expander("How Memory Creation Works", expanded=False):
            st.markdown(
                """
            ### Memory Creation Process
            1. Enter two versions of the same text: the original AI-formatted version and your edited version
            2. The system analyzes the differences between them
            3. If it identifies a pattern in your edits, it will create a "memory" of your preference
            4. This preference will be applied to future formatting tasks
            """
            )

        # Initialize session state for memory text inputs if they don't exist
        if "memory_original_text" not in st.session_state:
            st.session_state.memory_original_text = ""
        if "memory_edited_text" not in st.session_state:
            st.session_state.memory_edited_text = ""

        # Original AI text
        original_text = st.text_area(
            "Original AI-Formatted Text:",
            value=st.session_state.memory_original_text,
            height=250,
            key="memory_original_text_area",
            placeholder="Paste the original AI-formatted text here...",
        )
        st.session_state.memory_original_text = original_text

        # User-edited version
        edited_text = st.text_area(
            "Your Edited Version:",
            value=st.session_state.memory_edited_text,
            height=250,
            key="memory_edited_text_area",
            placeholder="Paste your edited version here (with your preferred formatting)...",
        )
        st.session_state.memory_edited_text = edited_text

        # Sample text buttons for quick population
        st.write("Need an example? Try one of these:")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Sample Original (Table Format)", key="sample_original"):
                sample_original = """### Clinical Note — Formatted as a Table
| Section | Content |
|---------|----------|
| Patient Info | Jane Doe, 45yo female |
| Vital Signs | BP 120/80, HR 72, Temp 98.6°F |
| Assessment | 1. Hypertension - controlled\n2. Type 2 Diabetes - uncontrolled |
| Plan | Continue current medications, follow-up in 2 weeks |"""
                st.session_state.memory_original_text = sample_original
                st.rerun()

        with col2:
            if st.button("Sample Edited (Narrative)", key="sample_edited"):
                sample_edited = """```
### Clinical Encounter Report

PATIENT: Jane Doe, 45yo female

VITAL SIGNS:
- BP: 120/80
- HR: 72
- Temperature: 98.6°F

ASSESSMENT:
1. Hypertension - controlled
2. Type 2 Diabetes - uncontrolled

PLAN:
- Continue current medications
- Follow-up in 2 weeks
```"""
                st.session_state.memory_edited_text = sample_edited
                st.rerun()

        # Create Memory button
        if st.button("Create Memory", key="create_memory"):
            if original_text.strip() and edited_text.strip():
                if original_text == edited_text:
                    st.warning(
                        "Original and edited texts are identical. Please make edits to create a memory."
                    )
                else:
                    with st.spinner("Analyzing differences and creating memory..."):
                        # Use cached memories from session state
                        current_memories = (
                            st.session_state.user_memories
                            if "user_memories" in st.session_state
                            else []
                        )

                        # Create a new memory
                        memory_result = create_memory_prompt(
                            client, original_text, edited_text, current_memories
                        )

                        if (
                            memory_result
                            and "memory_to_write" in memory_result
                            and memory_result["memory_to_write"]
                        ):
                            memory_text = memory_result["memory_to_write"]
                            save_success = save_memory(user_email, memory_text)

                            if save_success:
                                st.success(f"**New preference saved:** {memory_text}")
                                # Add the new memory to the session state
                                if "user_memories" in st.session_state:
                                    # Add to beginning of list for visibility
                                    st.session_state.user_memories.insert(
                                        0, memory_text
                                    )
                                else:
                                    st.session_state.user_memories = [memory_text]
                                # Force refresh the UI
                                st.session_state.memory_refreshed = True
                                st.rerun()
                            else:
                                st.error("Failed to save preference to database.")
                        else:
                            st.info(
                                "No meaningful formatting preference detected in your edits."
                            )
            else:
                st.warning(
                    "Please provide both the original text and your edited version."
                )

        # Display current memories with delete option
        st.subheader("Your Saved Preferences")
        # Use cached memories from session state instead of querying again
        current_preferences = (
            st.session_state.user_memories
            if "user_memories" in st.session_state
            else []
        )

        if current_preferences:
            for i, memory in enumerate(current_preferences):
                col1, col2 = st.columns([10, 1])
                with col1:
                    st.write(f"{i+1}. {memory}")

            # Option to refresh memories from database
            if st.button("Refresh Preferences", key="refresh_memories"):
                # Force refresh from database
                st.session_state.user_memories = get_memories(user_email)
                st.session_state.memory_refreshed = True
                st.success("Preferences refreshed from database")
                st.rerun()

            # Option to clear all memories
            if st.button("Clear All Preferences", key="clear_all_memories"):
                # Not implementing actual deletion here - would need additional functionality
                st.warning(
                    "This would delete all your saved preferences (not implemented in this demo)."
                )
        else:
            st.info(
                "You don't have any saved preferences yet. Create some using the form above."
            )

    # Sidebar logout button
    st.sidebar.button("Sign Out", on_click=sign_out, key="sign_out_button")


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

    if "user_email" not in st.session_state:
        st.session_state.user_email = None

    if st.session_state.user_email:
        main_app(st.session_state.user_email)
    else:
        auth_screen()
