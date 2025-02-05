# DeepSeek R1 🐋

## Overview

This repository contains a powerful coding assistant application that integrates with the DeepSeek API to process user conversations and generate structured JSON responses. Through an intuitive command-line interface, it can read local file contents, create new files, and apply diff edits to existing files in real time.

## Key Features

1. DeepSeek Client Configuration
   - Automatically configures an API client to use the DeepSeek service with a valid DEEPSEEK_API_KEY
   - Connects to the DeepSeek endpoint specified in the environment variable to stream GPT-like completions

2. Data Models
   - Leverages Pydantic for type-safe handling of file operations:
     - FileToCreate: Defines structure for new files to be created
     - FileToEdit: Specifies changes to be made to existing files
     - AssistantResponse: Structures chat responses and file operations

3. System Prompt
   - Implements a comprehensive system prompt that guides conversation flow
   - Ensures all responses follow strict JSON formatting
   - Supports both file creation and editing operations

4. File Operations
   - read_local_file(): Reads and returns file contents as string
   - create_file(): Creates or updates files with provided content
   - show_diff_table(): Displays proposed changes in a rich format
   - apply_diff_edit(): Implements precise file modifications

5. File Integration Commands
   - "/add path/to/file": Reads and includes single file content
   - "/add path/to/folder": Adds all non-binary, non-hidden files from directory
   - Enables context-aware assistance based on your codebase

6. Conversation Management
   - Maintains conversation history for context awareness
   - Streams responses via DeepSeek API in real-time
   - Parses responses as JSON for structured operations

7. Interactive CLI
   - Simple command-line interface
   - Real-time file modification previews
   - Easy confirmation of suggested changes
   - Exit with "exit" or "quit" commands

## Installation

1. Configure Environment:
   ```bash
   # Create .env file with API key
   echo "DEEPSEEK_API_KEY=your_api_key_here" > .env
   ```

2. Install & Run:

   Using pip:
   ```bash
   pip install -r requirements.txt
   python3 main.py
   ```

   Using uv (recommended for speed):
   ```bash
   uv venv
   uv run main.py
   ```

## Advanced Features: Reasoning Model

The `r1.py` script provides enhanced capabilities using DeepSeek's Reasoning Model:

- Chain of Thought (CoT) reasoning visualization
- Transparent decision-making process
- Full compatibility with base features
- Dedicated reasoning display panel
- Clean conversation history

Usage:
```bash
python3 src/deepseek_r1/r1.py
```
