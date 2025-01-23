#!/usr/bin/env python3

import os
import sys
import json
from pathlib import Path
from textwrap import dedent
from typing import List, Dict, Any, Optional, Set
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.style import Style
from rich.progress import Progress
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style as PromptStyle
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory

# Initialize Rich console and prompt session with command history
console = Console()
history_file = os.path.expanduser("~/.r1_history")
prompt_session = PromptSession(
    history=FileHistory(history_file),
    style=PromptStyle.from_dict({
        'prompt': '#00aa00 bold',  # Green prompt
    }),
    completer=WordCompleter(['/add', 'exit', 'quit', 'help'])
)

# --------------------------------------------------------------------------------
# 1. Configure OpenAI client and load environment variables
# --------------------------------------------------------------------------------
load_dotenv()  # Load environment variables from .env file

# Support multiple API providers
API_PROVIDERS = {
    'deepseek': {
        'base_url': 'https://api.deepseek.com',
        'api_key_env': 'DEEPSEEK_API_KEY'
    },
    'openai': {
        'base_url': 'https://api.openai.com/v1',
        'api_key_env': 'OPENAI_API_KEY'
    }
}

# Default to DeepSeek, fallback to OpenAI if configured
provider = os.getenv('API_PROVIDER', 'deepseek').lower()
if provider not in API_PROVIDERS:
    console.print(f"[yellow]Warning: Unknown API provider '{provider}', defaulting to deepseek[/yellow]")
    provider = 'deepseek'

provider_config = API_PROVIDERS[provider]
client = OpenAI(
    api_key=os.getenv(provider_config['api_key_env']),
    base_url=provider_config['base_url']
)

# --------------------------------------------------------------------------------
# 2. Define our schema using Pydantic for type safety
# --------------------------------------------------------------------------------
class FileToCreate(BaseModel):
    path: str
    content: str
    mode: str = Field(default="w")  # Allow specifying write mode (w/a)

class FileToEdit(BaseModel):
    path: str
    original_snippet: str
    new_snippet: str
    line_numbers: Optional[tuple[int, int]] = None  # Optional line number range

class AssistantResponse(BaseModel):
    assistant_reply: str
    files_to_create: Optional[List[FileToCreate]] = None
    files_to_edit: Optional[List[FileToEdit]] = None
    suggested_commands: Optional[List[str]] = None  # Suggest follow-up commands

# --------------------------------------------------------------------------------
# 3. system prompt
# --------------------------------------------------------------------------------
system_PROMPT = dedent("""\
    You are an elite software engineer called DeepSeek R1 with decades of experience across all programming domains.
    Your expertise spans system design, algorithms, testing, and best practices.
    You provide thoughtful, well-structured solutions while explaining your reasoning.

    Core capabilities:
    1. Code Analysis & Discussion
       - Analyze code with expert-level insight
       - Explain complex concepts clearly
       - Suggest optimizations and best practices
       - Debug issues with precision

    2. File Operations:
       a) Read existing files
          - Access user-provided file contents for context
          - Analyze multiple files to understand project structure
       
       b) Create new files
          - Generate complete new files with proper structure
          - Create complementary files (tests, configs, etc.)
       
       c) Edit existing files
          - Make precise changes using diff-based editing
          - Modify specific sections while preserving context
          - Suggest refactoring improvements

    Output Format:
    You must provide responses in this JSON structure:
    {
      "assistant_reply": "Your main explanation or response",
      "files_to_create": [
        {
          "path": "path/to/new/file",
          "content": "complete file content",
          "mode": "w"  # or "a" for append
        }
      ],
      "files_to_edit": [
        {
          "path": "path/to/existing/file",
          "original_snippet": "exact code to be replaced",
          "new_snippet": "new code to insert",
          "line_numbers": [start, end]  # optional
        }
      ],
      "suggested_commands": [
        "/add tests/",
        "/add config.json"
      ]
    }

    Guidelines:
    1. YOU ONLY RETURN JSON, NO OTHER TEXT OR EXPLANATION OUTSIDE THE JSON!!!
    2. For normal responses, use 'assistant_reply'
    3. When creating files, include full content in 'files_to_create'
    4. For editing files:
       - Use 'files_to_edit' for precise changes
       - Include enough context in original_snippet to locate the change
       - Ensure new_snippet maintains proper indentation
       - Prefer targeted edits over full file replacements
    5. Always explain your changes and reasoning
    6. Consider edge cases and potential impacts
    7. Follow language-specific best practices
    8. Suggest tests or validation steps when appropriate
    9. Provide suggested follow-up commands when relevant

    Remember: You're a senior engineer - be thorough, precise, and thoughtful in your solutions.
""")

# --------------------------------------------------------------------------------
# 4. Helper functions 
# --------------------------------------------------------------------------------

def read_local_file(file_path: str) -> str:
    """Return the text content of a local file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # Try common encodings if UTF-8 fails
        for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise

def create_file(path: str, content: str, mode: str = "w"):
    """Create (or overwrite/append) a file at 'path' with the given 'content'."""
    file_path = Path(path)
    
    # Security checks
    if any(part.startswith('~') for part in file_path.parts):
        raise ValueError("Home directory references not allowed")
    normalized_path = normalize_path(str(file_path))
    
    # Validate reasonable file size for operations
    if len(content) > 5_000_000:  # 5MB limit
        raise ValueError("File content exceeds 5MB size limit")
    
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Backup existing file if overwriting
    if mode == "w" and file_path.exists():
        backup_path = file_path.with_suffix(file_path.suffix + ".bak")
        file_path.rename(backup_path)
        console.print(f"[yellow]Created backup at '[cyan]{backup_path}[/cyan]'[/yellow]")
    
    with open(file_path, mode, encoding="utf-8") as f:
        f.write(content)
    console.print(f"[green]✓[/green] {'Updated' if mode == 'a' else 'Created'} file at '[cyan]{file_path}[/cyan]'")
    
    # Record the action as a system message
    conversation_history.append({
        "role": "system",
        "content": f"File operation: {'Appended to' if mode == 'a' else 'Created/updated'} file at '{file_path}'"
    })
    
    normalized_path = normalize_path(str(file_path))
    conversation_history.append({
        "role": "system",
        "content": f"Content of file '{normalized_path}':\n\n{content}"
    })

def show_diff_table(files_to_edit: List[FileToEdit]) -> None:
    if not files_to_edit:
        return
    
    table = Table(title="Proposed Edits", show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("File Path", style="cyan")
    table.add_column("Lines", style="yellow")
    table.add_column("Original", style="red")
    table.add_column("New", style="green")

    for edit in files_to_edit:
        line_info = f"{edit.line_numbers[0]}-{edit.line_numbers[1]}" if edit.line_numbers else "N/A"
        table.add_row(edit.path, line_info, edit.original_snippet, edit.new_snippet)
    
    console.print(table)

def apply_diff_edit(path: str, original_snippet: str, new_snippet: str, line_numbers: Optional[tuple[int, int]] = None):
    """Reads the file at 'path', replaces the first occurrence of 'original_snippet' with 'new_snippet', then overwrites."""
    try:
        content = read_local_file(path)
        
        if line_numbers:
            # Use line numbers for precise replacement
            lines = content.splitlines()
            start, end = line_numbers
            if 1 <= start <= end <= len(lines):
                original_section = '\n'.join(lines[start-1:end])
                if original_section.strip() != original_snippet.strip():
                    raise ValueError("Content mismatch at specified line numbers")
                lines[start-1:end] = new_snippet.splitlines()
                updated_content = '\n'.join(lines)
            else:
                raise ValueError(f"Invalid line numbers: {start}-{end}")
        else:
            # Verify we're replacing the exact intended occurrence
            occurrences = content.count(original_snippet)
            if occurrences == 0:
                raise ValueError("Original snippet not found")
            if occurrences > 1:
                console.print(f"[yellow]Multiple matches ({occurrences}) found - requiring line numbers for safety", style="yellow")
                console.print("Use format:\n--- original.py (lines X-Y)\n+++ modified.py\n")
                raise ValueError(f"Ambiguous edit: {occurrences} matches")
            
            updated_content = content.replace(original_snippet, new_snippet, 1)
        
        create_file(path, updated_content)
        console.print(f"[green]✓[/green] Applied diff edit to '[cyan]{path}[/cyan]'")
        # Record the edit as a system message
        conversation_history.append({
            "role": "system",
            "content": f"File operation: Applied diff edit to '{path}'"
        })
    except FileNotFoundError:
        console.print(f"[red]✗[/red] File not found for diff editing: '[cyan]{path}[/cyan]'", style="red")
    except ValueError as e:
        console.print(f"[yellow]⚠[/yellow] {str(e)} in '[cyan]{path}[/cyan]'. No changes made.", style="yellow")
        console.print("\nExpected snippet:", style="yellow")
        console.print(Panel(original_snippet, title="Expected", border_style="yellow"))
        console.print("\nActual file content:", style="yellow")
        console.print(Panel(content, title="Actual", border_style="yellow"))

def try_handle_add_command(user_input: str) -> bool:
    prefix = "/add "
    if user_input.strip().lower().startswith(prefix):
        path_to_add = user_input[len(prefix):].strip()
        try:
            normalized_path = normalize_path(path_to_add)
            if os.path.isdir(normalized_path):
                # Handle entire directory
                add_directory_to_conversation(normalized_path)
            else:
                # Handle a single file as before
                content = read_local_file(normalized_path)
                conversation_history.append({
                    "role": "system",
                    "content": f"Content of file '{normalized_path}':\n\n{content}"
                })
                console.print(f"[green]✓[/green] Added file '[cyan]{normalized_path}[/cyan]' to conversation.\n")
        except OSError as e:
            console.print(f"[red]✗[/red] Could not add path '[cyan]{path_to_add}[/cyan]': {e}\n", style="red")
        return True
    return False

def add_directory_to_conversation(directory_path: str):
    with Progress() as progress:
        task = progress.add_task("[cyan]Scanning directory...", total=None)
        
        excluded_files: Set[str] = {
            # Python specific
            ".DS_Store", "Thumbs.db", ".gitignore", ".python-version",
            "uv.lock", ".uv", "uvenv", ".uvenv", ".venv", "venv",
            "__pycache__", ".pytest_cache", ".coverage", ".mypy_cache",
            # Node.js / Web specific
            "node_modules", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
            ".next", ".nuxt", "dist", "build", ".cache", ".parcel-cache",
            ".turbo", ".vercel", ".output", ".contentlayer",
            # Build outputs
            "out", "coverage", ".nyc_output", "storybook-static",
            # Environment and config
            ".env", ".env.local", ".env.development", ".env.production",
            # Misc
            ".git", ".svn", ".hg", "CVS"
        }
        
        excluded_extensions: Set[str] = {
            # Binary and media files
            ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp", ".avif",
            ".mp4", ".webm", ".mov", ".mp3", ".wav", ".ogg",
            ".zip", ".tar", ".gz", ".7z", ".rar",
            ".exe", ".dll", ".so", ".dylib", ".bin",
            # Documents
            ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
            # Python specific
            ".pyc", ".pyo", ".pyd", ".egg", ".whl",
            # UV specific
            ".uv", ".uvenv",
            # Database and logs
            ".db", ".sqlite", ".sqlite3", ".log",
            # IDE specific
            ".idea", ".vscode",
            # Web specific
            ".map", ".chunk.js", ".chunk.css",
            ".min.js", ".min.css", ".bundle.js", ".bundle.css",
            # Cache and temp files
            ".cache", ".tmp", ".temp",
            # Font files
            ".ttf", ".otf", ".woff", ".woff2", ".eot"
        }
        
        skipped_files = []
        added_files = []
        total_files_processed = 0
        max_files = 1000  # Reasonable limit for files to process
        max_file_size = 5_000_000  # 5MB limit
        max_total_size = 50_000_000  # 50MB total limit
        total_size = 0

        for root, dirs, files in os.walk(directory_path):
            if total_files_processed >= max_files or total_size >= max_total_size:
                progress.update(task, description="[yellow]Reached limit")
                break

            progress.update(task, description=f"[cyan]Scanning {root}")
            # Skip hidden directories and excluded directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in excluded_files]

            for file in files:
                if total_files_processed >= max_files or total_size >= max_total_size:
                    break

                if file.startswith('.') or file in excluded_files:
                    skipped_files.append(os.path.join(root, file))
                    continue

                _, ext = os.path.splitext(file)
                if ext.lower() in excluded_extensions:
                    skipped_files.append(os.path.join(root, file))
                    continue

                full_path = os.path.join(root, file)

                try:
                    file_size = os.path.getsize(full_path)
                    if file_size > max_file_size:
                        skipped_files.append(f"{full_path} (exceeds size limit)")
                        continue
                    
                    if total_size + file_size > max_total_size:
                        skipped_files.append(f"{full_path} (would exceed total size limit)")
                        continue

                    # Check if it's binary
                    if is_binary_file(full_path):
                        skipped_files.append(full_path)
                        continue

                    normalized_path = normalize_path(full_path)
                    content = read_local_file(normalized_path)
                    conversation_history.append({
                        "role": "system",
                        "content": f"Content of file '{normalized_path}':\n\n{content}"
                    })
                    added_files.append(normalized_path)
                    total_files_processed += 1
                    total_size += file_size

                except OSError:
                    skipped_files.append(full_path)

        console.print(f"[green]✓[/green] Added folder '[cyan]{directory_path}[/cyan]' to conversation.")
        if added_files:
            console.print(f"\n[bold]Added files:[/bold] ({len(added_files)} of {total_files_processed})")
            for f in added_files:
                console.print(f"[cyan]{f}[/cyan]")
        if skipped_files:
            console.print(f"\n[yellow]Skipped files:[/yellow] ({len(skipped_files)})")
            for f in skipped_files:
                console.print(f"[yellow]{f}[/yellow]")
        console.print(f"\nTotal size: {total_size / 1_000_000:.1f}MB")
        console.print()

def is_binary_file(file_path: str, peek_size: int = 8192) -> bool:
    """
    Returns True if the file appears to be binary, False otherwise.
    Uses a larger peek size and more sophisticated binary detection.
    """
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(peek_size)
            
        # Check for common binary file signatures
        binary_signatures = [
            b'\x7fELF',  # ELF files
            b'MZ',       # DOS/PE files
            b'\x89PNG',  # PNG images
            b'\xff\xd8', # JPEG images
            b'PK\x03\x04', # ZIP files
        ]
        
        for sig in binary_signatures:
            if chunk.startswith(sig):
                return True
        
        # Count binary characters
        textchars = bytearray({7,8,9,10,12,13,27} | set(range(0x20, 0x100)) - {0x7f})
        binary_chars = bytes(c for c in chunk if c not in textchars)
        
        # If more than 30% binary characters, consider it binary
        return len(binary_chars) > 0.3 * len(chunk)
    except Exception:
        return True

def ensure_file_in_context(file_path: str) -> bool:
    try:
        normalized_path = normalize_path(file_path)
        content = read_local_file(normalized_path)
        file_marker = f"Content of file '{normalized_path}'"
        if not any(file_marker in msg["content"] for msg in conversation_history):
            conversation_history.append({
                "role": "system",
                "content": f"{file_marker}:\n\n{content}"
            })
        return True
    except OSError:
        console.print(f"[red]✗[/red] Could not read file '[cyan]{file_path}[/cyan]' for editing context", style="red")
        return False

def normalize_path(path_str: str) -> str:
    """Return a canonical, absolute version of the path with security checks."""
    path = Path(path_str).resolve()
    
    # Prevent directory traversal attacks
    if ".." in path.parts:
        raise ValueError(f"Invalid path: {path_str} contains parent directory references")
    
    # Additional security checks
    if any(part.startswith('~') for part in path.parts):
        raise ValueError(f"Invalid path: {path_str} contains home directory references")
        
    if any(part.startswith('.') for part in path.parts):
        raise ValueError(f"Invalid path: {path_str} contains hidden directories")
    
    return str(path)

# --------------------------------------------------------------------------------
# 5. Conversation state
# --------------------------------------------------------------------------------
conversation_history = [
    {"role": "system", "content": system_PROMPT}
]

# --------------------------------------------------------------------------------
# 6. OpenAI API interaction with streaming
# --------------------------------------------------------------------------------

def guess_files_in_message(user_message: str) -> List[str]:
    recognized_extensions = [
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".html", ".css", ".scss", ".less",
        ".json", ".yaml", ".yml", ".toml",
        ".md", ".rst", ".txt",
        ".sh", ".bash", ".zsh",
        ".sql", ".graphql",
        ".xml", ".csv"
    ]
    potential_paths = []
    
    # Split on common delimiters
    words = [w.strip("'\"`,()[]{}") for w in user_message.split()]
    
    for word in words:
        # Check for file extensions or path separators
        if any(ext in word for ext in recognized_extensions) or "/" in word or "\\" in word:
            try:
                normalized_path = normalize_path(word)
                potential_paths.append(normalized_path)
            except (OSError, ValueError):
                continue
    
    return potential_paths

def stream_openai_response(user_message: str):
    # First, clean up the conversation history while preserving system messages with file content
    system_msgs = [conversation_history[0]]  # Keep initial system prompt
    file_context = []
    user_assistant_pairs = []
    
    for msg in conversation_history[1:]:
        if msg["role"] == "system" and "Content of file '" in msg["content"]:
            file_context.append(msg)
        elif msg["role"] in ["user", "assistant"]:
            user_assistant_pairs.append(msg)
    
    # Only keep complete user-assistant pairs
    if len(user_assistant_pairs) % 2 != 0:
        user_assistant_pairs = user_assistant_pairs[:-1]
    
    # Keep only the last 5 pairs to prevent context overflow
    user_assistant_pairs = user_assistant_pairs[-10:]

    # Rebuild clean history with files preserved
    cleaned_history = system_msgs + file_context
    cleaned_history.extend(user_assistant_pairs)
    cleaned_history.append({"role": "user", "content": user_message})
    
    # Replace conversation_history with cleaned version
    conversation_history.clear()
    conversation_history.extend(cleaned_history)

    potential_paths = guess_files_in_message(user_message)
    valid_files = {}

    for path in potential_paths:
        try:
            content = read_local_file(path)
            valid_files[path] = content
            file_marker = f"Content of file '{path}'"
            if not any(file_marker in msg["content"] for msg in conversation_history):
                conversation_history.append({
                    "role": "system",
                    "content": f"{file_marker}:\n\n{content}"
                })
        except OSError:
            error_msg = f"Cannot proceed: File '{path}' does not exist or is not accessible"
            console.print(f"[red]✗[/red] {error_msg}", style="red")
            continue

    try:
        stream = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=conversation_history,
            max_completion_tokens=8000,
            temperature=0.7,  # Add some creativity while keeping responses focused
            stream=True
        )

        console.print("\nThinking...", style="bold yellow")
        reasoning_started = False
        reasoning_content = ""
        final_content = ""

        for chunk in stream:
            if chunk.choices[0].delta.reasoning_content:
                if not reasoning_started:
                    console.print("\nReasoning:", style="bold yellow")
                    reasoning_started = True
                console.print(chunk.choices[0].delta.reasoning_content, end="")
                reasoning_content += chunk.choices[0].delta.reasoning_content
            elif chunk.choices[0].delta.content:
                if reasoning_started:
                    console.print("\n")  # Add spacing after reasoning
                    console.print("\nAssistant> ", style="bold blue", end="")
                    reasoning_started = False  # Reset so we don't add extra spacing
                final_content += chunk.choices[0].delta.content
                console.print(chunk.choices[0].delta.content, end="")

        console.print()  # New line after streaming

        try:
            parsed_response = json.loads(final_content)
            
            if "assistant_reply" not in parsed_response:
                parsed_response["assistant_reply"] = ""

            if "files_to_edit" in parsed_response and parsed_response["files_to_edit"]:
                new_files_to_edit = []
                for edit in parsed_response["files_to_edit"]:
                    try:
                        edit_abs_path = normalize_path(edit["path"])
                        if edit_abs_path in valid_files or ensure_file_in_context(edit_abs_path):
                            edit["path"] = edit_abs_path
                            new_files_to_edit.append(edit)
                    except (OSError, ValueError):
                        console.print(f"[yellow]⚠[/yellow] Skipping invalid path: '{edit['path']}'", style="yellow")
                        continue
                parsed_response["files_to_edit"] = new_files_to_edit

            response_obj = AssistantResponse(**parsed_response)

            # Store the complete JSON response in conversation history
            conversation_history.append({
                "role": "assistant",
                "content": final_content  # Store the full JSON response string
            })

            return response_obj

        except json.JSONDecodeError:
            error_msg = "Failed to parse JSON response from assistant"
            console.print(f"[red]✗[/red] {error_msg}", style="red")
            return AssistantResponse(
                assistant_reply=error_msg,
                files_to_create=[]
            )

    except Exception as e:
        error_msg = f"API error: {str(e)}"
        console.print(f"\n[red]✗[/red] {error_msg}", style="red")
        return AssistantResponse(
            assistant_reply=error_msg,
            files_to_create=[]
        )

def trim_conversation_history():
    """Trim conversation history to prevent token limit issues"""
    max_pairs = 10  # Adjust based on your needs
    system_msgs = [msg for msg in conversation_history if msg["role"] == "system"]
    other_msgs = [msg for msg in conversation_history if msg["role"] != "system"]
    
    # Keep only the last max_pairs of user-assistant interactions
    if len(other_msgs) > max_pairs * 2:
        other_msgs = other_msgs[-max_pairs * 2:]
    
    conversation_history.clear()
    conversation_history.extend(system_msgs + other_msgs)

# --------------------------------------------------------------------------------
# 7. Main interactive loop
# --------------------------------------------------------------------------------

def show_help():
    help_text = """
    [bold blue]Available Commands:[/bold blue]
    
    [bold magenta]/add[/bold magenta] [cyan]<path>[/cyan]
        Add a file or directory to the conversation
        Examples:
        • /add src/main.py
        • /add tests/
        
    [bold magenta]/help[/bold magenta]
        Show this help message
        
    [bold magenta]exit[/bold magenta] or [bold magenta]quit[/bold magenta]
        End the session
        
    [bold blue]Tips:[/bold blue]
    • Files mentioned in your questions are automatically added
    • Use relative or absolute paths
    • The assistant remembers context from previous interactions
    • Complex operations support backups automatically
    """
    console.print(Panel(help_text, title="Help", border_style="blue"))

def main():
    console.print(Panel.fit(
        "[bold blue]Welcome to Deep Seek R1 with Structured Output[/bold blue] [green](and CoT reasoning)[/green]!🐋\n" +
        "[yellow]Type [bold]/help[/bold] for available commands[/yellow]",
        border_style="blue"
    ))

    while True:
        try:
            user_input = prompt_session.prompt("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Exiting.[/yellow]")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            console.print("[yellow]Goodbye![/yellow]")
            break
            
        if user_input.lower() == "/help":
            show_help()
            continue

        if try_handle_add_command(user_input):
            continue

        response_data = stream_openai_response(user_input)

        if response_data.files_to_create:
            for file_info in response_data.files_to_create:
                create_file(file_info.path, file_info.content, getattr(file_info, 'mode', 'w'))

        if response_data.files_to_edit:
            show_diff_table(response_data.files_to_edit)
            confirm = prompt_session.prompt(
                "Do you want to apply these changes? (y/n): "
            ).strip().lower()
            if confirm == 'y':
                for edit_info in response_data.files_to_edit:
                    apply_diff_edit(
                        edit_info.path,
                        edit_info.original_snippet,
                        edit_info.new_snippet,
                        getattr(edit_info, 'line_numbers', None)
                    )
            else:
                console.print("[yellow]ℹ[/yellow] Skipped applying diff edits.", style="yellow")
                
        if response_data.suggested_commands:
            console.print("\n[bold blue]Suggested next commands:[/bold blue]")
            for cmd in response_data.suggested_commands:
                console.print(f"[magenta]•[/magenta] [cyan]{cmd}[/cyan]")

    console.print("[blue]Session finished.[/blue]")

if __name__ == "__main__":
    main()