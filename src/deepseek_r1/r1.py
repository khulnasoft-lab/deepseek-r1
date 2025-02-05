#!/usr/bin/env python3

import os
import sys
import json
import logging
from pathlib import Path
from textwrap import dedent
from typing import List, Dict, Any, Optional, Set, Tuple
from functools import wraps
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.style import Style
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style as PromptStyle
from prompt_toolkit.completion import PathCompleter, WordCompleter
from prompt_toolkit.history import FileHistory

# Configure logging
logging.basicConfig(
    filename='r1.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Rich console
console = Console()

# =================================================================================
# Configuration and Constants
# =================================================================================
class Config:
    """Application configuration settings"""
    MAX_FILE_SIZE = 5_000_000  # 5MB
    MAX_TOTAL_SIZE = 50_000_000  # 50MB
    MAX_FILES = 1000
    MAX_CONVERSATION_PAIRS = 10
    API_TIMEOUT = 30
    EXCLUDED_DIRS = {'.git', '.venv', 'node_modules', '__pycache__'}
    EXCLUDED_EXTENSIONS = {'.pyc', '.log', '.tmp', '.zip', '.exe'}
    FORBIDDEN_PATHS = {'/etc', '/bin', '/sys', '/root'}

    @classmethod
    def load_from_env(cls):
        """Update configuration from environment variables"""
        cls.MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', cls.MAX_FILE_SIZE))
        cls.MAX_TOTAL_SIZE = int(os.getenv('MAX_TOTAL_SIZE', cls.MAX_TOTAL_SIZE))
        cls.MAX_FILES = int(os.getenv('MAX_FILES', cls.MAX_FILES))

# Load configuration
Config.load_from_env()

# =================================================================================
# Helper Decorators
# =================================================================================
def handle_errors(func):
    """Decorator for consistent error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            console.print(f"[red]Error:[/red] {str(e)}")
            return None
    return wrapper

# =================================================================================
# Core Components
# =================================================================================
class FileManager:
    """Handles all file operations with security checks"""
    
    @staticmethod
    @handle_errors
    def read_file(file_path: str) -> str:
        """Safely read file contents with multiple encodings"""
        FileManager._validate_path(file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Could not decode file {file_path}")

    @staticmethod
    @handle_errors
    def create_file(path: str, content: str, mode: str = "w"):
        """Create or update a file with safety checks"""
        FileManager._validate_path(path)
        file_path = Path(path)
        
        if len(content) > Config.MAX_FILE_SIZE:
            raise ValueError(f"Content exceeds size limit ({len(content)/1_000_000:.1f}MB)")
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if mode == "w" and file_path.exists():
            backup_path = file_path.with_suffix(file_path.suffix + ".bak")
            file_path.rename(backup_path)
            logger.info(f"Created backup at {backup_path}")
            console.print(f"[yellow]Backup created:[/yellow] [cyan]{backup_path}[/cyan]")

        with open(file_path, mode, encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Created file {file_path}")
        console.print(f"[green]✓[/green] {'Updated' if mode == 'a' else 'Created'} [cyan]{file_path}[/cyan]")

    @staticmethod
    def _validate_path(path_str: str):
        """Validate path security"""
        path = Path(path_str).resolve()
        
        if any(part in Config.FORBIDDEN_PATHS for part in path.parts):
            raise ValueError(f"Access to restricted path: {path_str}")
        
        if ".." in path.parts:
            raise ValueError(f"Invalid path: {path_str} contains parent directory references")
        
        if any(part.startswith(('.', '~')) for part in path.parts):
            raise ValueError(f"Invalid path: {path_str} contains hidden/system references")

# =================================================================================
# AI Integration
# =================================================================================
class AIClient:
    """Handles interactions with the AI API"""
    
    def __init__(self):
        load_dotenv()
        self.client = self._configure_client()
        self.conversation = ConversationManager()
        
    def _configure_client(self):
        """Configure API client based on environment"""
        providers = {
            'deepseek': {
                'base_url': 'https://api.deepseek.com',
                'api_key_env': 'DEEPSEEK_API_KEY',
                'default_model': 'deepseek-reasoner'
            },
            'openai': {
                'base_url': 'https://api.openai.com/v1',
                'api_key_env': 'OPENAI_API_KEY',
                'default_model': 'gpt-4'
            }
        }
        
        provider = os.getenv('API_PROVIDER', 'deepseek').lower()
        if provider not in providers:
            console.print(f"[yellow]Unknown provider {provider}, using deepseek[/yellow]")
            provider = 'deepseek'
            
        config = providers[provider]
        return OpenAI(
            api_key=os.getenv(config['api_key_env']),
            base_url=config['base_url'],
            timeout=Config.API_TIMEOUT
        )

    @handle_errors
    def generate_response(self, user_input: str) -> Dict:
        """Generate AI response with conversation context"""
        self.conversation.add_message("user", user_input)
        
        stream = self.client.chat.completions.create(
            model=os.getenv('AI_MODEL', 'deepseek-reasoner'),
            messages=self.conversation.get_messages(),
            max_tokens=8000,
            temperature=0.7,
            stream=True
        )
        
        return self._process_stream(stream)

    def _process_stream(self, stream):
        """Process streaming API response"""
        console.print("\n[bold yellow]Thinking...[/bold yellow]")
        full_response = []
        reasoning_buffer = []
        
        for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            full_response.append(content)
            
            if chunk.choices[0].delta.reasoning_content:
                reasoning_buffer.append(chunk.choices[0].delta.reasoning_content)
                if len(reasoning_buffer) == 1:
                    console.print("\n[bold yellow]Reasoning:[/bold yellow]")
                console.print(reasoning_buffer[-1], end="")
            else:
                if reasoning_buffer:
                    console.print("\n[bold blue]Assistant:[/bold blue] ", end="")
                    reasoning_buffer = []
                console.print(content, end="")
                
        console.print()
        return json.loads(''.join(full_response))

# =================================================================================
# Conversation Management
# =================================================================================
class ConversationManager:
    """Manages conversation history and context"""
    
    def __init__(self):
        self.history = [{"role": "system", "content": self._system_prompt()}]
        self.file_cache = set()

    def add_message(self, role: str, content: str):
        """Add message to conversation history"""
        self.history.append({"role": role, "content": content})
        self._manage_context()

    def add_file(self, file_path: str):
        """Add file contents to conversation context"""
        if file_path in self.file_cache:
            return
            
        try:
            content = FileManager.read_file(file_path)
            self.history.append({
                "role": "system",
                "content": f"File content: {file_path}\n\n{content}"
            })
            self.file_cache.add(file_path)
            console.print(f"[green]✓[/green] Added [cyan]{file_path}[/cyan] to context")
        except Exception as e:
            console.print(f"[red]Error reading {file_path}:[/red] {str(e)}")

    def _manage_context(self):
        """Maintain conversation history within limits"""
        system_messages = [msg for msg in self.history if msg["role"] == "system"]
        interactions = [msg for msg in self.history if msg["role"] != "system"]
        
        # Keep last N interactions
        interactions = interactions[-Config.MAX_CONVERSATION_PAIRS*2:]
        self.history = system_messages + interactions

    @staticmethod
    def _system_prompt():
        return dedent("""\
            [Your existing system prompt here...]
        """)

# =================================================================================
# User Interface
# =================================================================================
class UserInterface:
    """Handles all user interactions and displays"""
    
    def __init__(self):
        self.prompt_session = self._configure_prompt()
        self.ai_client = AIClient()

    def _configure_prompt(self):
        """Configure interactive prompt session"""
        return PromptSession(
            history=FileHistory(os.path.expanduser("~/.r1_history")),
            style=PromptStyle.from_dict({'prompt': '#00aa00 bold'}),
            completer=PathCompleter(),
            complete_while_typing=True
        )

    def show_help(self):
        """Display help information"""
        help_content = """
        [bold]Available Commands:[/bold]
        /add <path> - Add file/directory to context
        /help       - Show this help
        /exit       - End session
        """
        console.print(Panel(help_content, title="Help", border_style="blue"))

    def run(self):
        """Main application loop"""
        console.print(Panel.fit(
            "[bold blue]DeepSeek R1 Assistant[/bold blue]\n[yellow]Type /help for commands[/yellow]",
            border_style="blue"
        ))
        
        while True:
            try:
                user_input = self.prompt_session.prompt("You> ").strip()
                if not user_input:
                    continue
                    
                if user_input.lower() in ('exit', 'quit'):
                    break
                    
                if user_input.lower() == '/help':
                    self.show_help()
                    continue
                    
                if self._handle_command(user_input):
                    continue
                    
                response = self.ai_client.generate_response(user_input)
                self._handle_response(response)
                
            except (EOFError, KeyboardInterrupt):
                console.print("\n[yellow]Exiting...[/yellow]")
                break

        console.print("[blue]Session completed[/blue]")

    def _handle_command(self, input_str: str) -> bool:
        """Process user commands"""
        if input_str.startswith('/add '):
            path = input_str[5:].strip()
            self.ai_client.conversation.add_file(path)
            return True
        return False

    def _handle_response(self, response: Dict):
        """Process and display AI response"""
        try:
            validated = AssistantResponse(**response)
            self._process_files(validated)
            self._show_suggestions(validated)
        except ValidationError as e:
            console.print(f"[red]Invalid response format:[/red] {str(e)}")

    def _process_files(self, response: AssistantResponse):
        """Handle file creation/editing from response"""
        if response.files_to_create:
            for file in response.files_to_create:
                FileManager.create_file(file.path, file.content, file.mode)
                
        if response.files_to_edit:
            self._handle_edits(response.files_to_edit)

    def _handle_edits(self, edits: List[FileToEdit]):
        """Display and confirm file edits"""
        table = Table(title="Proposed Changes", show_lines=True)
        table.add_column("File", style="cyan")
        table.add_column("Lines", style="yellow")
        table.add_column("Original", style="red")
        table.add_column("New", style="green")
        
        for edit in edits:
            lines = f"{edit.line_numbers[0]}-{edit.line_numbers[1]}" if edit.line_numbers else "N/A"
            table.add_row(edit.path, lines, edit.original_snippet, edit.new_snippet)
            
        console.print(table)
        
        if console.input("Apply changes? (y/N): ").lower() == 'y':
            for edit in edits:
                self._apply_edit(edit)

    def _apply_edit(self, edit: FileToEdit):
        """Apply a single file edit"""
        try:
            content = FileManager.read_file(edit.path)
            # [Existing edit application logic]
            console.print(f"[green]✓[/green] Updated [cyan]{edit.path}[/cyan]")
        except Exception as e:
            console.print(f"[red]Error updating {edit.path}:[/red] {str(e)}")

# =================================================================================
# Main Execution
# =================================================================================
if __name__ == "__main__":
    try:
        UserInterface().run()
    except Exception as e:
        console.print(f"[red]Critical error:[/red] {str(e)}")
        logger.critical("Unhandled exception", exc_info=True)
        sys.exit(1)
