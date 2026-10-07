from __future__ import annotations

import typer
from rich.console import Console

from agent.config import get_settings
from agent.runner import AgentRunner
from agent.voice import (
    VoiceListener,
    VoiceSpeaker,
    list_recognizers,
    list_voices,
    microphone_diagnostic,
    recognizer_diagnostic,
)
from agent.web_app import run_web_app


app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def chat(
    skill: str = typer.Option("", "--skill", help="Start the chat with a specific skill."),
) -> None:
    """Start an interactive agent session."""
    settings = get_settings()
    runner = AgentRunner(settings)
    speaker = VoiceSpeaker(settings)
    listener = VoiceListener(settings)
    if skill.strip():
        try:
            runner.set_skill(skill)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1)

    console.print(f"[bold green]{settings.agent_name}[/bold green] is ready.")
    console.print(f"Provider: {settings.provider} | Model: {settings.model}")
    console.print(
        "Commands: /exit, /reset, /save, /where, /voice on, /voice off, "
        "/voices, /listen, /mic on, /mic off, /mic-check, /recognizers, /code-help, "
        "/skills, /skill <name|off>, /fast on, /fast off"
    )
    console.print(f"Active skill: {runner.active_skill.name if runner.active_skill else 'none'}")
    console.print(runner.fast_reply_summary())
    if listener.enabled:
        console.print("[yellow]Microphone input is active. Speak after Listening appears.[/yellow]")

    while True:
        if listener.enabled:
            user_input = _listen_for_input(listener)
            if not user_input:
                continue
        else:
            user_input = console.input("[bold cyan]You[/bold cyan]: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "/exit"}:
            break
        if user_input.lower() == "/reset":
            runner.reset_history()
            console.print("[yellow]Conversation memory cleared.[/yellow]")
            continue
        if user_input.lower() == "/save":
            runner.save_history()
            console.print(f"[yellow]Conversation saved to {runner.history_path()}[/yellow]")
            continue
        if user_input.lower() == "/where":
            console.print(f"[yellow]{runner.history_path()}[/yellow]")
            continue
        if user_input.lower() == "/fast on":
            console.print(f"[yellow]{runner.set_fast_reply(True)}[/yellow]")
            continue
        if user_input.lower() == "/fast off":
            console.print(f"[yellow]{runner.set_fast_reply(False)}[/yellow]")
            continue
        if user_input.lower() == "/skills":
            for line in runner.list_skills():
                console.print(f"[yellow]{line}[/yellow]")
            continue
        if user_input.lower().startswith("/skill "):
            skill_name = user_input.split(" ", 1)[1].strip()
            if skill_name.lower() in {"off", "none", "disable", "disabled"}:
                runner.set_skill(None)
                console.print("[yellow]Skill disabled.[/yellow]")
                continue
            try:
                runner.set_skill(skill_name)
            except ValueError as exc:
                console.print(f"[red]{exc}[/red]")
                continue
            console.print(f"[yellow]Skill enabled: {runner.active_skill.name}[/yellow]")
            continue
        if user_input.lower() == "/voice on":
            speaker.enabled = True
            console.print("[yellow]Voice enabled.[/yellow]")
            continue
        if user_input.lower() == "/voice off":
            speaker.enabled = False
            console.print("[yellow]Voice disabled.[/yellow]")
            continue
        if user_input.lower() == "/voices":
            voices = list_voices()
            if not voices:
                console.print("[yellow]No Windows speech voices found.[/yellow]")
                continue
            for voice in voices:
                console.print(f"[yellow]{voice}[/yellow]")
            continue
        if user_input.lower() == "/listen":
            user_input = _listen_for_input(listener)
            if not user_input:
                continue
        elif user_input.lower() == "/mic on":
            listener.enabled = True
            console.print("[yellow]Microphone input enabled. Say each prompt after Listening appears.[/yellow]")
            continue
        elif user_input.lower() == "/mic off":
            listener.enabled = False
            console.print("[yellow]Microphone input disabled.[/yellow]")
            continue
        elif user_input.lower() == "/mic-check":
            console.print(f"[yellow]{microphone_diagnostic()}[/yellow]")
            continue
        elif user_input.lower() == "/recognizers":
            recognizers = list_recognizers()
            if not recognizers:
                console.print(f"[yellow]{recognizer_diagnostic()}[/yellow]")
                continue
            for recognizer in recognizers:
                console.print(f"[yellow]{recognizer}[/yellow]")
            continue

        reply = runner.ask(user_input)
        console.print(f"[bold green]{settings.agent_name}[/bold green]: {reply}")
        speaker.speak(reply)


@app.command()
def web(
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    skill: str = typer.Option("", "--skill", help="Start the web app with a specific skill."),
) -> None:
    """Start the local browser conversation app."""
    run_web_app(host=host, port=port, open_browser=open_browser, skill=skill.strip() or None)


@app.callback(invoke_without_command=True)
def default(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        chat()


def _listen_for_input(listener: VoiceListener) -> str:
    if listener.engine == "windows" and not list_recognizers():
        console.print(
            f"[red]{recognizer_diagnostic()} "
            "Install a Windows speech language pack, then try /listen again.[/red]"
        )
        return ""

    console.print("[yellow]Listening...[/yellow]")
    try:
        user_input = listener.listen_once()
    except RuntimeError as exc:
        console.print(f"[red]Could not hear speech: {exc}[/red]")
        return ""

    console.print(f"[bold cyan]You[/bold cyan]: {user_input}")
    return user_input
