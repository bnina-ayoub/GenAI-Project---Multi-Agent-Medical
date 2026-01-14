import json

from dotenv import load_dotenv
from langgraph.graph import StateGraph
from langgraph.types import RunnableConfig
from clinical_assistant.langgraph_module.multi_agent.supervisor.supervisor import graph as supervisor_graph, SupervisorState
from langchain_core.messages import HumanMessage, AIMessageChunk
from rich.console import Console
from rich.panel import Panel

load_dotenv()

def get_responsive_width(console: Console) -> int:
    """Get responsive width with margins for panels."""
    return min(120, console.size.width - 4) if console.size.width > 10 else 80

SHOW_TOOL_LOGS = False  # set to True to see tool payloads in the console


async def stream_graph_responses(
        input: SupervisorState,
        graph: StateGraph,
        console: Console,
        **kwargs
        ):
    """Asynchronously stream the result of the graph run with subgraph support.

    Args:
        input: The input to the graph.
        graph: The compiled graph.
        console: Rich console for output.
        **kwargs: Additional keyword arguments.

    Returns:
        str: The final LLM or tool call response
    """
    # Agent styling configuration (clinical pipeline)
    AGENT_STYLES = {
        'symptom': {'color': 'cyan', 'emoji': '🩺', 'name': 'Symptom & Evidence'},
        'hypothesis': {'color': 'magenta', 'emoji': '🧠', 'name': 'Hypotheses'},
        'validator': {'color': 'yellow', 'emoji': '🔍', 'name': 'Validator'},
        'confident': {'color': 'blue', 'emoji': '⚖️', 'name': 'Triage'},
        'xai': {'color': 'green', 'emoji': '📑', 'name': 'XAI'},
        'supervisor': {'color': 'white', 'emoji': '🎯', 'name': 'Supervisor'},
    }

    # Track current AI message source to detect transitions
    current_ai_source = None
    current_content = ""
    current_tool_args = ""
    current_tool_name = ""

    async for chunk in graph.astream(
        input=input,
        stream_mode="messages",
        subgraphs=True,
        **kwargs
        ):
        # When subgraphs=True, the structure is (namespace, (message_chunk, metadata))
        namespace, (message_chunk, _) = chunk

        if isinstance(message_chunk, AIMessageChunk):
            # Determine the source of this AI message directly from namespace
            if namespace:
                # This is from a subgraph - detect agent from namespace
                namespace_str = str(namespace)
                if "call_symptom" in namespace_str:
                    ai_source = "symptom"
                elif "call_hypothesis" in namespace_str:
                    ai_source = "hypothesis"
                elif "call_validator" in namespace_str:
                    ai_source = "validator"
                elif "call_confident" in namespace_str:
                    ai_source = "confident"
                elif "call_xai" in namespace_str:
                    ai_source = "xai"
                else:
                    # Fallback for unknown subgraphs
                    ai_source = "supervisor"
            else:
                # This is from the main graph (supervisor)
                ai_source = "supervisor"

            # Check if we're transitioning between different AI sources
            if current_ai_source != ai_source:
                # Finalize previous agent's content in a panel
                if current_content.strip() and current_ai_source:
                    style = AGENT_STYLES[current_ai_source]
                    panel = Panel(
                        current_content.strip(),
                        title=f"{style['emoji']} {style['name']}",
                        border_style=style['color'],
                        title_align="left",
                        padding=(1, 2),
                        width=get_responsive_width(console)
                    )
                    console.print(panel)
                    console.print()  # Add spacing after completed panel

                # Start new agent
                current_ai_source = ai_source
                current_content = ""
            elif current_ai_source is None:
                # First AI message
                current_ai_source = ai_source
                current_content = ""

            # Handle tool calls
            if SHOW_TOOL_LOGS and message_chunk.response_metadata:
                finish_reason = message_chunk.response_metadata.get("finish_reason", "")
                if finish_reason == "tool_calls":
                    if current_tool_args.strip():
                        formatted_args = current_tool_args.strip()
                        try:
                            parsed = json.loads(formatted_args)
                            formatted_args = json.dumps(parsed, indent=2, ensure_ascii=False)
                        except Exception:
                            pass

                        if current_ai_source:
                            style = AGENT_STYLES[current_ai_source]
                            console.print(f"  [dim {style['color']}]TOOL PAYLOAD:\n{formatted_args}[/dim {style['color']}]")
                        else:
                            console.print(f"  [dim]TOOL PAYLOAD:\n{formatted_args}[/dim]")
                        current_tool_args = ""
                    console.print()

            if SHOW_TOOL_LOGS and message_chunk.tool_call_chunks:
                tool_chunk = message_chunk.tool_call_chunks[0]
                tool_name = tool_chunk.get("name", "")
                args = tool_chunk.get("args", "")

                if tool_name and tool_name != current_tool_name:
                    console.print(f"  🔧 [yellow]{tool_name}[/yellow]")
                    current_tool_name = tool_name
                    current_tool_args = ""

                if args:
                    current_tool_args += args
            else:
                if message_chunk.content:
                    current_content += message_chunk.content
        else:
            # Handle other message types
            pass

    # Print any remaining tool args
    if SHOW_TOOL_LOGS and current_tool_args.strip():
        if current_ai_source:
            style = AGENT_STYLES[current_ai_source]
            console.print(f"  [dim {style['color']}]{current_tool_args.strip()}[/dim {style['color']}]")
        else:
            console.print(f"  [dim]{current_tool_args.strip()}[/dim]")
        console.print()

    # Finalize the last agent's content in a panel
    if current_content.strip() and current_ai_source:
        style = AGENT_STYLES[current_ai_source]
        panel = Panel(
            current_content.strip(),
            title=f"{style['emoji']} {style['name']}",
            border_style=style['color'],
            title_align="left",
            padding=(1, 2),
            width=get_responsive_width(console)
        )
        console.print(panel)
        console.print()  # Add spacing after final panel


async def main():
    """Main function to run the supervisor with subgraphs."""
    # Create console without fixed width - let it be responsive
    console = Console()

    try:
        config = RunnableConfig(configurable={
            "thread_id": "1",
            "recursion_limit": 50,
        })

        # Welcome panel with responsive width
        welcome_panel = Panel(
            "Multi-Agent Supervisor with Subgraphs\nType 'exit' or 'quit' to stop",
            title="🚀 Clinical Assistant",
            border_style="blue",
            title_align="center",
            padding=(1, 2),  # Add padding to welcome panel
            width=get_responsive_width(console)
        )
        console.print(welcome_panel)
        console.print()  # Add spacing after welcome

        while True:
            console.print()
            user_input = console.input("[bold blue]User:[/bold blue] ")
            console.print()  # Add spacing after user input

            if user_input.lower() in ["exit", "quit"]:
                console.print("\n[yellow]Exit command received. Goodbye! 👋[/yellow]\n")
                break

            graph_input = SupervisorState(
                messages=[HumanMessage(content=user_input)]
            )

            await stream_graph_responses(graph_input, supervisor_graph, console, config=config)

    except Exception as e:
        console.print(f"[red]Error: {type(e).__name__}: {str(e)}[/red]")
        raise


if __name__ == "__main__":
    import asyncio
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())