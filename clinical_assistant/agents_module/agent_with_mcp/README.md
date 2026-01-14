# Agent With MCP

In this tutorial, we build on the customer service agent from the `agent_from_scratch` module. You're going to learn how to take your AI agents and applications to the next level by connecting them to any number of external tools, resources, and prompts using MCP. You will learn how to build your own MCP server, create a client that can connect to multiple MCP servers, and how to use the client to call tools, retrieve resources, and execute prompts.

If you haven't already, watch the first video on [building agents from scratch](https://youtu.be/gKOwIqBcEZA?si=i6ftvgbFeTX42Wk2)!

## Video

[![Agent with MCP](../../static/thumbnails/mcp_agents2.png)](https://youtu.be/s8loawTCDvc)

## Getting Started

1. If you haven't already, install the project by following the `Getting Started` instructions in the project root [README](../../../README.md).

2. Once the `clinical-assistant` project is installed, navigate to the `clinical_assistant/agents_module/agent_with_mcp` directory and start running the code in any file.

    - Any python file that begins with a number (e.g. 1_file.py) is meant to be run in interactive mode. Highlight the lines of code you want to run and press `shift+enter` to run them. Running the file as a script will produce all of the outputs at once which may not be helpful.

3. The `main.py` file contains the full agent loop. You can run this file as a script to see the agent in action. Follow the video on youtube for a detailed walkthrough.

## Resources

- [Official MCP Docs](https://modelcontextprotocol.io/docs/getting-started/intro)
- [FastMCP Repository](https://github.com/jlowin/fastmcp)
- [Official FastMCP Docs](https://gofastmcp.com/getting-started/welcome)
- [FastMCP Client Setup](https://gofastmcp.com/clients/client)
- [MCP Inspector (requires NodeJS)](https://modelcontextprotocol.io/legacy/tools/inspector#python)
- [OpenAI API - Docs](https://platform.openai.com/docs/api-reference/introduction)
- [OpenAI API - Function Calling](https://platform.openai.com/docs/guides/function-calling/function-calling-behavior#page-top)
- [ChromaDB Docs](https://docs.trychroma.com/docs/overview/getting-started)
- [Tavily Docs](https://docs.tavily.com/sdk/python/quick-start)
