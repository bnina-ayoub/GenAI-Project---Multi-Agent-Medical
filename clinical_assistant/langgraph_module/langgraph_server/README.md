# Langgraph Server

## Getting Started

1. If you haven't already, install the project by following the `Getting Started` instructions in the project root [README](../../../README.md).

## Langgraph Development Server

1. Once the `clinical-assistant` project is installed, navigate to the `langgraph_server` directory.

    ```bash
    cd clinical_assistant/langgraph_module/langgraph_server
    ```

2. Use the Langgraph CLI to run the Langgraph Server for local development.

    ```bash
    langgraph dev
    ```

    This will start the Langgraph server on `http://localhost:2024`. You can now make API calls to the server to create and manage agents, threads, and messages. See your Langgraph API documentation at `http://localhost:2024/docs`.

You can now interact with your agents via the Langgraph API. For a full tutorial on how to build a UI on top of the Langgraph API, check out the `frontends/streamlit_ui` directory.

### Loading Your Own Agents

You can use any agents within the `clinical_assistant/langgraph_module` directory or create your own in `clinical_assistant/langgraph_module/frontends/agents`. To load them into the Langgraph server, update the `langgraph.json` file. For more info on the `langgraph.json` configuration file, see the resources section below.

## Langgraph Production Server

Documentation coming soon... See my [Youtube video](https://youtu.be/xd0oy2FC5g0) on deploying Langgraph to production.

## Resources

- [Langgraph Configuration (langgraph.json)](https://docs.langchain.com/langgraph-platform/application-structure#configuration-file)
- [Langgraph CLI](https://docs.langchain.com/langgraph-platform/cli)
- [Langgraph Server](https://docs.langchain.com/langgraph-platform/langgraph-server)
