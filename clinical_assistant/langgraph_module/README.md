# Langgraph Module

![Langgraph](../static/wordmark_dark.svg)

This module contains tutorials, code templates, and examples to help you master Langgraph.

Langgraph is an open-source, agent orchestration framework. It's great for building complex AI workflows, agents, and applications. It has a ton of features required to build, deploy, evaluate, and monitor production AI applications.

As with any framework, there are pros and cons to using Langgraph. I discuss the tradeoffs in my Intro to Langgraph [video](https://youtu.be/31JoTDm7jkM).

To learn more, see the official Langgraph [documentation](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/).

## Getting Started

1. If you haven't already, install the project by following the `Getting Started` instructions in the project root [README](../../README.md).

## Tutorials

1. [Building Effective Agents (langgraph_module/effective_agents)](langgraph_module/effective_agents/README.md)

    - Watch the [Youtube Video](https://youtu.be/jsT4YUgz1E4)!
    - Anthropic's guide to [building effective agents](https://www.anthropic.com/engineering/building-effective-agents) is a gold standard for building effective agents. I recommend everyone that's building AI agents to read this. Understand these core design patterns and you'll be able to solve almost any problem with AI.
    - Learn how to implement all of the core design patterns of AI workflows from routing to parallelization, to a powerful LinkedIn content creator agent, all in Langgraph.

2. [Frontends for Langgraph Agents (langgraph_module/frontends)](langgraph_module/frontends/README.md)

    - Learn how to build a UI on top of the Langgraph API to interact with your agents.
    - We'll build a simple Streamlit app to interact with any AI agents.

3. [Langgraph Server (langgraph_module/langgraph_server)](langgraph_module/langgraph_server/README.md)

    - Learn how to run a Langgraph server to manage and interact with your agents.
    - We'll use the Langgraph CLI to run a local development server and deploy to production.
