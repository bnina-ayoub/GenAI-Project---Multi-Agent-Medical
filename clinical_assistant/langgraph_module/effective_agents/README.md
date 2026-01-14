# Building Effective Agents

The world of AI and especially AI Agents is still in its infancy, and constantly changing. But with all the noise around new models, tools, and frameworks, it can be difficult to know where to start and to keep up.

Despite all of the hype, there are just a few key principles that have stood the test of time and are still essential for building effective agents. Those key principles and design patterns are best summarized by Anthropic's [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) blog post. I highly recommend reading it if you haven't already.

Here, we will implement Anthropic's entire guide using Langgraph. First, we will build the key building block of an effective agent: the augmented LLM. We will then implement the core AI workflows. Finally, we will build a powerful LinkedIn content creator agent that will leverage 4+ tools to create highly engaging content for LinkedIn.

**Understand the core design patterns in this guide and you'll be able to solve almost any problem with AI.**

## Video

[![Building Effective Agents](../../static/thumbnails/effective-agents-thumb.jpeg)](https://youtu.be/jsT4YUgz1E4)

## Getting Started

1. If you haven't already, install the project by following the `Getting Started` instructions in the project root [README](../../../README.md).
2. Once the `ai-launchpad` project is installed, navigate to the `langgraph_module/effective_agents` directory and start running the code in any file.

    - If you're new to Langgraph and agents, I recommend starting with `1_llm.py` in the `building_blocks` directory and working your way up through the numbered files. However each file is self-contained so you can skip around as needed.
    - Any python file that begins with a number (e.g. 1_file.py) is meant to be run in interactive mode. Highlight the lines of code you want to run and press `shift+enter` to run them. Running the file as a script will produce all of the outputs at once which may not be helpful.
