"""
Evaluator-optimizer is useful when you want to generate something and then iteratively improve it based on feedback.

In this example we'll build a python code generator that iteratively improves the code based on feedback.

LLM generation can range from being highly deterministic and repeatable (structured outputs) to being highly stochastic and creative (marketing copy). In the former case, we can use traditional testing and evaluation methods. In the latter case, we need to be able to provide general evaluation guidelines but still account for a wide range of acceptable outputs. This is where an LLM as a judge can be very useful.

Examples of hard-coded evaluations:
1. Does the code compile?
3. Is the post within the character limit?

Examples of more creative evaluations where an LLM-as-a-judge is useful:
1. Does the post maintain a professional tone?
2. Is the post engaging?
3. Does the post effectively answer the user's query?
4. Does the email include personalization?

## KEY TAKEAWAYS
1. The evaluator-optimizer pattern is best suited for when you can clearly define the evaluation criteria and there is measurable improvement in the output with each evaluation cycle.
2. Evaluation criteria can range from being hard-coded (does the code compile?) to being more creative (is the post engaging?). In the latter case, an LLM-as-a-judge can be very useful.
3. You can effectively combined different types of evaluators to create a more robust evaluation process.
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, add_messages, END, START
from langgraph.types import RunnableConfig

load_dotenv()

llm = ChatOpenAI(
    name="Jude",
    model="gpt-4.1-mini-2025-04-14",
    temperature=0.1,
)


#################################
# State
#################################

class CodeReview(BaseModel):
    """The results of reviewing the code."""
    is_valid: bool = Field(..., description="Whether the code meets the requirements")
    feedback: str | None = Field(None, description="Feedback on how to improve the code if it's not valid")

class WorkflowState(BaseModel):
    messages: Annotated[list, add_messages] = []
    code: str | None = None
    code_review: CodeReview | None = None


#################################
# Code Generator
#################################

# The generation step can be any type of output: code, a social media post, an email, etc.

def generate_code(state: WorkflowState):
    system_prompt = SystemMessage(content=f"""
    You are a code generator. Your job is to write clean, functional code based on user requirements. You will be provided with either a user request or feedback on a previous draft of the code.

    <Instructions>
    - Write code that solves the given problem
    - Include basic error handling where appropriate
    - Use clear variable names and add brief comments for complex logic
    - Focus on correctness first, then readability
    - If requirements are unclear, make reasonable assumptions and state them
    </Instructions>

    Output only the code. The code will be reviewed and you may need to revise it based on feedback.
    """)
    if state.code and state.code_review and not state.code_review.is_valid:
        code_and_feedback = f"Previous code and feedback: \n\n{state.code}\n\n{state.code_review.feedback}"
        response = llm.invoke([system_prompt] + state.messages + [code_and_feedback])
    else:
        response = llm.invoke([system_prompt] + state.messages)
    return {"code": response.content}


#################################
# Evaluator
#################################

# Evaluators can be hard-coded or LLM-as-a-judge. An example of a hard-coded evaluator is a linter or a compiler that checks that the code will actually run. An example of an LLM-as-a-judge is below.
# You can effectively combined different types of evaluators to create a more robust evaluation process.

code_review_llm = llm.with_structured_output(schema=CodeReview)

def evaluate_code(state: WorkflowState):
    system_prompt = SystemMessage(content=f"""
    You are a code evaluator. Your job is to review code and provide specific feedback to improve it.

    <Evaluation_Criteria>
    - Correctness: Does the code solve the problem as specified?
    - Security: Are there any security vulnerabilities or input validation issues?
    - Performance: Are there obvious inefficiencies or better algorithmic approaches?
    - Code quality: Is it readable, well-structured, and maintainable?
    - Error handling: Does it handle edge cases and potential failures appropriately?
    </Evaluation_Criteria>
                                  
    <Instructions>
    - If the code passes all criteria, respond with is_valid: true
    - If the code needs improvement, provide specific, actionable feedback
    - Focus on the most critical issues first
    - Be direct and constructive in your feedback
    - Don't approve code with security vulnerabilities or major correctness issues
    </Instructions>

    Format your feedback as a bulleted list of specific changes needed. The goal is to catch critical issues, not to nitpick.
    """)
    response = code_review_llm.invoke(system_prompt.content + f'\n\n<CODE>\n{state.code}\n</CODE>')
    return {"code_review": response}

def evaluator_router(state: WorkflowState) -> str:
    """This is a conditional edge checks if the code is valid and routes to the generate_code node if it is not."""
    if state.code_review and state.code_review.is_valid:
        return END
    return "generate_code"


#################################
# Build the Graph
#################################

builder = StateGraph(WorkflowState)

builder.add_node(generate_code)
builder.add_node(evaluate_code)

builder.set_entry_point("generate_code")

builder.add_edge("generate_code", "evaluate_code")
builder.add_conditional_edges(
    "evaluate_code",
    evaluator_router,
    {
        "generate_code": "generate_code",
        END: END,
    }
)

graph = builder.compile()

# Visualize the graph
from IPython.display import Image
Image(graph.get_graph().draw_mermaid_png())

# Try these example queries
"create a fastapi backend to serve langgraph agents. it should handle managing different conversation threads, streaming, and state management."

response = graph.invoke(
    input=WorkflowState(messages=["write a python script that reads a csv into a pandas dataframe, plots a histogram of the first column, and saves it to a file"]),
    config=RunnableConfig(
        recursion_limit=15
    )
    )

print(response["code"])

# This design pattern can be combined with orchestrator-worker where the orchestrator generates different evaluators and then combines them to give overall, more complete feedback.
