import os
import sys
import re

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from utils.llm_pick import pick_llm
from utils.etl_tools import ETLTools
from models.schema import ETLAgentSchema

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool


# ------------------------------------ AGENT TOOLS ------------------------------------ #

@tool
def extract_load_tool(url: str, output_folder: str, format: str) -> str:
    """
    This tool extracts the data from the API (url) and loads it into the
    the desired location (output_folder).

    Args:
        url (str): The API endpoint from which to extract data.
        output_folder (str): The folder where the extracted data will be saved.
        format (str): The format in which to save the extracted data (csv, json, parquet).
    
    Returns:
        str: A message indicating the success or failure of the operation.

    """
    etl_tools = ETLTools()
    return etl_tools.extract_load(url, output_folder, format)


@tool
def transform_load_tool(
    input_file_path: str,
    output_folder: str,
    output_format: str,
    user_question: str) -> str:
    
    """
    This tool transforms the data from the specified file and loads it into the
    desired location (output_folder).

    Args:
        input_file_path (str): The path to the file containing the data to be transformed.
        output_folder (str): The folder where the transformed data will be saved.
        output_format (str): The format in which to save the transformed data (csv, json, parquet).
    
    Returns:
        str: A message indicating the success or failure of the operation.

    """
    etl_tools = ETLTools()

    top_3_rows = etl_tools.transform_load_context(input_file_path)
   
    prompt = f"""
            You are a Python Data Analyst who uses Pandas to analyze data. 
            You need to provide only the Pandas Code that will help to perform the right ETL operations on the data stored in the file : {input_file_path}
            as per the user's question. Do not provide any explanation or comments, only
            the code should be provided. The code should be in a format that can be executed 
            in a Python environment with Pandas installed. 
            Don't write anything else than Pandas Code. \n
            
            Create the Pandas Dataframe from the data stored in the file : {input_file_path} and then 
            write the code to transform and save the data at {output_folder}.
            Here's the user's question: {user_question}\n
            Here's the context of the data you will be analyzing: {top_3_rows}\n

        """

    # Use Groq through the existing LLM factory
    llm = pick_llm("medium")
    response = llm.invoke(prompt)

    pandas_code = (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    ).strip()

    # Remove optional Markdown code fences safely
    pandas_code = re.sub(r"^```(?:python)?\s*", "", pandas_code)
    pandas_code = re.sub(r"\s*```$", "", pandas_code).strip()

    # Execute the generated code using your ETL utility
    results = etl_tools.execute_code(pandas_code)

    return (
        f"Transformation completed.\n"
        f"Output folder: {output_folder}\n"
        f"Output format: {output_format}\n\n"
        f"Pandas code executed:\n{pandas_code}\n\n"
        f"Execution result:\n{results}"
    )


# ------------------------------------ GROQ LLM ------------------------------------ #

tools = [extract_load_tool, transform_load_tool]

# Use your existing Groq model factory instead of ChatAnthropic
llm = pick_llm("medium")
llm_bind = llm.bind_tools(tools)


# ------------------------------------ AGENT GRAPH ------------------------------------ #

def llm_node(state: ETLAgentSchema):
    messages = state.messages

    prompt = f"""
        You are an ETL assistant with access to these tools:
        - extract_load_tool: extracts data from an API and saves it.
        - transform_load_tool: transforms a file using Pandas and saves it.

        Understand the user's request and call the appropriate tool.
        Use the user's requested paths and output format.
        If the user requests multiple ETL steps, execute them in the correct order.
        After the tools finish, summarize the result for the user.
        Do not claim an operation succeeded unless the tool result confirms it.

        Conversation history:
        {messages}
        """

    response = llm_bind.invoke(prompt)

    # Assumes messages uses the LangGraph add reducer in ETLAgentSchema
    return {"messages": [response]}


def tool_node(state: ETLAgentSchema):
    tools_by_name = {t.name: t for t in tools}
    tool_calls = state.messages[-1].tool_calls

    tool_results = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        try:
            observation = tools_by_name[tool_name].invoke(tool_args)
        except Exception as exc:
            observation = f"Tool execution failed: {exc}"

        tool_results.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"]
            )
        )

    return {"messages": tool_results}


def is_tool_call(state: ETLAgentSchema):
    tool_calls = state.messages[-1].tool_calls
    return "tool_node" if tool_calls else "end"


# ------------------------------------ BUILD GRAPH ------------------------------------ #

etl_analyst_graph = StateGraph(ETLAgentSchema)

etl_analyst_graph.add_node("llm_node", llm_node)
etl_analyst_graph.add_node("tool_node", tool_node)

etl_analyst_graph.add_edge(START, "llm_node")

etl_analyst_graph.add_conditional_edges(
    "llm_node",
    is_tool_call,
    {
        "tool_node": "tool_node",
        "end": END
    }
)

etl_analyst_graph.add_edge("tool_node", "llm_node")

etl_analyst = etl_analyst_graph.compile()


# ------------------------------------ RUN AGENT ------------------------------------ #

if __name__ == "__main__":

    response = etl_analyst.invoke({
        "messages": [
            HumanMessage(
                content=(
                    "Extract data from the API endpoint "
                    "'https://pokeapi.co/api/v2/pokemon' and save it "
                    "in the data/extract folder in CSV format."
                )
            )
        ]
    })

    print(response["messages"][-1].content)
