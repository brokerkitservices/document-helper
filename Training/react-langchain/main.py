from typing import Union, List
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.tools.render import render_text_description
from langchain.tools import tool, Tool
from langchain_openai import ChatOpenAI
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.schema.agent import AgentAction, AgentFinish
from callbacks import AgentLoggerCallbackHandler

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text}")
    text = text.strip("'\n").strip()

    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for t in tools:
        if t.name == tool_name:
            return t
    raise ValueError(f"Tool with name {tool_name} not found")


if __name__ == "__main__":
    print("Hello React LangChain!")
    tools = [get_text_length]

    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    llm = ChatOpenAI(
        temperature=0, stop=["\nObservation"], callbacks=[AgentLoggerCallbackHandler()]
    )
    intermediate_steps = []

    def format_log_to_str(intermediate_steps):
        if not intermediate_steps:
            return ""
        else:
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += f"\nAction: {action.tool}\nAction Input: {action.tool_input}\nObservation: {observation}"
            return thoughts

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    # Loop until we get a final answer
    agent_step = ""
    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the length in characters of the text 'Dog'?",
                "agent_scratchpad": intermediate_steps,
            }
        )
        print(agent_step)

        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input

            observation = tool_to_use.func(str(tool_input))
            print(f"Tool Output: {observation}")
            intermediate_steps.append((agent_step, str(observation)))

        if isinstance(agent_step, AgentFinish):
            # We have the final answer
            print("\n" + "=" * 50)
            print(f"FINAL ANSWER: {agent_step.return_values['output']}")
            print("=" * 50)
