from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from callbacks import AgentLoggerCallbackHandler

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text}")
    text = text.strip("'\n").strip()

    return len(text)


if __name__ == "__main__":
    print("Hello React LangChain with bind_tools!")

    # Define tools
    tools = [get_text_length]

    # Initialize LLM with callbacks
    llm = ChatOpenAI(temperature=0, callbacks=[AgentLoggerCallbackHandler()])

    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)

    # Create initial message
    messages = [
        HumanMessage(content="What is the length in characters of the text 'Dog'?")
    ]

    # Invoke the LLM with tools
    response = llm_with_tools.invoke(messages)
    print(f"AI Response: {response}")

    # Check if the AI wants to use a tool
    if response.tool_calls:
        messages.append(response)  # Add AI's response to history

        # Execute each tool call
        for tool_call in response.tool_calls:
            print(f"\nExecuting tool: {tool_call['name']}")
            print(f"With arguments: {tool_call['args']}")

            # Find and execute the tool
            selected_tool = None
            for t in tools:
                if t.name == tool_call["name"]:
                    selected_tool = t
                    break

            if selected_tool:
                # Execute the tool with the provided arguments
                tool_result = selected_tool.func(**tool_call["args"])
                print(f"Tool Output: {tool_result}")

                # Add tool result to messages
                messages.append(
                    ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
                )

        # Get final response from LLM after tool execution
        final_response = llm_with_tools.invoke(messages)
        print("\n" + "=" * 50)
        print(f"FINAL ANSWER: {final_response.content}")
        print("=" * 50)
    else:
        # If no tool calls, just print the response
        print("\n" + "=" * 50)
        print(f"FINAL ANSWER: {response.content}")
        print("=" * 50)
