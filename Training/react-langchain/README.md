# LangChain Agent Implementations Comparison

This repository demonstrates two different approaches for implementing agents in LangChain:

## 1. ReAct Pattern Implementation (`main_react.py`)
The traditional ReAct (Reasoning + Acting) pattern uses:
- Custom prompt template with specific format instructions
- Manual parsing of LLM output using `ReActSingleInputOutputParser`
- Explicit loop to handle agent actions and observations
- Text-based reasoning format

### Key Features:
- Uses a detailed prompt template describing the thought/action/observation cycle
- Manually manages intermediate steps
- Parses text output to extract actions and inputs
- More control over the reasoning format

### Run:
```bash
python main_react.py
```

## 2. Function Calling Implementation (`main_function_calling.py`)
The modern function calling approach uses:
- Native tool calling via `.bind_tools()`
- Message-based conversation flow
- Direct tool execution through LLM's capabilities
- No manual parsing required

### Key Features:
- Cleaner, more maintainable code
- Uses `HumanMessage`, `AIMessage`, and `ToolMessage`
- LLM directly generates tool calls in structured format
- Less code, fewer imports

### Run:
```bash
python main_function_calling.py
```

## Comparison

| Aspect | ReAct Pattern | Function Calling |
|--------|--------------|------------------|
| Lines of Code | ~109 | ~76 |
| Complexity | Higher - requires prompt engineering | Lower - uses native capabilities |
| Control | More control over reasoning format | Less control, but cleaner |
| Parsing | Manual text parsing required | Automatic structured output |
| Maintenance | More complex to maintain | Easier to maintain |
| LangChain Version | Works with older versions | Requires newer LangChain |

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Set your OpenAI API key in `.env`:
```
OPENAI_API_KEY=your_api_key_here
```

## Output Example

Both implementations will:
1. Accept a question about text length
2. Invoke the `get_text_length` tool
3. Return the final answer

The main difference is in how they process and format the intermediate steps.
