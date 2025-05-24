import streamlit as st
import json
from tools import tools, get_current_weather, get_time, get_news, calculate_sum, get_joke, get_quote
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Function Calling Demo", page_icon="🤖", layout="centered")

st.title("🤖 Function Calling with LLM Tools")
st.markdown("""
Enter your prompt below. The LLM will decide which tool(s) to call, and you'll see the tool name, parameters, and output for each call.
""")

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)


user_prompt = st.text_input("Enter your prompt:", "What's the weather in Paris and tell me a joke?")

if st.button("Submit"):
    with st.spinner("Thinking..."):
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0,
            max_tokens=300,
            tools=tools,
            tool_choice="auto"
        )
        groq_response = response.choices[0].message
        st.subheader("LLM Response")
        st.code(groq_response, language="json")

        if groq_response.tool_calls:
            for i, tool_call in enumerate(groq_response.tool_calls):
                tool_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                with st.expander(f"Tool Call {i+1}: {tool_name}"):
                    st.write("**Tool Name:**", tool_name)
                    st.write("**Parameters:**", args)
                    st.write("**Output:**")
                    if tool_name == "get_current_weather":
                        result = get_current_weather(**args)
                        st.code(result, language="json")
                    elif tool_name == "get_time":
                        result = get_time(**args)
                        st.code(result, language="json")
                    elif tool_name == "get_news":
                        result = get_news(**args)
                        st.code(result, language="json")
                    elif tool_name == "calculate_sum":
                        result = calculate_sum(**args)
                        st.code(result, language="json")
                    elif tool_name == "get_joke":
                        result = get_joke()
                        st.code(result, language="json")
                    elif tool_name == "get_quote":
                        result = get_quote()
                        st.code(result, language="json")
                    else:
                        st.warning(f"Unknown tool: {tool_name}")
        else:
            st.info("No tool calls in response.") 