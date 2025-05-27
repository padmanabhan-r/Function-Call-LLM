from groq import Groq
import json
import os
import streamlit as st
from tools import tools, get_current_weather, get_time, get_news, calculate_sum, get_joke, get_quote
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Agentic Function Calling", page_icon="🤖")

st.title("🤖 Function Calling with LLM Tools")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

tool_fn_map = {
    "get_current_weather": get_current_weather,
    "get_time": get_time,
    "get_news": get_news,
    "calculate_sum": calculate_sum,
    "get_joke": get_joke,
    "get_quote": get_quote
}

user_prompt = st.text_input("Enter your prompt:", "What's the weather in Chennai and tell me a joke?")
if st.button("Submit"):
    with st.spinner("Thinking..."):
        messages = [{"role": "user", "content": user_prompt}]
        final_response = None

        for step in range(3):  # prevent infinite loops
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0,
                max_tokens=300,
            )
            msg = response.choices[0].message
            tool_calls = msg.tool_calls
            st.write(msg)

            if tool_calls:
                messages.append(msg)  # Add the assistant's message containing tool_calls
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    func = tool_fn_map.get(tool_name)

                    if func:
                        tool_output = func(**args) if args else func()
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_output
                        })
            else:
                final_response = msg.content
                break

        if final_response:
            st.subheader("LLM Final Response")
            st.write(final_response)
        else:
            st.info("No final response received.")
