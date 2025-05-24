# # Function Calling with OpenAI APIs

import os
import json
from dotenv import load_dotenv

load_dotenv()

from groq import Groq

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)


# 1. Define Function to fetch context

# Get the current weather
def get_current_weather(location):
    """Get the current weather in a given location (LLM synthetic)"""
    prompt = f"""
    You are a weather API. Given the location '{location}', respond with a JSON object with keys 'location' and 'temperature' (in Fahrenheit) for that location. Make the temperature realistic for the location, but you can make it up.
    """
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=100,
    )
    return response.choices[0].message.content

# Additional tools

def get_time(location):
    """Get the current time in a given location (LLM synthetic)"""
    prompt = f"""
    You are a time API. Given the location '{location}', respond with a JSON object with keys 'location' and 'time' (in 12-hour format, e.g., '3:45 PM') for the current local time in that location. Make up a plausible time.
    """
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=60,
    )
    return response.choices[0].message.content

def get_news(topic):
    """Get the latest news about a topic (LLM synthetic)"""
    prompt = f"""
    You are a news API. Given the topic '{topic}', respond with a JSON object with keys 'topic' and 'headline' where 'headline' is a plausible, recent-sounding news headline about the topic.
    """
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=80,
    )
    return response.choices[0].message.content

def calculate_sum(a, b):
    """Calculate the sum of two numbers (LLM synthetic)"""
    prompt = f"""
    You are a math API. Given the numbers a={a} and b={b}, respond with a JSON object with keys 'a', 'b', and 'sum' (where 'sum' is the sum of a and b).
    """
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=60,
    )
    return response.choices[0].message.content

def get_joke():
    """Get a random joke (LLM synthetic)"""
    prompt = "You are a joke API. Respond with a JSON object with a single key 'joke' and a value that is a short, funny joke."
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=60,
    )
    return response.choices[0].message.content

def get_quote():
    """Get a random inspirational quote (LLM synthetic)"""
    prompt = "You are a quote API. Respond with a JSON object with a single key 'quote' and a value that is a short, inspirational quote."
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=60,
    )
    return response.choices[0].message.content

# ### Define Functions
# 
# As demonstrated in the OpenAI documentation, here is a simple example of how to define the functions that are going to be part of the request. 
# 
# The descriptions are important because these are passed directly to the LLM and the LLM will use the description to determine whether to use the functions or how to use/call.



# Define a function for LLM to use as a tool
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        },   
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Get the latest news about a topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to get news about",
                    }
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_sum",
            "description": "Calculate the sum of two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "First number",
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number",
                    }
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_joke",
            "description": "Get a random joke",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_quote",
            "description": "Get a random inspirational quote",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

# Ask user for input
user_prompt = input("Enter your prompt: ")

response = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=[
        {
            "role": "user",
            "content": user_prompt,
        }
    ],
    temperature=0,
    max_tokens=300,
    tools=tools,
    tool_choice="auto"
)

# print(response.choices[0].message.content)


groq_response = response.choices[0].message
print(groq_response)

# response.tool_calls[0].function.arguments

# We can now capture the arguments:

if groq_response.tool_calls:
    for tool_call in groq_response.tool_calls:
        tool_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        print(f"Tool: {tool_name}, Args: {args}")
        if tool_name == "get_current_weather":
            print("output:")
            print(get_current_weather(**args))
        elif tool_name == "get_time":
            print("output:")
            print(get_time(**args))
        elif tool_name == "get_news":
            print("output:")
            print(get_news(**args))
        elif tool_name == "calculate_sum":
            print("output:")
            print(calculate_sum(**args))
        elif tool_name == "get_joke":
            print("output:")
            print(get_joke())
        elif tool_name == "get_quote":
            print("output:")
            print(get_quote())
        else:
            print(f"Unknown tool: {tool_name}")
else:
    print("No tool calls in response.")
#  Put this into another LLM call and return the response in text formatLin