import os
import json
from dotenv import load_dotenv

load_dotenv()

from groq import Groq

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

# Get the current weather
def get_current_weather(location):
    """Get the current weather in a given location (LLM synthetic)"""
    prompt = f"""
    You are a weather API. Given the location '{location}', respond with a JSON object with keys 'location' and 'temperature' (in Fahrenheit) for that location. Make the temperature realistic for the location, but you can make it up.
    """
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100,
        )
        content = response.choices[0].message.content
        return content
    except Exception as e:
        return json.dumps({"error": f"Failed to get weather: {str(e)}"})

# Additional tools

def get_time(location):
    """Get the current time in a given location (LLM synthetic)"""
    prompt = f"""
    You are a time API. Given the location '{location}', respond with a JSON object with keys 'location' and 'time' (in 12-hour format, e.g., '3:45 PM') for the current local time in that location. Make up a plausible time.
    """
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=60,
        )
        content = response.choices[0].message.content
        return content
    except Exception as e:
        return json.dumps({"error": f"Failed to get time: {str(e)}"})

def get_news(topic):
    """Get the latest news about a topic (LLM synthetic)"""
    prompt = f"""
    You are a news API. Given the topic '{topic}', respond with a JSON object with keys 'topic' and 'headline' where 'headline' is a plausible, recent-sounding news headline about the topic.
    """
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=80,
        )
        content = response.choices[0].message.content
        return content
    except Exception as e:
        return json.dumps({"error": f"Failed to get news: {str(e)}"})

def calculate_sum(a, b):
    """Calculate the sum of two numbers (LLM synthetic)"""
    prompt = f"""
    You are a math API. Given the numbers a={a} and b={b}, respond with a JSON object with keys 'a', 'b', and 'sum' (where 'sum' is the sum of a and b).
    """
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=60,
        )
        content = response.choices[0].message.content
        return content
    except Exception as e:
        return json.dumps({"error": f"Failed to calculate sum: {str(e)}"})

def get_joke():
    """Get a random joke (LLM synthetic)"""
    prompt = "You are a joke API. Respond with a JSON object with a single key 'joke' and a value that is a short, funny joke."
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=60,
        )
        content = response.choices[0].message.content
        return content
    except Exception as e:
        return json.dumps({"error": f"Failed to get joke: {str(e)}"})

def get_quote():
    """Get a random inspirational quote (LLM synthetic)"""
    prompt = "You are a quote API. Respond with a JSON object with a single key 'quote' and a value that is a short, inspirational quote."
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=60,
        )
        content = response.choices[0].message.content
        return content
    except Exception as e:
        return json.dumps({"error": f"Failed to get quote: {str(e)}"})




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

