import arxiv
import json
import os
from typing import List
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv() 

PAPER_DIR = "papers"
LLM_MODEL = "gpt-4o"

#It's tricky to configure our Azure Client due to its url structure
#https://github.com/openai/openai-python/blob/main/src/openai/lib/azure.py
client = AzureOpenAI(
    base_url=os.getenv("AZURE_OPENAI_API_URL"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_ad_token=os.getenv("AZURE_OPENAI_API_KEY")
)
#print(client.auth_headers)
#print(client.base_url)

def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    Search for papers on arXiv based on a topic and store their information.
    
    Args:
        topic: The topic to search for
        max_results: Maximum number of results to retrieve (default: 5)
        
    Returns:
        List of paper IDs found in the search
    """
    
    # Use arxiv to find the papers 
    arx_client = arxiv.Client()

    # Search for the most relevant articles matching the queried topic
    search = arxiv.Search(
        query = topic,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.Relevance
    )

    papers = arx_client.results(search)
    
    # Create directory for this topic
    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)
    
    file_path = os.path.join(path, "papers_info.json")

    # Try to load existing papers info
    try:
        with open(file_path, "r") as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}

    # Process each paper and add to papers_info  
    paper_ids = []
    for paper in papers:
        paper_ids.append(paper.get_short_id())
        paper_info = {
            'title': paper.title,
            'authors': [author.name for author in paper.authors],
            'summary': paper.summary,
            'pdf_url': paper.pdf_url,
            'published': str(paper.published.date())
        }
        papers_info[paper.get_short_id()] = paper_info
    
    # Save updated papers_info to json file
    with open(file_path, "w") as json_file:
        json.dump(papers_info, json_file, indent=2)
    
    print(f"Results are saved in: {file_path}")
    
    return paper_ids

def extract_info(paper_id: str) -> str:
    """
    Search for information about a specific paper across all topic directories.
    
    Args:
        paper_id: The ID of the paper to look for
        
    Returns:
        JSON string with paper information if found, error message if not found
    """
 
    for item in os.listdir(PAPER_DIR):
        item_path = os.path.join(PAPER_DIR, item)
        if os.path.isdir(item_path):
            file_path = os.path.join(item_path, "papers_info.json")
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r") as json_file:
                        papers_info = json.load(json_file)
                        if paper_id in papers_info:
                            return json.dumps(papers_info[paper_id], indent=2)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    continue
    
    return f"There's no saved information related to paper {paper_id}."

tools = [
    {
        "type": "function",
        "function": {
        "name": "search_papers",
        "description": "Search for papers on arXiv based on a topic and store their information.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic to search for"
                }, 
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to retrieve",
                    "default": 5
                }
            },
            "required": ["topic"]
        }
    }},
    {
        "type": "function",
        "function": {
        "name": "extract_info",
        "description": "Search for information about a specific paper across all topic directories.",
        "parameters": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "The ID of the paper to look for"
                }
            },
            "required": ["paper_id"]
        }
    }
    }
]

mapping_tool_function = {
    "search_papers": search_papers,
    "extract_info": extract_info
}

def execute_tool(tool_name, tool_args):
    
    result = mapping_tool_function[tool_name](**tool_args)

    if result is None:
        result = "The operation completed but didn't return any results."
        
    elif isinstance(result, list):
        result = ', '.join(result)
        
    elif isinstance(result, dict):
        # Convert dictionaries to formatted JSON strings
        result = json.dumps(result, indent=2)
    
    else:
        # For any other type, convert using str()
        result = str(result)
    return result


def process_query(query):
    
    messages = [{'role': 'user', 'content': query}]
    
    response = client.chat.completions.create(max_tokens = 2024,
                                  model = LLM_MODEL, 
                                  tools = tools,
                                  messages = messages)
    
    process_query = True
    while process_query:
        assistant_content = []

        # Adapted to use OpenAI chat completions response structure
        for choice in response.choices:
            message = choice.message
            if message.content is not None:
                # Standard text response from the assistant
                print("Standard Text response")
                print(message.content)
                process_query = False
            elif message.tool_calls:
                # The assistant is requesting a tool call
                messages.append({'role': 'assistant', 'content': message.content, 'tool_calls': message.tool_calls})
                
                for tool_call in message.tool_calls:
                    tool_id = tool_call.id
                    tool_args = json.loads(tool_call.function.arguments)
                    tool_name = tool_call.function.name
                    print(f"Calling tool {tool_name} with args {tool_args}")

                    result = execute_tool(tool_name, tool_args)
                    messages.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tool_id
                    })
                    response = client.chat.completions.create(
                        max_tokens=2024,
                        model=LLM_MODEL,
                        tools=tools,
                        messages=messages
                    )

                    # Check if the next response is a text message and print it
                    if response.choices and response.choices[0].message.content is not None:
                        print(response.choices[0].message.content)
                        process_query = False

def chat_loop():
    print("Type your queries or 'quit' to exit.")
    while True:
        try:
            query = input("\nQuery: ").strip()
            if query.lower() == 'quit':
                break
    
            process_query(query)
            print("\n")
        except Exception as e:
            print(f"\nError: {str(e)}")


chat_loop()