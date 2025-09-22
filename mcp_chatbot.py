import os
import json
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from openai import AzureOpenAI
from typing import List
import asyncio
import nest_asyncio

nest_asyncio.apply()

load_dotenv()
LLM_MODEL = "gpt-4o"

class MCP_ChatBot:

    def __init__(self):
        # Initialize session and client objects
        load_dotenv() 

        self.session: ClientSession = None
        self.available_tools: List[dict] = []
        self.openai = AzureOpenAI(
            base_url=os.getenv("AZURE_OPENAI_API_URL"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_ad_token=os.getenv("AZURE_OPENAI_API_KEY")
        )

    async def process_query(self, query):
        
        messages = [{'role': 'user', 'content': query}]
        
        response = self.openai.chat.completions.create(max_tokens = 2024,
                                    model = LLM_MODEL, 
                                    tools = self.available_tools,
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
                    print("Calling tools...")
                    assistant_message = {
                        'role': 'assistant',
                        'tool_calls': message.tool_calls
                    }
                    if isinstance(message.content, str) or isinstance(message.content, list):
                        assistant_message['content'] = message.content
                    else:
                        assistant_message['content'] = ""
                    messages.append(assistant_message)

                    for tool_call in message.tool_calls:
                        tool_id = tool_call.id
                        tool_args = json.loads(tool_call.function.arguments)
                        tool_name = tool_call.function.name
                        print(f"Calling tool {tool_name} with args {tool_args}")

                        result = await self.session.call_tool(tool_name, arguments=tool_args)

                        # Normalize tool result to a string for OpenAI API compatibility
                        if isinstance(result, (dict, list)):
                            safe_result = json.dumps(result)
                        else:
                            safe_result = str(result)

                        messages.append({
                            "role": "tool",
                            "content": safe_result,
                            "tool_call_id": tool_id
                        })
                        response = self.openai.chat.completions.create(max_tokens = 2024,
                            model = LLM_MODEL, 
                            tools = self.available_tools,
                            messages = messages
                        )

                        # Check if the next response is a text message and print it
                        if response.choices and response.choices[0].message.content is not None:
                            print(response.choices[0].message.content)
                            process_query = False

    
    
    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
        
                if query.lower() == 'quit':
                    break
                    
                await self.process_query(query)
                print("\n")
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def connect_to_server_and_run(self):
        # Create server parameters for stdio connection
        server_params = StdioServerParameters(
            command="uv",  # Executable
            args=["run", "research_server.py"],  # Optional command line arguments
            env=None,  # Optional environment variables
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                # Initialize the connection
                await session.initialize()
    
                # List available tools
                response = await session.list_tools()
                
                tools = response.tools
                print("\nConnected to server with tools:", [tool.name for tool in tools])
                
                self.available_tools = [{
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                } for tool in response.tools]
    
                await self.chat_loop()

async def main():
    chatbot = MCP_ChatBot()
    await chatbot.connect_to_server_and_run()
  

if __name__ == "__main__":
    asyncio.run(main())
