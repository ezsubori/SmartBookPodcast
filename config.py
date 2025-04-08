import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_env_variable(name, default=None):
    """
    Get an environment variable or return a default value
    
    Args:
        name: Name of the environment variable
        default: Default value if environment variable is not set
        
    Returns:
        Value of the environment variable or default
    """
    value = os.environ.get(name, default)
    if value is None:
        print(f"Warning: Environment variable {name} is not set")
    return value

# API keys and endpoints
BEDROCK_API_BASE = get_env_variable("BEDROCK_API_BASE", "https://hackfest-bedrock-proxy.diligentoneplatform-dev.com/api/v1")
BEDROCK_API_KEY = get_env_variable("BEDROCK_API_KEY")
OPENAI_API_KEY = get_env_variable("OPENAI_API_KEY")
