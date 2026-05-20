import os
from dotenv import load_dotenv
try:
    from supabase import create_client
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    print(f"Connecting to: {url}")
    client = create_client(url, key)
    print("Success: Client created")
except Exception as e:
    print(f"Error: {e}")
