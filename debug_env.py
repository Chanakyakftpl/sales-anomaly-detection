import os
from dotenv import load_dotenv

load_dotenv()

# Debug: Let's see exactly what's in your environment variable
print("Debugging environment variables:")
print(f"REVENUE_IMPACT_CRITICAL raw value: '{os.getenv('REVENUE_IMPACT_CRITICAL', 'NOT_FOUND')}'")
print(f"Length: {len(os.getenv('REVENUE_IMPACT_CRITICAL', ''))}")

# Let's see all environment variables that might have issues
problematic_vars = [
    'REVENUE_IMPACT_CRITICAL',
    'REVENUE_IMPACT_HIGH', 
    'REVENUE_IMPACT_MEDIUM',
    'VOLUME_IMPACT_CRITICAL',
    'VOLUME_IMPACT_HIGH',
    'VOLUME_IMPACT_MEDIUM',
    'AOV_IMPACT_CRITICAL',
    'AOV_IMPACT_HIGH',
    'AOV_IMPACT_MEDIUM'
]

for var in problematic_vars:
    value = os.getenv(var, 'NOT_FOUND')
    print(f"{var}: '{value}'")
    if '#' in value:
        print(f"  ^ PROBLEM: Contains comment!")
        clean_value = value.split('#')[0].strip()
        print(f"  Clean value would be: '{clean_value}'")
    print()

print("Current working directory:", os.getcwd())
print("Looking for .env file...")

# Check if .env file exists and show its content
env_file_path = os.path.join(os.getcwd(), '.env')
if os.path.exists(env_file_path):
    print(f"Found .env file at: {env_file_path}")
    with open(env_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print("Contents of .env file:")
    for i, line in enumerate(lines, 1):
        if 'REVENUE_IMPACT_CRITICAL' in line:
            print(f"Line {i}: {repr(line)}")  # repr() shows exact characters including spaces
else:
    print("No .env file found!")