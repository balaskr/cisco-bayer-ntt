

def read_prompt(agent):
    try:
        with open(f"{agent}.prompt", "r", encoding='utf-8') as prompt_file:
            content = prompt_file.read()
        return content
    except FileNotFoundError:
        print(f"Warning: Prompt file '{agent}.prompt' not found. Returning empty string.")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred while reading '{agent}.prompt': {e}")
        return ""

def mock_api()->str:
    """Returns sites information of the given user"""
    with open("data.json", 'r', encoding='utf-8') as file:
        content = file.read()
    return content


class Prompts:
    def __init__(self):
        # The prompts are loaded when an instance of Prompts is created
        self.delegator = read_prompt("delegator")
        self.sites = read_prompt("sites")
        self.tasks = read_prompt("tasks")
        self.overall = read_prompt("overall")

prompts = Prompts()