from datetime import timedelta, datetime
import time
from openai import OpenAI

from relhelperspy.primitives.rel_result import RelResult
from relhelperspy.text.ColorHelper import ColorHelper as _color
from relhelperspy.io.read_helper import ReadHelper as _read
from relhelperspy.io.project_helper import ProjectHelper as _project

class OpenAITurboChatClient:

    @staticmethod
    def get_api_key():
        env_path = _project.from_root(".env")
        _env = _read.read_key_value(env_path)
        api_key = _env.get("OPENAI_API_KEY")
        
        if api_key is None:
            raise Exception("OPENAI_API_KEY not found in .env")

        return api_key
    
    def __init__(self):
        self.rate_limits = {
            'per_minute': 50,
            'tokens_per_minute': 40000
        }
        self.requests_made = []
        self.tokens_used = 0
        self.client = OpenAI(api_key=OpenAITurboChatClient.get_api_key())

    def make_request(self, prompt) -> RelResult[str, str]:
        self.check_rate_limit(prompt)
        try:
            response = self.client.chat.completions.create(model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ])
            content = response.choices[0].message.content
        except Exception as e:
            return RelResult.error(str(e))

        if "I apologize" in content:
            return RelResult.error("Apology found in response")

        # Update tokens used based on the estimated response size plus the prompt size
        self.tokens_used += len(prompt.split()) + len(content.split())

        return RelResult.success(content)

    def check_rate_limit(self, prompt):
        current_time = datetime.now()
        # Filter out requests that are older than 1 minute
        self.requests_made = [req_time for req_time in self.requests_made if current_time - req_time < timedelta(minutes=1)]
        tokens_estimate = len(prompt.split()) + 150  # Adjust based on expected response length

        # Check if rate limit for requests or tokens per minute is reached
        if len(self.requests_made) >= self.rate_limits['per_minute'] or self.tokens_used + tokens_estimate > self.rate_limits['tokens_per_minute']:
            _color.print_blue_text("Rate limit reached, waiting until reset")
            time.sleep(60)  # Wait for 1 minute before trying again
            self.requests_made.clear()
            self.tokens_used = 0

    def run_single_prompt(self, prompt: str):
        self.requests_made.append(datetime.now())
        result = self.make_request(prompt)

        if result.has_error():
            print(result.get_error())
            return []

        return [result.get_success()]