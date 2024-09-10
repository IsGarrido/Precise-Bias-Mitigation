<<<<<<< HEAD
from datetime import timedelta, datetime
import time
from groq import Groq
from relhelperspy.primitives.rel_result import RelResult
from relhelperspy.text.ColorHelper import ColorHelper as _color

class RelGroqClient:
    
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.rate_limits = {'per_minute': 20, 'per_ten_minutes': 25}
        self.requests_made = []

    def make_request(self, prompt) -> RelResult[str, str]:
        
        try:
            response = self.client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": prompt,
                    
                    }],
                model="mixtral-8x7b-32768",
            )
            content = response.choices[0].message.content
        except Exception as e:
            return RelResult.error(str(e))
        
        if "I apologize" in content:
            return RelResult.error("Apology found in response")

        return RelResult.success(content)

    def make_follow_up_request(self) -> RelResult[str, str]:
        return self.make_request(f"more")
    
    def check_rate_limit(self):
        current_time = datetime.now()
        self.requests_made = [req_time for req_time in self.requests_made if current_time - req_time < timedelta(minutes=10)]
        
        limit_a = len(self.requests_made) >= self.rate_limits['per_ten_minutes']
        limit_b = len([req_time for req_time in self.requests_made if current_time - req_time < timedelta(minutes=1)]) >= self.rate_limits['per_minute']
        
        while limit_a or limit_b:
            _color.print_blue_text("Rate limit reached, waiting 60 seconds")
            time.sleep(60)
            current_time = datetime.now()
            self.requests_made = [req_time for req_time in self.requests_made if current_time - req_time < timedelta(minutes=10)]

    def run_prompt(self, prompt: str, follow_ups: int = None):

        res = []

        self.check_rate_limit()
        result = self.make_request(prompt)
        self.requests_made.append(datetime.now())
        
        if result.has_error():
            print(result.get_error())
            return []
                    
        res.append(result.get_success())
        
        if follow_ups is not None:
            for _ in range(follow_ups):
                
                self.check_rate_limit()
                result = self.make_follow_up_request()
                
                if result.error():
                    print(result.get_error())
                    continue
                
                res.append(result.get_success())
                self.requests_made.append(datetime.now())

        return res
            

      
=======
from datetime import timedelta, datetime
import time
from groq import Groq
from relhelperspy.primitives.rel_result import RelResult
from relhelperspy.text.ColorHelper import ColorHelper as _color

class RelGroqClient:
    
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.rate_limits = {'per_minute': 30}
        self.requests_made = []

    def make_request(self, prompt) -> RelResult[str, str]:
        
        try:
            response = self.client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": prompt,
                    
                    }],
                model="mixtral-8x7b-32768",
            )
            content = response.choices[0].message.content
        except Exception as e:
            return RelResult.error(str(e))
        
        if "I apologize" in content:
            return RelResult.error("Apology found in response")

        return RelResult.success(content)

    def make_follow_up_request(self) -> RelResult[str, str]:
        return self.make_request(f"more")
    
    def check_rate_limit(self):
        current_time = datetime.now()
        self.requests_made = [req_time for req_time in self.requests_made if current_time - req_time < timedelta(minutes=10)]
        
        limit_b = len([req_time for req_time in self.requests_made if current_time - req_time < timedelta(minutes=1)]) >= self.rate_limits['per_minute']
        
        while limit_b:
            _color.print_blue_text("Rate limit reached, waiting 60 seconds")
            time.sleep(60)
            current_time = datetime.now()
            self.requests_made = [req_time for req_time in self.requests_made if current_time - req_time < timedelta(minutes=10)]

    def run_prompt(self, prompt: str, follow_ups: int = None):

        res = []

        self.check_rate_limit()
        result = self.make_request(prompt)
        self.requests_made.append(datetime.now())
        
        if result.has_error():
            print(result.get_error())
            return []
                    
        res.append(result.get_success())
        
        if follow_ups is not None:
            for _ in range(follow_ups):
                
                self.check_rate_limit()
                result = self.make_follow_up_request()
                
                if result.error():
                    print(result.get_error())
                    continue
                
                res.append(result.get_success())
                self.requests_made.append(datetime.now())

        return res
            

      
>>>>>>> cde4e5b21bcc453849b9954c49639a1aa881dd34
