from datetime import timedelta, datetime
import time
from groq import Groq
from relhelperspy.primitives.rel_result import RelResult
from relhelperspy.text.ColorHelper import ColorHelper as _color
<<<<<<< HEAD
=======
from relhelperspy.io.read_helper import ReadHelper as _read
from relhelperspy.io.project_helper import ProjectHelper as _project
>>>>>>> cde4e5b21bcc453849b9954c49639a1aa881dd34

# https://github.com/definitive-io/conversational-chatbot-groq/blob/main/app.py
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

class RelGroqChatClient:
    
<<<<<<< HEAD
    def __init__(self, api_key):
        self.rate_limits = {'per_minute': 20, 'per_ten_minutes': 25}
=======
    @staticmethod
    def get_client(api_key:str = None):
        if api_key is None:
            env_path = _project.from_root(".env")
            _env = _read.read_key_value(env_path)
        api_key = _env["GROQ_API_KEY"]
        
        if api_key is None:
            raise Exception("GROQ_API_KEY not found in .env")

        client = RelGroqChatClient(api_key=api_key)
        return client
    
    def __init__(self, api_key):
        self.rate_limits = {'per_minute': 30}
>>>>>>> cde4e5b21bcc453849b9954c49639a1aa881dd34
        self.requests_made = []
        
        self.groq_chat = ChatGroq(
            groq_api_key=api_key, 
            model_name="mixtral-8x7b-32768"
        )

    def make_request(self, conversation, prompt) -> RelResult[str, str]:
        try:
            response = conversation(prompt)
            content = response['response']
        except Exception as e:
            return RelResult.error(str(e))
        
        if "I apologize" in content:
            return RelResult.error("Apology found in response")

        return RelResult.success(content)

<<<<<<< HEAD
    def make_follow_up_request(self, conversation) -> RelResult[str, str]:
        return self.make_request(conversation, f"more")
=======
    def make_follow_up_request(self, conversation, follow_up_text: str = "more") -> RelResult[str, str]:
        return self.make_request(conversation, follow_up_text)
>>>>>>> cde4e5b21bcc453849b9954c49639a1aa881dd34
    
    def check_rate_limit(self):
        current_time = datetime.now()
        self.requests_made = [req_time for req_time in self.requests_made if current_time - req_time < timedelta(minutes=10)]
        
<<<<<<< HEAD
        limit_a = len(self.requests_made) >= self.rate_limits['per_ten_minutes']
        limit_b = len([req_time for req_time in self.requests_made if current_time - req_time < timedelta(minutes=1)]) >= self.rate_limits['per_minute']
        
        while limit_a or limit_b:
=======
        limit_b = len([req_time for req_time in self.requests_made if current_time - req_time < timedelta(minutes=1)]) >= self.rate_limits['per_minute']
        
        while limit_b:
>>>>>>> cde4e5b21bcc453849b9954c49639a1aa881dd34
            _color.print_blue_text("Rate limit reached, waiting 60 seconds")
            time.sleep(60)
            current_time = datetime.now()
            self.requests_made = [req_time for req_time in self.requests_made if current_time - req_time < timedelta(minutes=10)]
<<<<<<< HEAD
            limit_a = len(self.requests_made) >= self.rate_limits['per_ten_minutes']
            limit_b = len([req_time for req_time in self.requests_made if current_time - req_time < timedelta(minutes=1)]) >= self.rate_limits['per_minute']


    def run_prompt(self, prompt: str, follow_ups: int = None):
=======
            limit_b = len([req_time for req_time in self.requests_made if current_time - req_time < timedelta(minutes=1)]) >= self.rate_limits['per_minute']

    def run_single_prompt(self, prompt:str):
        memory=ConversationBufferWindowMemory(k=1)
        conversation = ConversationChain(
            llm=self.groq_chat,
            memory=memory
        )
        
        self.check_rate_limit()
        result = self.make_request(conversation, prompt)
        self.requests_made.append(datetime.now())
        
        if result.has_error():
            print(result.get_error())
            return []
                    
        return result.get_success()

    def run_prompt(self, prompt: str, follow_ups: int = None, follow_up_texts: str = ["more"]):
>>>>>>> cde4e5b21bcc453849b9954c49639a1aa881dd34

        follow_ups = follow_ups if follow_ups is not None else 0
        memory=ConversationBufferWindowMemory(k=follow_ups)

        conversation = ConversationChain(
            llm=self.groq_chat,
            memory=memory
        )
        
        res = []

        self.check_rate_limit()
        result = self.make_request(conversation, prompt)
        self.requests_made.append(datetime.now())
        
        if result.has_error():
            print(result.get_error())
            return []
                    
        res.append(result.get_success())
        
        if follow_ups is not None:
<<<<<<< HEAD
            for _ in range(follow_ups):
                
                self.check_rate_limit()
                result = self.make_follow_up_request(conversation)
=======
            for follow_up_index in range(follow_ups):
                self.check_rate_limit()
                follow_up_text = follow_up_texts[follow_up_index % len(follow_up_texts)]
                result = self.make_follow_up_request(conversation, follow_up_text)
>>>>>>> cde4e5b21bcc453849b9954c49639a1aa881dd34
                
                if result.has_error():
                    print(result.get_error())
                    continue
                
                res.append(result.get_success())
                self.requests_made.append(datetime.now())

        return res
            

      
