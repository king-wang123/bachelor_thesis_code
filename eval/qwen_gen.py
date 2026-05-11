import openai

class QwenGen:
    def __init__(self, ip="127.0.0.1", port=35000, temperature=0, max_tokens=8192, system_prompt="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."):
        """Initialize the QwenGen class with port and temperature settings."""
        self.temperature = temperature
        self.ip = ip
        self.port = port
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.client = openai.Client(base_url=f"http://{ip}:{port}/v1", api_key="EMPTY")

    def response(self, prompt):
        """Generate a response for the given prompt."""
        tmp_repeat = 0
        while True:
            try:
                if self.system_prompt:
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                else:
                    messages = [
                        {"role": "user", "content": prompt}
                    ]

                completion = self.client.chat.completions.create(
                    model='default',
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )

                return completion.choices[0].message.content
            except Exception as e:
                print(e)
                tmp_repeat += 1
                print(f'repeat {tmp_repeat}')
                if tmp_repeat == 5:
                    break
        return ''
    
    def stream_response(self, prompt):
        """Generate a streaming response for the given prompt."""
        tmp_repeat = 0
        while True:
            try:
                if self.system_prompt:
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                else:
                    messages = [
                        {"role": "user", "content": prompt}
                    ]

                stream = self.client.chat.completions.create(
                    model='default',
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True
                )

                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return
            except Exception as e:
                print(e)
                tmp_repeat += 1
                print(f'repeat {tmp_repeat}')
                if tmp_repeat == 5:
                    break
        return
    
    def response_message_str(self, message_str):
        completion = self.client.completions.create(
            model='default',
            prompt=message_str,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )