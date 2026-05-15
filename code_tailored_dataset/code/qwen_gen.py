import openai
import httpx


class QwenGen:
    def __init__(self, ip="127.0.0.1", port=35000, temperature=0, max_tokens=8192, system_prompt="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."):
        """Initialize the QwenGen class with port and temperature settings."""
        self.temperature = temperature
        self.ip = ip
        self.port = port
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.base_url = f"http://{ip}:{port}/v1"
        # Bypass proxy for local model servers
        self.client = openai.Client(
            base_url=self.base_url,
            api_key="EMPTY",
            http_client=httpx.Client(trust_env=False, timeout=600),
        )
        # Auto-detect model name from the server
        self.model_name = self._detect_model_name()

    def _detect_model_name(self):
        """Query /v1/models to get the actual model name."""
        try:
            models = self.client.models.list()
            if models.data:
                return models.data[0].id
        except Exception:
            pass
        return 'default'

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
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )

                return completion.choices[0].message.content
            except Exception as e:
                print(e)
                tmp_repeat += 1
                print(f'repeat {tmp_repeat}')
                if tmp_repeat >= 5:
                    return ''

    def response_messages(self, messages):
        """Generate a response from a full message list (multi-turn)."""
        tmp_repeat = 0
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(e)
                tmp_repeat += 1
                print(f'repeat {tmp_repeat}')
                if tmp_repeat >= 5:
                    return ''

    def response_message_str(self, message_str):
        completion = self.client.completions.create(
            model=self.model_name,
            prompt=message_str,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return completion.choices[0].text

# CUDA_VISIBLE_DEVICES="0,1,2,3" python -m vllm.entrypoints.openai.api_server --served-model-name default --model="/data/share/Qwen3-Coder-Next" --trust-remote-code --tensor-parallel-size=4 --port="35000"
# nohup env CUDA_VISIBLE_DEVICES="0,1,2,3" python -m vllm.entrypoints.openai.api_server --served-model-name default --model="/data/share/Qwen3-Coder-Next" --trust-remote-code --tensor-parallel-size=4 --port="35000" > /data/250010072/zlh/king/code_tailored_dataset/log/qwen0.log 2>&1 &
# 10.120.6.217
# 10.120.7.123