from crewai.llms.base_llm import BaseLLM
import requests

class GeminiLLM(BaseLLM):
    def __init__(self, api_key, model="gemini-2.5-pro", temperature=None):
        super().__init__(model=model, temperature=temperature)
        self.api_key = api_key

    def call(self, messages, tools=None, callbacks=None, available_functions=None, **kwargs):
        # Combine messages into prompt text
        prompt_text = ""
        if isinstance(messages, str):
            prompt_text = messages
        else:
            for msg in messages:
                prompt_text += msg.get("content", "") + "\n"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt_text}]
                }
            ]
        }

        response = requests.post(url, json=payload)
        try:
            data = response.json()
        except Exception as e:
            print("Gemini API raw response (non-JSON):", response.text)
            return f"Error from Gemini (non-JSON): {e}"

        try:
            output_text = data["candidates"][0]["content"]["parts"][0]["text"]
            return output_text
        except Exception as e:
            print("Gemini API raw response:", data)
            return f"Error from Gemini: {e}"

    def supports_stop_words(self):
        return False

    def get_context_window_size(self):
        return 32768  # or the correct value for Gemini
