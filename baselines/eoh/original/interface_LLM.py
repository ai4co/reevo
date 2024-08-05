
class InterfaceAPI:
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.client = model_LLM
        self.debug_mode = debug_mode
        self.n_trial = 5

    def get_response(self, prompt_content):

        response = self.client.chat_completion(1, [{"role": "user", "content": prompt_content}], temperature=1.)
        ret = response[0].message.content
        return ret