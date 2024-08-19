# coding: utf-8
# @author: kaimin
# @time: 2024-03-28 10:41
from typing import List, Dict


class GenerateArgs(object):
    def __init__(
            self,
            p_temperature: float = 0.8,
            p_top_p: float = 0.9,
            p_top_k: int = 5,
            p_max_new_tokens: int = 800,
            p_repetition_penalty: float = 1.1
    ):
        self.temperature = p_temperature
        self.top_p = p_top_p
        self.top_k = p_top_k
        self.max_new_tokens = p_max_new_tokens
        self.repetition_penalty = p_repetition_penalty


class Conversation(object):
    def __init__(self, p_data: Dict, gpt_internal_api: bool = False):
        if gpt_internal_api:
            self.traceid = p_data['trace_id']
            self.question_text = p_data['questionText']
            self.generate_args = GenerateArgs()
            p_data_config = p_data.get('choices', {})

            if 'temperature' in p_data_config and p_data_config.get('temperature') is not None:
                self.generate_args.temperature = p_data_config['temperature']

            if 'top_p' in p_data_config and p_data_config.get('top_p') is not None:
                self.generate_args.top_p = p_data_config['top_p']

            if 'top_k' in p_data_config and p_data_config.get('top_k') is not None:
                self.generate_args.top_k = p_data_config['top_k']

            if 'max_new_tokens' in p_data_config and p_data_config.get('max_new_tokens') is not None:
                self.generate_args.max_new_tokens = p_data_config['max_new_tokens']

            if 'repetition_penalty' in p_data_config and p_data_config.get('repetition_penalty') is not None:
                self.generate_args.repetition_penalty = p_data_config['repetition_penalty']
        else:
            self.traceid = p_data['trace_id']
            self.system_content = p_data['system_content']
            self.messages = p_data['messages']

            self.generate_args = GenerateArgs()

            if 'temperature' in p_data and p_data.get('temperature') is not None:
                self.generate_args.temperature = p_data['temperature']

            if 'top_p' in p_data and p_data.get('top_p') is not None:
                self.generate_args.top_p = p_data['top_p']

            if 'top_k' in p_data and p_data.get('top_k') is not None:
                self.generate_args.top_k = p_data['top_k']

            if 'max_new_tokens' in p_data and p_data.get('max_new_tokens') is not None:
                self.generate_args.max_new_tokens = p_data['max_new_tokens']

            if 'repetition_penalty' in p_data and p_data.get('repetition_penalty') is not None:
                self.generate_args.repetition_penalty = p_data['repetition_penalty']

    def set_system_content(self, p_system_content):
        self.system_content = p_system_content

    def to_mx_input(self):
        mx_input = {
            "trace_id": self.traceid,
            "system_content": self.system_content,
            "messages": self.messages,
            "top_p": self.generate_args.top_p,
            "top_k": self.generate_args.top_k,
            "temperature": self.generate_args.temperature,
            "max_new_tokens": self.generate_args.max_new_tokens,
        }
        return mx_input

    def to_ms_input(self):
        # 默认用8k
        ms_input = {
            "model": "moonshot-v1-8k",
            "messages": [{"role": "system", "content": self.system_content}] + self.messages,
            "temperature": self.generate_args.temperature,
            "max_tokens": self.generate_args.max_new_tokens,
            "top_p": self.generate_args.top_p,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
        }
        return ms_input

    def to_gpt_input(self):
        gpt_input = {
            "messages": [{"role": "system", "content": self.system_content}] + self.messages,
            "temperature": self.generate_args.temperature,
            "max_tokens": self.generate_args.max_new_tokens,
            "top_p": self.generate_args.top_p,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
        }
        return gpt_input

    def to_internal_gpt_input(self, uid: str, version: str = 'gpt-4o'):
        """
        :param uid: 用户的身份识别码
        :param version: gpt版本，默认为'gpt-4o'
        :return:
        """
        gpt_input = {
            "uid": uid,
            "questionText": self.question_text,
            "type": version,
            "choices": {
                "temperature": self.generate_args.temperature,
                "max_tokens": self.generate_args.max_new_tokens,
                "top_p": self.generate_args.top_p
            },
        }
        return gpt_input

    def to_qwen_input(self):
        qwen_input = {
            "messages": [{"role": "system", "content": self.system_content}] + self.messages,
            "temperature": self.generate_args.temperature,
            "max_tokens": self.generate_args.max_new_tokens,
            "top_p": self.generate_args.top_p,
            "presence_penalty": 0,
            "stop": None,
        }
        return qwen_input
    def add_message(self, message):
        self.messages.append(message)
    def to_glm_input(self, p_model: str):
        if self.generate_args.temperature == 0:
            self.generate_args.temperature = 0.01
        glm_input = {
            "model": p_model,
            "messages": [{"role": "system", "content": self.system_content}] + self.messages,
            "temperature": self.generate_args.temperature,
            "max_tokens": self.generate_args.max_new_tokens,
            "top_p": self.generate_args.top_p
        }
        return glm_input

    def build_mx_data(self, p_response: str = None):
        mx_data = {
            "trace_id": self.traceid,
            "system_content": self.system_content,
            "messages": self.messages,
            "top_p": self.generate_args.top_p,
            "top_k": self.generate_args.top_k,
            "temperature": self.generate_args.temperature,
            "max_new_tokens": self.generate_args.max_new_tokens,
            "response": p_response
        }
        return mx_data
