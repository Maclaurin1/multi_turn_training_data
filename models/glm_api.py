# coding: utf-8
# @author: kaimin
# @time: 2024-04-09 10:18
import json
from concurrent.futures import ThreadPoolExecutor
from zhipuai import ZhipuAI
from typing import List

from .base_api import BaseApi
from .conversation import Conversation


class GLMApi(BaseApi):
    def __init__(self, model: str, token: str, workers: int = 4, retry: int = 2):
        super().__init__(model, retry)
        # GLM 调用需要的token
        self.token = token
        self.workers = workers
        self.client = ZhipuAI(api_key=self.token)

    def generate_single(self, data: Conversation) -> str:
        return self._generate(data)

    def generate_batch(self, data_list: List[Conversation]) -> List[str]:
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            results = list(executor.map(self._generate, data_list))
        return results

    def _generate(self, data: Conversation, ignore_error: bool = True) -> str:
        req_data = data.to_glm_input(self.model)

        max_num_retries = 0
        while max_num_retries < self.retry:
            try:
                response = self.client.chat.completions.create(**req_data)
            except:
                max_num_retries += 1
                print(json.dumps(req_data, ensure_ascii=False, indent=2))
                print('Got connection error, retrying...')
                continue

            try:
                res = response.choices[0].message.content
                return res
            except:
                print(json.dumps(req_data, ensure_ascii=False, indent=2))
                print('Find error message in response: ', str(response.error_message))
            max_num_retries += 1

        if ignore_error:
            return ""
        else:
            raise RuntimeError('Calling GLM API failed after retrying for ' f'{max_num_retries} times. Check the logs for details.')
