# coding: utf-8
# @author: kaimin
# @time: 2024-04-09 13:29
import json
import requests

from concurrent.futures import ThreadPoolExecutor
from typing import List

from .base_api import BaseApi
from .conversation import Conversation


class MsApi(BaseApi):
    def __init__(self, model: str, token: str, workers: int = 4, retry: int = 2):
        super().__init__(model, retry)
        self.workers = workers
        self.token = token

    def generate_single(self, data: Conversation) -> str:
        return self._generate(data)

    def generate_batch(self, data_list: List[Conversation]) -> List[str]:
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            results = list(executor.map(self._generate, data_list))
        return results

    def _generate(self, data: Conversation, ignore_error: bool = True) -> str:
        headers = {
            'content-type': 'application/json',
            'Authorization': f'Bearer {self.token}'
        }
        req_data = data.to_ms_input()

        max_num_retries = 0
        while max_num_retries < self.retry:
            try:
                response = requests.post(self.model, headers=headers, data=json.dumps(req_data), timeout=300)
            except:
                max_num_retries += 1
                print(json.dumps(req_data, ensure_ascii=False, indent=2))
                print('Got connection error, retrying...')
                continue
            try:
                res = response.json()
                if res is not None and res['choices'][0]['message'].get('content') is not None:
                    return res['choices'][0]['message']['content']
            except:
                print(json.dumps(req_data, ensure_ascii=False, indent=2))
                print('Find error message in response: ', str(response))
            max_num_retries += 1

        if ignore_error:
            return ""
        else:
            raise RuntimeError('Calling MS API failed after retrying for ' f'{max_num_retries} times. Check the logs for details.')
