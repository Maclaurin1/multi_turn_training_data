# coding: utf-8
# @author: kaimin
# @time: 2024-04-08 17:20
import json
import requests

from concurrent.futures import ThreadPoolExecutor
from typing import List

from .base_api import BaseApi
from .conversation import Conversation


class MxApi(BaseApi):
    def __init__(self, model: str, workers: int = 4, retry: int = 2):
        super().__init__(model, retry)
        self.workers = workers

    def generate_single(self, data: Conversation) -> str:
        return self._generate(data)

    def generate_batch(self, data_list: List[Conversation]) -> List[str]:
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            results = list(executor.map(self._generate, data_list))
        return results

    def _generate(self, data: Conversation) -> str:
        headers = {'content-type': 'application/json'}
        req_data = data.to_mx_input()

        max_num_retries = 0
        while max_num_retries < self.retry:
            try:
                response = requests.post(self.model, headers=headers, data=json.dumps(req_data))
            except:
                max_num_retries += 1
                print('Got connection error, retrying...')
                print(json.dumps(req_data, ensure_ascii=False, indent=2))
                continue
            try:
                res = json.loads(response.text)
                if res.get('response') is not None:
                    return res['response']
            except:
                print(json.dumps(req_data, ensure_ascii=False, indent=2))
                print('Find error message in response: ', str(response))
            max_num_retries += 1

        raise RuntimeError('Calling EM API failed after retrying for ' f'{max_num_retries} times. Check the logs for details.')
