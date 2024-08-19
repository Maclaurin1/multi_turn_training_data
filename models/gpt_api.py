# coding: utf-8
# @author: kaimin
# @time: 2024-04-08 17:20
import json
import time

import requests
from typing import List

from .base_api import BaseApi
from .conversation import Conversation


class GPTApi(BaseApi):
    def __init__(self, model: str, token: str, retry: int = 2, sleep_seconds: int = 0):
        super().__init__(model, retry)
        # GPT 调用需要的token
        self.token = token
        # gpt 限速可能用到的 sleep
        self.sleep_seconds = sleep_seconds

    def generate_single(self, data: Conversation) -> str:
        return self._generate(data)

    def generate_batch(self, data_list: List[Conversation]) -> List[str]:
        results = list()
        for t_data in data_list:
            t_res = self._generate(t_data)
            results.append(t_res)
        return results

    def _generate(self, data: Conversation, ignore_error: bool = True) -> str:
        headers = {
            'Content-Type': 'application/json',
            "api-key": self.token
        }
        req_data = data.to_gpt_input()

        max_num_retries = 0
        while max_num_retries < self.retry:
            try:
                response = requests.post(self.model, headers=headers, data=json.dumps(req_data), timeout=300)
            except:
                print(json.dumps(req_data, ensure_ascii=False, indent=2))
                print('Got connection error, retrying...')
                if self.sleep_seconds > 0:
                    time.sleep(secs=self.sleep_seconds)
                max_num_retries += 1
                continue

            try:
                res = response.json()
                if res is not None and res['choices'][0]['message'].get('content') is not None:
                    return res['choices'][0]['message']['content']
            except:
                print(json.dumps(req_data, ensure_ascii=False, indent=2))
                print('Find error message in response: ', str(response))
            max_num_retries += 1

            if self.sleep_seconds > 0:
                time.sleep(secs=self.sleep_seconds)

        print(json.dumps(req_data, ensure_ascii=False, indent=2))
        if ignore_error:
            return ""
        else:
            raise RuntimeError('Calling GPT API failed after retrying for ' f'{max_num_retries} times. Check the logs for details.')
