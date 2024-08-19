# coding: utf-8
# @author: kaimin
# @time: 2024-04-08 17:20
import json
import time

import requests
from typing import List

from .base_api import BaseApi
from .conversation import Conversation


class GptIntApi(BaseApi):
    def __init__(self, model: str, token: str, retry: int = 2, sleep_seconds: int = 0, gpt_version: str = 'gpt-4o'):
        super().__init__(model, retry)
        # GPT 调用需要的token
        self.token = token
        # gpt 限速可能用到的 sleep
        self.sleep_seconds = sleep_seconds
        if gpt_version not in {"gpt-4o", "gpt-4-turbo", "gpt-4-32k", "gpt-3.5-turbo", "gpt-3.5-turbo"}:
            print("WARNING: UNKNOWN gpt version. USE gpt-4o instead...")
            self.gpt_version = 'gpt-4o'
        else:
            self.gpt_version = gpt_version

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
            'Content-Type': 'application/json'
        }
        req_data = data.to_internal_gpt_input(version=self.gpt_version, uid=self.token)

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
                if res is not None and res['data']['choices'][0]['message'].get('content') is not None:
                    return res['data']['choices'][0]['message']['content']
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