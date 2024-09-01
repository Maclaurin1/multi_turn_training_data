# coding: utf-8
# @author: kaimin
# @time: 2024-04-08 17:21
from typing import List

from .conversation import Conversation


class BaseApi:
    def __init__(
            self,
            model: str,
            retry: int,
    ):
        # 模型名 或者 url
        self.model = model
        self.retry = retry
        self.workers: int = 1

    def generate_single(self, data: Conversation) -> str:
        pass

    def generate_batch(self, data_list: List[Conversation]) -> List[str]:
        pass

    def _generate(self, data: Conversation) -> str:
        pass
