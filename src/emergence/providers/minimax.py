"""MiniMax plan chat protocol; required reasoning needs a verified shared bound."""

from emergence.providers.base import ChatAdapter


class MinimaxAdapter(ChatAdapter):
    output_parameter = "max_completion_tokens"
    mandatory_reasoning = True
