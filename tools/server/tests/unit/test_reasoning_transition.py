import pytest
from utils import *

# reasoning_transition.reasoning_open: whether the reply ended inside reasoning (the end tag not
# produced), read from the parser's input, with or without a reasoning budget sampler.
# The tiny model writes what a grammar allows, so the thinking ends exactly where the test says.

server: ServerProcess

MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.jinja = True
    # its generation prompt opens the thinking: "<|im_start|>assistant\n<think>\n"
    server.chat_template_file = "../../../models/templates/Qwen3.5-4B.jinja"


def chat(**body):
    res = server.make_request("POST", "/chat/completions", data={"messages": MESSAGES, **body})
    assert res.status_code == 200, res.body
    return res.body["choices"][0], res.body["reasoning_transition"]


def tokens_of(grammar: str) -> int:
    """Completion tokens up to the end of grammar's text (the reply stops there by itself)."""
    choice, tr = chat(max_tokens=64, grammar=grammar)
    assert choice["finish_reason"] == "stop"
    return tr["completion_tokens"] - 1  # the end-of-generation token


def test_natural_end_then_limit_with_no_answer_token():
    global server
    server.start()
    n = tokens_of('root ::= "x</think>"')
    choice, tr = chat(max_tokens=n, grammar='root ::= "x</think>" [a-z]+')
    assert choice["finish_reason"] == "length"
    assert choice["message"]["reasoning_content"] == "x"
    assert not choice["message"].get("content")
    assert tr["started"] == 0 and tr["completed"] == 0
    assert tr["reasoning_open"] is False


def test_natural_end_then_limit_with_line_breaks_only():
    global server
    server.start()
    choice, tr = chat(max_tokens=24, grammar='root ::= "x</think>" "\\n"+')
    assert choice["finish_reason"] == "length"
    assert choice["message"]["reasoning_content"] == "x"
    assert not choice["message"].get("content", "").strip()
    assert tr["reasoning_open"] is False


@pytest.mark.parametrize("stream", [False, True])
def test_natural_end_with_an_answer(stream: bool):
    global server
    server.start()
    body = {"messages": MESSAGES, "max_tokens": 64, "grammar": 'root ::= "x</think>hello"'}
    if not stream:
        choice, tr = chat(**{k: v for k, v in body.items() if k != "messages"})
        assert choice["message"]["content"] == "hello"
    else:
        last = None
        for data in server.make_stream_request("POST", "/chat/completions", data={**body, "stream": True}):
            last = data
        assert last is not None
        tr = last["reasoning_transition"]
    assert tr["finish_reason"] == "stop"
    assert tr["reasoning_open"] is False


@pytest.mark.parametrize("stream", [False, True])
def test_limit_inside_thinking_without_budget_sampler(stream: bool):
    global server
    server.start()
    if not stream:
        choice, tr = chat(max_tokens=8)
        assert choice["finish_reason"] == "length"
        assert choice["message"]["reasoning_content"] and not choice["message"].get("content")
    else:
        last = None
        for data in server.make_stream_request("POST", "/chat/completions",
                                               data={"messages": MESSAGES, "max_tokens": 8, "stream": True}):
            last = data
        tr = last["reasoning_transition"]
    assert tr["finish_reason"] == "length"
    assert tr["started"] == 0 and tr["completed"] == 0
    assert tr["reasoning_open"] is True


def test_limit_inside_thinking_with_idle_budget_sampler():
    global server
    server.start()
    # a budget sampler is created (reasoning_budget_tokens >= 0) but never sees the start tag:
    # this vocab's space prefix tokenizes "<think>" alone unlike in the prompt. Its counts say
    # nothing; reasoning_open does not depend on them.
    choice, tr = chat(max_tokens=8, reasoning_budget_tokens=1000)
    assert choice["finish_reason"] == "length"
    assert tr["completed"] == 0
    assert tr["reasoning_open"] is True


def test_limit_inside_a_partial_end_tag():
    global server
    server.start()
    n = tokens_of('root ::= "x</think>"')
    choice, tr = chat(max_tokens=n - 1, grammar='root ::= "x</think>"')
    assert choice["finish_reason"] == "length"
    assert tr["reasoning_open"] is True


def test_thinking_off():
    global server
    server.start()
    choice, tr = chat(max_tokens=8, chat_template_kwargs={"enable_thinking": False})
    assert choice["finish_reason"] == "length"
    assert not choice["message"].get("reasoning_content")
    assert tr["reasoning_open"] is False


def test_template_without_thinking_tags():
    global server
    server.chat_template_file = None
    server.start()
    choice, tr = chat(max_tokens=8)
    assert choice["finish_reason"] == "length"
    assert tr["reasoning_open"] is None



def gemma():
    # its tokenizer has no space prefix, so the budget sampler matches the "<think>" of the prompt
    global server
    server = ServerPreset.tinygemma3()
    server.jinja = True
    server.chat_template_file = "../../../models/templates/Qwen3.5-4B.jinja"
    return server


@pytest.mark.parametrize("stream", [False, True])
def test_forced_hand_over_completed(stream: bool):
    gemma().start()
    body = {"messages": MESSAGES, "max_tokens": 40, "reasoning_budget_tokens": 2, "reasoning_budget_message": "Stop."}
    if not stream:
        res = server.make_request("POST", "/chat/completions", data=body)
        assert res.status_code == 200
        assert res.body["choices"][0]["message"]["reasoning_content"].endswith("Stop.")
        tr = res.body["reasoning_transition"]
    else:
        last = None
        for data in server.make_stream_request("POST", "/chat/completions", data={**body, "stream": True}):
            last = data
        tr = last["reasoning_transition"]
    assert tr["started"] == 1 and tr["completed"] == 1
    assert tr["reasoning_open"] is False


def test_limit_inside_thinking_with_active_budget_sampler():
    gemma()
    server.n_ctx = 64
    server.start()
    # the sampler is counting (budget not reached) when the context ends the reply
    choice, tr = chat(max_tokens=1000, reasoning_budget_tokens=1000)
    assert choice["finish_reason"] == "length"
    assert choice["message"]["reasoning_content"] and not choice["message"].get("content")
    assert tr["started"] == 0 and tr["completed"] == 0
    assert tr["reasoning_open"] is True
