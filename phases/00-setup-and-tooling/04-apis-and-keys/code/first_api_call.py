import os
import json
import urllib.request


def call_with_sdk_with_stepfun_api():
    try:
        import anthropic
    except ImportError:
        print("Install the SDK: pip install anthropic")
        return

    client = anthropic.Anthropic(api_key= os.environ.get("STEP_API_KEY"), base_url="https://api.stepfun.com/step_plan")
    response = client.messages.create(
        model="step-3.7-flash",
        max_tokens=1024,
        system="你是由阶跃星辰提供的AI聊天助手，你擅长中文，英文，以及多种其他语言的对话。在保证用户数据安全的前提下，你能对用户的问题和请求，作出快速和精准的回答。同时，你的回答和建议应该拒绝黄赌毒，暴力恐怖主义的内容。",
        messages=[{"role": "user", "content": "What is a neural network in one sentence?"}]
    )
    
    print(response.content[1].text)


def call_raw_http():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY environment variable first")
        return

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    body = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "What is a neural network in one sentence?"}],
    }).encode()

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
        print(f"Raw HTTP response: {result['content'][0]['text']}")
        print(f"Tokens used: {result['usage']['input_tokens']} in, {result['usage']['output_tokens']} out")


def call_raw_http_with_stepfun_api():
    api_key = os.environ.get("STEP_API_KEY")
    if not api_key:
        print("Set STEP_API_KEY environment variable first")
        return
    
    url = "https://api.stepfun.com/step_plan/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    body = json.dumps({
        "model": "step-3.7-flash",
        "messages": [
            {
                "role": "system",
                "content": "你是由阶跃星辰提供的AI聊天助手，你擅长中文，英文，以及多种其他语言的对话。在保证用户数据安全的前提下，你能对用户的问题和请求，作出快速和精准的回答。同时，你的回答和建议应该拒绝黄赌毒，暴力恐怖主义的内容。"
            },
            {
                "role": "user",
                "content": "什么是神经网络？"
            }
        ]
    }).encode()

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")


    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
        print(f"Raw HTTP response: {result['choices'][0]['message']['content']}")


if __name__ == "__main__":
    print("=== API Calls ===\n")
    print("1. Using the SDK with StepFun API:")
    call_with_sdk_with_stepfun_api()
    # print("\n2. Using raw HTTP:")
    # call_raw_http()
    # print("\n3. Using raw HTTP with StepFun API:")
    # call_raw_http_with_stepfun_api()
