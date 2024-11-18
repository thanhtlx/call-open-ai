
import argparse
import glob 
import os
import time
import datetime
import json
from pathlib import Path
from openai import OpenAI


def run(args):
    client = OpenAI(
          organization='org-T4ew5eiTxSpJrQVcBLLZQhTF',
            project='proj_S4TqSHTu9lYlbVakIbUmlTpU',
    )
    print(f"Start call api: {args.input_dir}")
    input_temp = os.path.join(args.input_dir,'*.jsonl')
    for file in glob.glob(input_temp):
        file_name = str(Path(file).name)
        if file_name.startswith('output'):
            continue
        print(f'process: {file}')
        output_file = os.path.join(args.output_dir,'output_'+str(Path(file).name))
        if os.path.exists(output_file):
            continue
        result = list()
        with open(file) as ff:
            for line in ff.readlines():
                obj = json.loads(line.strip())
                body = obj['body']
                chat_completion = client.chat.completions.create(
                    model=body['model'],
                    messages=body['messages'],
                    max_tokens=body['max_tokens'],
                    temperature=body['temperature']
                )
                msg = chat_completion.choices[0].message.content
                # {"id": "batch_req_USpfpD8tEcno8TvkdurHEBlT", "custom_id": "qemu___qemu_62f94fc94f98095173146e753a1f03d7c2cc7ba3", "response": {"status_code": 200, "request_id": "ebf2b6d413cb47e58d02cfc536572d8c", "body": {"id": "chatcmpl-9Lmz7hcl8QpK4pn1hCYqefBogGiMl", "object": "chat.completion", "created": 1714980213, "model": "gpt-3.5-turbo-0125", "choices": [{"index": 0, "message": {"role": "assistant", "content": "icp: Add unrealize function to icp_kvm and icp classes"}, "logprobs": null, "finish_reason": "stop"}], "usage": {"prompt_tokens": 425, "completion_tokens": 16, "total_tokens": 441}, "system_fingerprint": "fp_3b956da36b"}}, "error": null}

                result.append({'custom_id':obj['custom_id']})
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")
    parser.add_argument('--sleep_time',default=60)
    parser.add_argument("--endpoint",default="/v1/chat/completions")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()