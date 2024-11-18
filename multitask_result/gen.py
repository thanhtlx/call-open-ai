import torch
from transformers import AutoModelForCausalLM, CodeLlamaTokenizer
from tqdm import tqdm
from peft import PeftModel
import datasets
import json

def read_contextual_medit_examples(filename):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            examples.append(js['prompt'])
    return examples

def write_string_to_file(absolute_filename, string):
        with open(absolute_filename, 'a') as fout:
            fout.write(string)

def split_batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def gen(model_base, model_peft, output_file,args):
    tokenizer = CodeLlamaTokenizer.from_pretrained(model_base)
    dataset_id = "thanhtlx/test_mcmd_race_sample_no_preprocessing"
    dataset = datasets.load_dataset(dataset_id, split="test")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 


    model = AutoModelForCausalLM.from_pretrained(model_base, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)

    model = PeftModel.from_pretrained(model, model_peft)
    K = 1000
    model.eval()
    sources = [
        data if len(data.split()) < K else ' '.join(data.split()[:K]) + '\nMsg:'
        for data in dataset['prompt']
    ]
    batch_list = split_batch(sources, args.batch_size)
    len_batch = len(sources) // args.batch_size
    
    with tqdm(total=len_batch, desc="gen") as pbar:
        for batch in batch_list:
            #mm = max([len(el) for el in batch])
            model_inputs = tokenizer(batch, return_tensors="pt", padding='longest').to("cuda")
            print(model_inputs['input_ids'].shape)
            if model_inputs['input_ids'].shape[1] > 2500:                
                for b in batch:
                    model_input = tokenizer(b, return_tensors="pt").to("cuda")
                    output = tokenizer.decode(model.generate(**model_input, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id)[0], skip_special_tokens=True)
                    res = output[len(b):].strip()
                    print(res)
                    write_string_to_file(args.output_file, res.strip() + '\n')
                continue
            generated_ids = model.generate(**model_inputs, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id)
            output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for idx, source in enumerate(batch):
                res = output[idx][len(source):].strip()
                if len(res) == 0:
                    print('*'*50,'\n',batch[idx])
                write_string_to_file(args.output_file, res.strip() + '\n')
            pbar.update(1)
   
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--output_file", type=str, default="gen.output")
    args = parser.parse_args()
    gen('codellama/CodeLlama-7b-hf', 'tmp/code-llama-output', 'test.codellama.reload.output',args)
