import vllm
import argparse
import os 
import json 
import utils.utils as utils
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from pprint import pprint


def load_data(args, filename):


    lines = open(os.path.join(args.data_path, filename), encoding='utf-8').readlines()
    lines = [json.loads(x) for x in lines if x.strip()]
    list_data_dict =  lines 
    
    if 'qwen' in args.base_model.lower() or 'qw2' in args.base_model.lower():
        prompts = []
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        # list_data_dict = list_data_dict[:10]
        for example in list_data_dict:
        
            prompt = '<|im_start|>'+'user\n'+ example["instruction"] +'<|im_end|>\n<|im_start|>assistant\n'

            # prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' + \
            # '<|im_start|>'+'user\n'+ example["instruction"] +'<|im_end|>\n<|im_start|>assistant\n'
        
            prompts.append(prompt)
        print('qwen2:', prompts[0])

    elif 'llama-3' in args.base_model.lower() or 'dpsk' in args.base_model.lower():
        prompts = []
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)

        for example in list_data_dict:

            prompt = example['instruction']
            prompts.append(prompt)
        print('llama3:', prompts[0])

    assert len(prompts) == len(list_data_dict)
    return prompts, list_data_dict

def run(args):

    sampling_params = vllm.SamplingParams(n = args.sample_n, temperature=args.temperature, top_p=0.95, max_tokens=8000)

    print("args:", args)
    model = vllm.LLM(model=args.base_model, tensor_parallel_size=2, trust_remote_code=True)

    fnames = [x for x in os.listdir(args.data_path) if x.endswith('.jsonl')]
    for filename in fnames:
        print(filename)
        prompts, raw_datas = load_data(args, filename)
        print(args.temperature)

        outputs = model.generate(prompts, sampling_params)

        assert len(outputs) == len(raw_datas)
        
        for idx, output in enumerate(outputs):
            prompt = output.prompt
            generated_texts = [item.text for item in output.outputs ]

            raw_datas[idx]["raw_generation"] = generated_texts


        save_path = os.path.join(args.outdir, args.base_model.split('/')[-1]+'_'+filename.split('.')[0]+'.jsonl')

        with open(save_path, 'w') as f:
            for item in raw_datas:
                f.write(json.dumps(item)+'\n')


if __name__ == '__main__': 

    parser = argparse.ArgumentParser(description='Parameters')

    parser.add_argument("--base_model", default="", type=str, help="model path")
    parser.add_argument("--data_path", default="", type=str, help="config path")
    parser.add_argument("--temperature", default=0.0, type=float, help="config path")
    parser.add_argument("--task", default="complete", type=str, help="config path")
    parser.add_argument("--outdir", default="outputs_size", type=str, help="config path")
    parser.add_argument("--do_sample", default=False, type=bool, help="config path")
    parser.add_argument("--model_max_length", type=int, default=8000, help="beam size")
    parser.add_argument("--sample_n", type=int, default=1, help="beam size")

    args = parser.parse_args()

    run(args)