# import json5
# print(json5.loads('''
# {
#     a: 1,
#     a: 2
# }
# '''))

# exit()

from transformers import AutoModelForCausalLM,AutoProcessor,AutoTokenizer
from PIL import Image

import torch

model_path = "/data3/workhome/luyaxi/VCPM-R1/models/MiniCPM-V-HW-7B-hg"

model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,torch_dtype=torch.bfloat16).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path,trust_remote_code=True)


# model.save_pretrained("/data3/workhome/luyaxi/VCPM-R1/models/MiniCPM-V-HW-7B-hg",save_optimizer=False,)

# with torch.no_grad():

    # inputs = processor(
    #     processor.tokenizer.apply_chat_template([
    #         {"role": "user", "content": "描述图像(<image>./</image>)"},
    #     ],tokenize=False,add_generation_prompt=True),
    #     [Image.open("test.png")],
    #     return_tensors="pt"    
    # ).to("cuda")
    # inputs["inputs_embeds"],_ = model.get_vllm_embedding(inputs)
    # print(inputs["inputs_embeds"].shape)
# import pdb 
# pdb.set_trace()
# inputs = model.llm.prepare_inputs_for_generation(**inputs)
# output = model.llm(**inputs)
# inputs.pop('image_sizes',None)

# res = model.generate(
#     **inputs,
#     do_sample = True,
#     tokenizer=processor.tokenizer,
#     top_p = 0.98,
#     temperature = 1,
#     repetition_penalty = 1.2,
#     num_beams = 4,
#     num_return_sequences=4,
#     max_new_tokens=100
# )

# for idx, r in enumerate(res):
#     print(f"gen {idx}: {processor.tokenizer.decode(r,skip_special_tokens=True)}")

conv = [{
    "role":"user",
    "content": [
        "描述图像",
        Image.open("test.png")
    ]
}]

# conv = [
#     {"role": "user", "content":"how are you"},
# ]

res = model.chat(None,conv,tokenizer=tokenizer)

print(res)