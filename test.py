# import json5
# print(json5.loads('''
# {
#     "a": 1,
#     "a": 2
# }
# '''))

# exit()

from transformers import AutoModelForCausalLM,AutoProcessor,AutoTokenizer
from PIL import Image

import torch

model_path = "/data3/workhome/luyaxi/VCPM-R1/models/MiniCPM-o-2_6-hg"

model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,torch_dtype=torch.bfloat16, init_audio=False, init_tts=False).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path,trust_remote_code=True)


# model.save_pretrained("/data3/workhome/luyaxi/VCPM-R1/models/MiniCPM3-V-1_6B-SFT1000",save_optimizer=False,)


inputs = processor(
    processor.tokenizer.apply_chat_template([
        {"role": "user", "content": "fix the code in the given pic. start with ```python \n(<image>./</image>)"},
    ],tokenize=False,add_generation_prompt=True),
    [Image.open("sample.png")],
    return_tensors="pt"    
).to("cuda")

# inputs["inputs_embeds"],_ = model.get_vllm_embedding(inputs)
# import pdb 
# pdb.set_trace()
# inputs = model.llm.prepare_inputs_for_generation(**inputs)
# output = model.llm(**inputs)
inputs.pop('image_sizes')

res = model.generate(
    **inputs,
    do_sample = True,
    tokenizer=processor.tokenizer,
    top_p = 0.98,
    temperature = 1,
    repetition_penalty = 1.2,
    num_beams = 4,
    num_return_sequences=4,
    max_new_tokens=100
)
for idx, r in enumerate(res[0]):
    print(f"gen {idx}: {r}")

# conv = [{
#     "role":"user",
#     "content": [
#         "fix the code in the given pic.",
#         Image.open("test.png")
#     ]
# }]

# # conv = [
# #     {"role": "user", "content":"how are you"},
# # ]

# res = model.chat(None,conv,tokenizer=tokenizer)

# print(res)