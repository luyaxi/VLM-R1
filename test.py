from transformers import AutoModelForCausalLM,AutoProcessor,AutoTokenizer
from PIL import Image

import torch

model_path = "/data3/workhome/luyaxi/VCPM-R1/models/MiniCPM3-V-1_6B-hg"

model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,torch_dtype=torch.bfloat16).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path,trust_remote_code=True)


# model.save_pretrained("/data3/workhome/luyaxi/VCPM-R1/models/MiniCPM3-V-1_6B-hg",save_optimizer=False,)


# inputs = processor(
#     processor.tokenizer.apply_chat_template([
#         {"role": "user", "content": "fix the code in the given pic.\n(<image>./</image>)"},
#     ],tokenize=False,add_generation_prompt=True),
#     [Image.open("test.png")],
#     return_tensors="pt"    
# ).to("cuda")

# inputs["inputs_embeds"],_ = model.get_vllm_embedding(inputs)
# import pdb 
# pdb.set_trace()
# inputs = model.llm.prepare_inputs_for_generation(**inputs)
# output = model.llm(**inputs)



conv = [{
    "role":"user",
    "content": [
        "fix the code in the given pic.",
        Image.open("test.png")
    ]
}]

conv = [
    {"role": "user", "content":"how are you"},
]

res = model.chat(None,conv,tokenizer=tokenizer)

print(res)