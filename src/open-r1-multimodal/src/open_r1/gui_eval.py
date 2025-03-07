import re
import json
import json5
import random
import jsonschema
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
import os
import difflib

from concurrent.futures import ProcessPoolExecutor

SCHEMA = {
    "type": "object",
    "description": "执行操作并决定当前任务状态",
    "additionalProperties": False,
    "properties": {
    #   "thought": {
    #     "type": "string"
    #   },
      "POINT": {
        "description": "点击屏幕上的指定位置",
        "$ref": "#/$defs/Location"
      },
      "to": {
        "description": "移动，组合手势参数",
        "oneOf": [
          {
            "enum": [
              "up",
              "down",
              "left",
              "right"
            ],
            "description": "结合POINT操作，实现向上下左右滑动"
          },
          {
            "$ref": "#/$defs/Location",
            "description": "移动到某个位置"
          }
        ]
      },
      "duration": {
        "type": "integer",
        "description": "动作执行的时间或等待时间，毫秒",
        "minimum": 0,
        "default": 200
      },
      "PRESS": {
        "type": "string",
        "description": "触发特殊按键，HOME为回到主页按钮，BACK为返回按钮，ENTER为回撤按钮，APPSELECT为查看已打开APP列表按钮",
        "enum": [
          "HOME",
          "BACK",
          "ENTER",
          "APPSELECT"
        ]
      },
      "TYPE": {
        "type": "string",
        "description": "输入文本"
      },
      "DEEP_LINK": {
        "type": "null",
        "description": "跳转到最近打开的APP"
      },
      "CLEAR": {
        "type": "null",
        "description": "清空输入框的内容"
      },
      "STATUS": {
        "type": "string",
        "description": "当前任务的状态。特殊情况：satisfied，无需操作；impossible，任务无法完成；interrupt，任务中断；need_feedback，需要用户反馈；",
        "enum": [
          "continue",
          "finish",
          "satisfied",
          "impossible",
          "interrupt",
          "need_feedback"
        ],
        "default": "continue"
      }
    },
    "$defs": {
      "Location": {
        "type": "array",
        "description": "坐标为相对于屏幕左上角位原点的相对位置，并且按照宽高比例缩放到0～1000，数组第一个元素为横坐标x，第二个元素为纵坐标y",
        "items": {
          "type": "integer",
          "minimum": 0,
          "maximum": 1000
        },
        "minItems": 2,
        "maxItems": 2
      }
    }
}

def load_and_validate_action(res:str,):
    # action_str = re.search(r'```json(.*?)```', res, re.DOTALL)
    # if action_str:
    #     action_str = action_str.group(1).strip()
    # else:
    #     action_str = res
    action_str = res
    
    action = json5.loads(action_str,allow_duplicate_keys=False)
    jsonschema.validate(action, SCHEMA)
    return action

global_executor = ProcessPoolExecutor(max_workers=8)

def _action_schema_check(res:str):
    try:
        action:dict = load_and_validate_action(res)
        return 1.0
    except jsonschema.ValidationError as e:
        return 0.5
    except Exception as e:
        return 0.0

def action_schema_check(completions, **kwargs):
    global global_executor
    futures = [global_executor.submit(_action_schema_check,completion[0]["content"],) for completion in completions]
    scores = []
    for future in futures:
        try:
            scores.append(future.result(timeout=5))
        except TimeoutError as e:
            print("Timeout while checking schema.")
            scores.append(0.0)

    return scores

def _action_type_check(res:str, solution: list[dict]):
    try:
        action = load_and_validate_action(res)
        jaccard_index = len(set(action.keys()) & set(solution.keys())) / len(set(solution.keys()).union(set(action.keys())))
        if jaccard_index < 1:
            print("Mismatched keys in action, Expected: ", solution.keys(), " Got: ", action.keys())
        return jaccard_index
    except Exception as e:
        return 0.0
    

def action_type_check(completions, solution: list[dict], **kwargs):
    global global_executor
    futures = [global_executor.submit(_action_type_check,completion[0]["content"],sol) for completion,sol in zip(completions,solution)]
    scores = []
    for future in futures:
        try:
            scores.append(future.result(timeout=5))
        except TimeoutError as e:
            print("Timeout while checking type.")
            scores.append(0.0)

    return scores

def _action_args_check(res:str, solution: list[dict]):
    try:
        action = load_and_validate_action(res)
    except Exception as e:
        return 0.0
    
    sub_scores = []
    for k in solution.keys():
        if k not in action:
            sub_scores.append(0.0)
            continue
        sub_score = 0.0
        match k:
            case "POINT":
                continue
            
            case "duration":
                if action[k] > 150 or action[k] < 5000:
                    sub_score = 1.0
                else:
                    print("Invalid duration: ", action[k])
                    sub_score = 0.0
            
            case "TYPE":
                similarity = difflib.SequenceMatcher(None, action[k], solution[k]).ratio()
                sub_score = similarity
                print("Text: ",solution[k],", Got: ", action[k],". Similarity: ", similarity)
                
            case "to":
                if isinstance(solution[k], list):
                    continue
                else:
                    if isinstance(action[k],list):
                        sub_score = 0.0
                        print(f"Invalid to for direction {solution[k]}: ", action[k])
                    else:
                        if action[k] == solution[k]:
                            sub_score = 1.0
                        else:
                            sub_score = 0.0
                            print("Invalid to: ", action[k])
            
            case _:
                if solution[k] is None:
                    if action[k] is None:
                        sub_score = 1.0
                    else:
                        print("Required ", k, ", got: ", action[k])
                        sub_score = 0.0
                else:
                    if action[k] == solution[k]:
                        sub_score = 1.0
                    else:
                        print("Required ", k, ", got: ", action[k])
                        sub_score = 0.0
                        
        sub_scores.append(sub_score)
    if not sub_scores:
        print("No args to check.")
        return 0.0
    else:
        return sum(sub_scores) / len(sub_scores)
    

def action_args_check(completions, solution: list[dict], **kwargs):
    global global_executor
    futures = [global_executor.submit(_action_args_check,completion[0]["content"],sol) for completion,sol in zip(completions,solution)]

    scores = []
    for future in futures:
        try:
            scores.append(future.result(timeout=5))
        except TimeoutError as e:
            print("Timeout while checking type.")
            scores.append(0.0)

    return scores

def calculate_dist_score(x:int,y:int,gt_x:int,gt_y:int):
    dist_score = 0.0
    delta_x = abs(gt_x - x)
    delta_y = abs(gt_y - y)
    max_delta = max(delta_x,delta_y)
    if max_delta > 500:
        dist_score = 0.0
        print("Pixel Distance too large: ", max_delta)
    else:
        dist_score = 1 - max_delta / 1000
    
    return dist_score


def _point_distance_check(res:str, solution: list[dict]):
    try:
        action = load_and_validate_action(res)
    except Exception as e:
        return 0.0
    
    point_score = None
    if "POINT" in solution:
        if "POINT" not in action:
            point_score = 0.0
        else:
            point_score = calculate_dist_score(*action["POINT"], *solution["POINT"])
    
    to_score = None
    if "to" in solution:
        if "to" not in action:
            to_score = 0.0
        else:
            if isinstance(solution["to"], list):
                if not isinstance(action["to"],list):
                    print("Must be coordinate to format: ", action)
                    to_score = 0.0
                else:
                    to_score = calculate_dist_score(*action["to"], *solution["to"])
            else:
                to_score = None
    
    if point_score is not None and to_score is not None:
        return (point_score + to_score) / 2
    elif point_score is not None and to_score is None:
        return point_score
    elif point_score is None and to_score is not None:
        return to_score
    else:
        return 0.0


def point_distance_check(completions, solution: list[dict], **kwargs):
    global global_executor
    futures = [global_executor.submit(_point_distance_check,completion[0]["content"],sol) for completion,sol in zip(completions,solution)]
    scores = []
    for future in futures:
        try:
            scores.append(future.result(timeout=5))
        except TimeoutError as e:
            print("Timeout while checking type.")
            scores.append(0.0)
    
    return scores


def compact_json_dumps(obj):
    return json.dumps(obj, indent=None, separators=(",", ":"), ensure_ascii=False)


SYSTEM_PROMPT = f"""# Role
你是一名熟悉安卓系统触屏GUI操作的智能体，将根据用户的意图，分析当前界面的GUI元素和布局，生成相应的操作。

# Task
针对用户意图，根据输入的当前屏幕截图、屏幕元素描述，输出下一步的操作。

# Rule
- 以紧凑JSON格式输出**一个**操作
- 输出操作必须遵循Schema约束
- 通过注释描述你的思考过程

# Schema
""" + compact_json_dumps(SCHEMA) + \
"""
# Example Output
/* 当前界面... */
{"POINT":[100,200]}
```"""


class GUIRFTDataset(Dataset):
    def __init__(self, jsonl_file_path: str, *args, **kwargs):
        super().__init__()
        self.data = []
        self.jsonl_file_path = jsonl_file_path
        with open(jsonl_file_path, "r") as f:
            for line in tqdm(f.readlines(), desc="Loading dataset",dynamic_ncols=True):
                try:
                    self.data.append(json.loads(line))
                except:
                    print("Error while loading line.")
                    continue
        self.image_root = os.path.dirname(os.path.dirname(jsonl_file_path))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        
        for img_id,img_file in item["image"].items():
            try:
                img = Image.open(os.path.join(self.image_root,img_file.replace("/home/test/test03/lyx/check_gui_data-filter_similar/",""))).convert("RGB")
            except:
                print("Error while loading image: ", img_file)
                return self[random.randint(0,len(self.data)-1)]
            h,w = img.size
            
            # resize the max height and width to 1000
            max_line = 1024
            if h > max_line:
                w = int(w * max_line / h)
                h = max_line
            if w > max_line:
                h = int(h * max_line / w)
                w = max_line
            img = img.resize((h,w),resample=Image.Resampling.BILINEAR)
            # img = img.resize((h//2,w//2),resample=Image.Resampling.BILINEAR)
            
            break
        
        try:
            # process the conversation
            user_query = item["conversations"][-2]["content"]
            user_query = re.match(r"<Question>(.*?)</Question>", user_query).group(1)
            action = json.loads(item["conversations"][-1]['content'])
        except:
            print("Error while processing conversation.")
            return self[random.randint(0,len(self.data)-1)]
        conv = []
        conv.append({"role":"system","content":SYSTEM_PROMPT})
        conv.append({"role":"user","content":[
            img,
            user_query
        ]})
        
        return {
            "image":img,
            "resolution": img.size,
            "solution": action,
            "prompt": conv
        }



if __name__=="__main__":
    dataset = GUIRFTDataset("/data3/workhome/luyaxi/VCPM-R1/GUIData/mb_data/tasks.jsonl")
    dataset[10]["image"].save("sample.png")