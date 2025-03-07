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
    action_str = re.search(r'```json(.*?)```', res, re.DOTALL)
    if action_str:
        action_str = action_str.group(1).strip()
    else:
        action_str = res
    
    action = json5.loads(action_str,allow_duplicate_keys=False)
    jsonschema.validate(action, SCHEMA)
    return action

def action_schema_check(completions, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    scores = []
    for res in contents:
        res:str = res.strip()
        # then load and validate the action
        try:
            action:dict = load_and_validate_action(res)
            scores.append(1.0)
        except jsonschema.ValidationError as e:
            # print("Invalid Action Format: ", res)
            scores.append(0.5)
        except Exception as e:
            # print("Error while loading action: ", res)
            scores.append(0.0)

    return scores

def action_type_check(completions, solution: list[dict], **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    scores = []
    for res, sol in zip(contents, solution):
        res:str = res.strip()
        # then load and validate the action
        try:
            action: dict = load_and_validate_action(res)
            if set(action.keys()) != set(sol.keys()):
                print("Mismatched keys in action, Expected: ", sol.keys(), " Got: ", action.keys())
                scores.append(0.0)
            else:
                scores.append(1.0)
        except Exception as e:
            scores.append(0.0)

    return scores

def action_args_check(completions, solution: list[dict], **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    scores = []
    for res, sol in zip(contents, solution):
        res:str = res.strip()
        # then load and validate the action
        try:
            action: dict = load_and_validate_action(res)
        except Exception as e:
            scores.append(0.0)
            continue
        
        sub_scores = []
        for k in sol.keys():
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
                    similarity = difflib.SequenceMatcher(None, action[k], sol[k]).ratio()
                    sub_score = similarity
                    print("Text: ",sol[k],", Got: ", action[k],". Similarity: ", similarity)
                    
                case "to":
                    if isinstance(sol[k], list):
                        continue
                    else:
                        if isinstance(action[k],list):
                            sub_score = 0.0
                            print(f"Invalid to for direction {sol[k]}: ", action[k])
                        else:
                            if action[k] == sol[k]:
                                sub_score = 1.0
                            else:
                                sub_score = 0.0
                                print("Invalid to: ", action[k])
                
                case _:
                    if sol[k] is None:
                        if action[k] is None:
                            sub_score = 1.0
                        else:
                            print("Required ", k, ", got: ", action[k])
                            sub_score = 0.0
                    else:
                        if action[k] == sol[k]:
                            sub_score = 1.0
                        else:
                            print("Required ", k, ", got: ", action[k])
                            sub_score = 0.0
                            
            sub_scores.append(sub_score)
        if not sub_scores:
            print("No args to check.")
            scores.append(0.0)
        else:
            scores.append(sum(sub_scores) / len(sub_scores))
    
    return scores

def point_distance_check(completions, solution: list[dict], **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    scores = []
    
    def calculate_dist_score(x,y,gt_x,gt_y):
        dist_score = 0.0
        delta_x = abs(gt_x - x)
        delta_y = abs(gt_y - y)
        max_delta = max(delta_x,delta_y)
        if max_delta > 300:
            dist_score = 0.0
            print("Pixel Distance too large: ", max_delta)
        else:
            dist_score = 1 - max_delta / 1000
        
        return dist_score
    
    for res, sol in zip(contents, solution):
        res:str = res.strip()

        # then load and validate the action
        try:
            action: dict = load_and_validate_action(res)
        except Exception as e:
            scores.append(0.0)
            continue
        
        point_score = None
        if "POINT" in sol:
            if "POINT" not in action:
                point_score = 0.0
            else:
                point_score = calculate_dist_score(*action["POINT"], *sol["POINT"])
        
        to_score = None
        if "to" in sol:
            if "to" not in action:
                to_score = 0.0
            else:
                if isinstance(sol["to"], list):
                    if not isinstance(action["to"],list):
                        print("Must be coordinate to format: ", action)
                        to_score = 0.0
                    else:
                        to_score = calculate_dist_score(*action["to"], *sol["to"])
                else:
                    to_score = None
        
        if point_score is not None and to_score is not None:
            scores.append((point_score + to_score) / 2)
        elif point_score is not None and to_score is None:
            scores.append(point_score)
        elif point_score is None and to_score is not None:
            scores.append(to_score)
        else:
            scores.append(0.0)
    
    return scores
        
def action_match_reward(completions, solution: list[dict], resolution: list[tuple[int,int]],**kwargs):
    contents = [completion[0]["content"] for completion in completions]
    scores = []
    for res, sol, img_res in zip(contents, solution, resolution):
        res:str = res.strip()
        score = 0.0
        
        # first extract the json action from the completion
        action_str = re.search(r'```json(.*?)```', res, re.DOTALL)
        if action_str:
            action_str = action_str.group(1).strip()
        else:
            action_str = res
        
        # then load and validate the action
        try:
            action = json5.loads(action_str)
            jsonschema.validate(action, SCHEMA)
        except jsonschema.ValidationError as e:
            print("Invalid Action Format: ", action)
            scores.append(0.05)
            continue
        except Exception as e:
            print("Error while loading action: ", action_str)
            scores.append(0.0)
            continue
        
        # check if the action is same
        score = 0.1
        
        # type check
        extra_key = set(action.keys()) - set(sol.keys())
        if extra_key:
            score = 0.1
            scores.append(score)
            continue

        for k in sol.keys():
            if k in action:
                score += 0.05
        
        sub_scores = []
        for k in sol.keys():
            if k not in action:
                sub_scores.append(0.0)
                continue
            sub_score = 0.0
            # type check passed, now check args
            match k:
                case "POINT":
                    # calculate the distance between the two points
                    x,y = action[k]
                    gt_x, gt_y = sol[k]
                    h,w = img_res
                    
                    # calculate the score according to the L1 distance
                    l1_dist = abs(gt_x - x) + abs(gt_y - y)
                    max_l1_dist = 1000+1000
                    sub_score = 1 - l1_dist / max_l1_dist
                    
                    if abs(gt_x - x) > 200 or abs(gt_y - y) > 200:
                        sub_score = 0.0

                case "duration":
                    if action[k] > 150 or action[k] < 5000:
                        sub_score = 1.0
                    else:
                        sub_score = 0.0
                
                case "TYPE":
                    similarity = difflib.SequenceMatcher(None, action[k], sol[k]).ratio()
                    sub_score = similarity
                
                case "to":
                    if isinstance(action[k], list):
                        if not isinstance(sol[k],list):
                            sub_score = 0.0
                            break
                        # calculate the distance between the two points
                        x,y = action[k]
                        gt_x, gt_y = sol[k]
                        h,w = img_res
                        
                        # calculate the score according to the L1 distance
                        l1_dist = abs(gt_x - x) + abs(gt_y - y)
                        max_l1_dist = 1000+1000
                        sub_score = 1 - l1_dist / max_l1_dist
                        
                        if abs(gt_x - x) > 200 or abs(gt_y - y) > 200:
                            sub_score = 0.0

                    else:
                        if isinstance(sol[k],list):
                            sub_score = 0.0
                        else:
                            try:
                                if action[k] != sol[k]:
                                    sub_score = 0.0
                                else:
                                    sub_score = 1.0
                            except:
                                sub_score = 0.0

                
                case _:
                    if action[k] != sol[k]:
                        sub_score = 0.0
                    else:
                        sub_score = 1.0
            
            sub_scores.append(sub_score)
            
        subscore_max = (1 - score) / len(sub_scores)
        for sub_score in sub_scores:
            score += sub_score * subscore_max
            
        scores.append(score)

    return scores

def compact_json_dumps(obj):
    return json.dumps(obj, indent=None, separators=(",", ":"), ensure_ascii=False)


SYSTEM_PROMPT = f"""# Role
你是一名熟悉安卓系统触屏GUI操作的智能体，将根据用户的意图，分析当前界面的GUI元素和布局，生成相应的操作。

# Task
针对用户意图，根据输入的当前屏幕截图、屏幕元素描述，输出下一步的操作。

# Rule
- 以紧凑JSON格式输出__一个__操作
- 输出操作必须遵循Schema约束
- 以行注释（//）的形式描述你的思考过程

# Schema
""" + compact_json_dumps(SCHEMA) + \
"""
# Example Output

{
// 我应该...
"POINT":[100,200]
}"""


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
            img = img.resize((w//3,h//3),resample=Image.Resampling.BILINEAR)
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



# if __name__=="__main__":
    # print(action_match_reward(
    #     [
    #         ({"content": "// hello\n{\n  \"POINT\": [100, 200],\n  \"duration\": 200\n}"},0),
    #     ], 
    #     [{
    #         "POINT": [100, 200],
    #         "duration": 200
    #     }], 
    #     [(1000, 2000)],
    # ))