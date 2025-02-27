import re
import json
import json5
import jsonschema
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image

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
            scores.append(0.0)
            continue
        except Exception as e:
            scores.append(0.0)
            continue
        
        # check if the action is same
        
        # type check
        for k in sol.keys():
            if k not in action:
                score = 0.0
                break
        
            # type check passed, now check args
            match k:
                case "POINT":
                    # calculate the distance between the two points
                    x,y = action[k]
                    gt_x, gt_y = sol[k]
                    h,w = img_res
                    
                    if abs(gt_x - x) / h < 0.05:
                        if abs(gt_y - y) / w < 0.05:
                            score = 1.0
                        else:
                            score = 0.0
                    else:
                        score = 0.0
                    
                    

                case "duration":
                    if action[k] < 150 or action[k] > 5000:
                        score = 0.0
                
                case "TYPE":
                    if sol[k] not in action[k]:
                        score = 0.0
                
                case "to":
                    if isinstance(action[k], list):
                        if not isinstance(sol[k],list):
                            score = 0.0
                            break
                        # calculate the distance between the two points
                        x,y = action[k]
                        gt_x, gt_y = sol[k]
                        h,w = img_res
                        
                        if abs(gt_x - x) / h < 0.05:
                            if abs(gt_y - y) / w < 0.05:
                                score = 1.0
                            else:
                                score = 0.0
                        else:
                            score = 0.0
                    else:
                        if isinstance(sol[k],list):
                            score = 0.0
                        else:
                            try:
                                if action[k] != sol[k]:
                                    score = 0.0
                            except:
                                score = 0.0

                
                case _:
                    if action[k] != sol[k]:
                        score = 0.0
                        
        scores.append(score)

    return scores

def compact_json_dumps(obj):
    return json.dumps(obj, indent=None, separators=(",", ":"), ensure_ascii=False)


SYSTEM_PROMPT = f"""# Role
你是一名熟悉安卓系统触屏GUI操作的智能体，将根据用户的意图，分析当前界面的GUI元素和布局，生成相应的操作。

# Task
针对用户意图，根据输入的当前屏幕截图、屏幕元素描述，输出下一步的操作。

# Rule
- 以紧凑JSON格式输出操作
- 输出操作必须遵循Schema约束

# Schema
{compact_json_dumps(SCHEMA)}

# Example Output
```
// I should click ...
{compact_json_dumps({"POINT":[123,456]})}
```"""


class GUIRFTDataset(Dataset):
    def __init__(self, jsonl_file_path: str, ):
        self.data = []
        with open(jsonl_file_path, "r") as f:
            for line in tqdm(f.readlines(), desc="Loading dataset",dynamic_ncols=True):
                try:
                    self.data.append(json.loads(line))
                except:
                    print("Error while loading line.")
                    continue

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        
        for img_id,img_file in item["images"].items():
            img = Image.open(img_file).convert("RGB")
            break
        
        
        # process the conversation
        user_query = item["conversations"][-2]["content"]
        user_query = re.match(r"<Question>(.*?)</Question>", user_query).group(1)
        action = json.loads(item["conversations"][-1]['content'])
        conv = []
        conv.append({"role":"system","content":SYSTEM_PROMPT})
        conv.append({"role":"user","content":[
            {"type":"image"},
            {"type":"text","text": user_query}
        ]})
        
        return {
            "image":img,
            "solution": action,
            "prompt": conv
        }



if __name__=="__main__":
    print(action_match_reward(
        [
            ({"content": "// hello\n{\n  \"POINT\": [100, 200],\n  \"duration\": 200\n}"},0),
        ], 
        [{
            "POINT": [100, 200],
            "duration": 200
        }], 
        [(1000, 2000)],
    ))