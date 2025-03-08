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
import math

from concurrent.futures import ProcessPoolExecutor

SCHEMA = {
    "type": "object",
    "description": "执行操作并决定当前任务状态",
    "additionalProperties": False,
    "required": ["thought"],
    "properties": {
        "thought": {
            "type": "string",
            "description": "对当前任务的思考，用于描述当前操作的目的"
        },
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
        "Coordinate": {
            "type": "array",
            "description": "坐标为相对于屏幕左上角位原点的相对位置，并且按照宽高比例缩放到0～1000，数组第一个元素为横坐标x，第二个元素为纵坐标y",
            "items": {
                "type": "integer",
                "minimum": 0,
                "maximum": 1000
            },
            "minItems": 2,
            "maxItems": 2
        },
        "Location": {
            "type": "array",
            "description": "由两个坐标组成的数组，表示一个矩形区域，第一个元素为左上角坐标，第二个元素为右下角坐标",
            "items": {
                "$ref": "#/$defs/Coordinate"
            },
            "minItems": 2,
            "maxItems": 2
        }
    }
}

def compact_json_dumps(obj):
    return json.dumps(obj, indent=None, separators=(",", ":"), ensure_ascii=False)


SYSTEM_PROMPT = f"""# Role
你是一名熟悉安卓系统触屏GUI操作的智能体。

# Task
针对用户意图，根据输入的当前屏幕截图、屏幕元素描述，思考并输出下一步的操作。

# Rule
- 以紧凑JSON格式输出**一个**操作
- 输出操作必须遵循Schema约束
- 通过**注释**描述你的思考过程

# Schema
""" + compact_json_dumps(SCHEMA) + \
"""
# Example Output
/* 当前界面... */
// 用户...
{"POINT":[[123,123],[456,456]]}
```"""



def load_and_validate_action(res:str,):
    # action_str = re.search(r'```json(.*?)```', res, re.DOTALL)
    # if action_str:
    #     action_str = action_str.group(1).strip()
    # else:
    #     action_str = res
    if isinstance(res, str):
        action_str = res
        action = json5.loads(action_str,allow_duplicate_keys=False)
    else:
        action = res
        
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

def _action_type_check(res:str, solution: dict):
    try:
        action = load_and_validate_action(res)
        action_keys = set(action.keys())
        solution_keys = set(solution.keys())
        if "thought" in action_keys:
            action_keys.remove("thought")
        if "thought" in solution_keys:
            solution_keys.remove("thought")
        jaccard_index = len(action_keys & solution_keys) / len(solution_keys.union(action_keys))
        if jaccard_index < 1:
            print("Mismatched keys in action, Expected: ", solution_keys, " Got: ", action_keys)
        return jaccard_index * 5
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

def _action_args_check(res:str, solution: dict, reso: tuple, bbox: list[list]):
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
                if k in action:
                    sub_score = calculate_dist_score(action[k], solution[k], reso, bbox[0])
                else:
                    print("No POINT in action: ", action)
                    sub_score = 0.0
            
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
                    if isinstance(action[k],list):
                        sub_score = calculate_dist_score(action[k], solution[k], reso, bbox[1])
                    else:
                        print(f"Invalid to for direction {solution[k]}: ", action[k])
                        sub_score = 0.0
                    
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
    

def action_args_check(completions, solution: list[dict], resolution, bboxs,**kwargs):
    global global_executor
    futures = [global_executor.submit(_action_args_check,completion[0]["content"],sol,reso,bbox) for completion,sol,reso,bbox in zip(completions,solution,resolution,bboxs)]

    scores = []
    for future in futures:
        try:
            scores.append(future.result(timeout=5))
        except TimeoutError as e:
            print("Timeout while checking type.")
            scores.append(0.0)

    return scores


def calculate_dist_score(pred_loc: list[list[int,int]], gt_loc: list[int,int], res: tuple[int,int], bbox: list[int]):    
    pred_left_top = [int(pred_loc[0][0]/1000*res[0]),int(pred_loc[0][1]/1000*res[1])]
    pred_right_bottom = [int(pred_loc[1][0]/1000*res[0]),int(pred_loc[1][1]/1000*res[1])]
    
    if pred_left_top[0] >= pred_right_bottom[0] or pred_left_top[1] >= pred_right_bottom[1]:
        print("Invalid prediction box: ", pred_left_top, pred_right_bottom)
        return -1.0
    
    if bbox is None or not isinstance(bbox, list):
        dist_score = 0.0
        print("No bbox provided.")
        gt_x = gt_loc[0]
        gt_y = gt_loc[1]
        x_ratio = (pred_loc[0][0] + pred_loc[1][0]) / 2000
        y_ratio = (pred_loc[0][1] + pred_loc[1][1]) / 2000
        delta_x = abs(gt_x - x_ratio)
        delta_y = abs(gt_y - y_ratio)
        max_delta = max(delta_x,delta_y)
        dist_score = - max_delta / 1000
        return dist_score
    
    # x_ratio = x_ratio / 1000
    # y_ratio = y_ratio / 1000
    # est_x = int(res[0] * x_ratio)
    # est_y = int(res[1] * y_ratio)
    # left_top = bbox[0]
    # right_bottom = bbox[1]
    # if left_top[0] <= est_x <= right_bottom[0] and left_top[1] <= est_y <= right_bottom[1]:
    #     dist_score = 0.9
    #     # remain 0.1 for centering
    #     max_delta = max(abs(est_x - (left_top[0] + right_bottom[0]) / 2), abs(est_y - (left_top[1] + right_bottom[1]) / 2))
    #     dist_score += 0.1 * ((1 - max_delta / 1000)**3)
    # else:
    #     dist_score = 0.0
    #     print("Point out of bbox: ", est_x, est_y, " Bbox: ", left_top, right_bottom)
    
    # calculate CIoU score
    left_top = bbox[0]
    right_bottom = bbox[1]
    
    
    # Intersection area
    x1 = max(left_top[0], pred_left_top[0])
    y1 = max(left_top[1], pred_left_top[1])
    x2 = min(right_bottom[0], pred_right_bottom[0])
    y2 = min(right_bottom[1], pred_right_bottom[1])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Compute areas of ground truth and predicted boxes
    gt_area = max(right_bottom[0] - left_top[0], 0) * max(right_bottom[1] - left_top[1], 0)
    pred_area = max(pred_right_bottom[0] - pred_left_top[0], 0) * max(pred_right_bottom[1] - pred_left_top[1], 0)
    
    # IoU calculation with smooth term to avoid division by zero
    iou = inter_area / (gt_area + pred_area - inter_area + 1e-6)
    
    # Centers of ground truth and predicted boxes
    gt_center_x = (left_top[0] + right_bottom[0]) / 2.0
    gt_center_y = (left_top[1] + right_bottom[1]) / 2.0
    pred_center_x = (pred_left_top[0] + pred_right_bottom[0]) / 2.0
    pred_center_y = (pred_left_top[1] + pred_right_bottom[1]) / 2.0
    
    # Squared distance between the centers
    center_distance_sq = (pred_center_x - gt_center_x) ** 2 + (pred_center_y - gt_center_y) ** 2
    
    # Smallest enclosing box
    enc_left = min(left_top[0], pred_left_top[0])
    enc_top = min(left_top[1], pred_left_top[1])
    enc_right = max(right_bottom[0], pred_right_bottom[0])
    enc_bottom = max(right_bottom[1], pred_right_bottom[1])
    c_diag_sq = (enc_right - enc_left) ** 2 + (enc_bottom - enc_top) ** 2 + 1e-6  # add smooth term
    
    # Widths and heights for aspect ratio consistency calculation
    gt_w = right_bottom[0] - left_top[0]
    gt_h = right_bottom[1] - left_top[1]
    pred_w = pred_right_bottom[0] - pred_left_top[0]
    pred_h = pred_right_bottom[1] - pred_left_top[1]
    
    # Compute the aspect ratio penalty term v
    if gt_h == 0 or pred_h == 0:
        v = 0.0
    else:
        angle_gt = math.atan(gt_w / (gt_h + 1e-6))
        angle_pred = math.atan(pred_w / (pred_h + 1e-6))
        v = (4 / (math.pi ** 2)) * (angle_gt - angle_pred) ** 2
    
    alpha = v / (1 - iou + v + 1e-6)
    ciou = iou - (center_distance_sq / c_diag_sq) - alpha * v
    
    return ciou



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
                origin_img = Image.open(os.path.join(self.image_root,img_file.replace("/home/test/test03/lyx/check_gui_data-filter_similar/",""))).convert("RGB")
            except:
                print("Error while loading image: ", img_file)
                return self[random.randint(0,len(self.data)-1)]
            resolution = origin_img.size
            h,w = resolution
            # resize the max height and width to 1000
            max_line = 1024
            if h > max_line:
                w = int(w * max_line / h)
                h = max_line
            if w > max_line:
                h = int(h * max_line / w)
                w = max_line
            img = origin_img.resize((h,w),resample=Image.Resampling.BILINEAR)
            
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
            "fullres_image": origin_img,
            "resolution": resolution,
            "bboxs": [item.get("bbox",None),item.get("bbox2",None)],
            "solution": action,
            "prompt": conv
        }



if __name__=="__main__":
    dataset = GUIRFTDataset("/data3/workhome/luyaxi/VCPM-R1/GUIData/bboxdata/tasks.jsonl")
    from PIL import ImageDraw
    item = dataset[0]
    img = item["fullres_image"]
    draw = ImageDraw.Draw(img)
    draw.rectangle(item["bboxs"][0][0] + item["bboxs"][0][1],outline="red",width=3)
    img.save("test.png")