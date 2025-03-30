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
    "description": "可用的动作和参数",
    "additionalProperties": False,
    # "required": ["think"],
    "properties": {
        "think": {
            "type": "string",
            "description": "对当前任务的思考，用于描述当前操作的目的"
        },
        "POINT": {
            "description": "点击屏幕上的指定位置",
            "$ref": "#/$defs/Location"
        },
        "to": {
            "description": "组合手势参数",
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
                    "description": "移向到某个位置"
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
            "description": "触发特殊按键",
            "enum": [
                "HOME",
                "BACK",
                "ENTER",
                "APPSELECT"
            ]
        },
        "TYPE": {
            "type": "string",
            "description": "向设备键入文本",
        },
        # "DEEP_LINK": {
        #     "type": "null",
        #     "description": "跳转到最近打开的APP"
        # },
        # "CLEAR": {
        #     "type": "null",
        #     "description": "清空输入框的内容"
        # },
        "STATUS": {
            "type": "string",
            "description": "当任务结束时设置为finish",
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
            # "description": "坐标为相对于屏幕左上角位原点的绝对像素数，数组第一个元素为横坐标x，第二个元素为纵坐标y",
            "items": {
                "type": "integer",
                "minimum": 0,
                "maximum": 1000
            },
            "minItems": 2,
            "maxItems": 2
        },
        # "Location": {
        #     "type": "array",
        #     "description": "由两个坐标组成的数组，表示一个矩形区域。两个坐标的连线不平行于X轴或Y轴。",
        #     "items": {
        #         "$ref": "#/$defs/Coordinate"
        #     },
        #     "minItems": 2,
        #     "maxItems": 2
        # }
    }
}

def compact_json_dumps(obj):
    return json.dumps(obj, indent=None, separators=(",", ":"), ensure_ascii=False)




def load_and_validate_action(res:str,):
    action_str = re.search(r'```json(.*?)```', res, re.DOTALL)
    if action_str:
        action_str = action_str.group(1).strip()
    else:
        action_str = res
    action = json5.loads(action_str,allow_duplicate_keys=False)
    # if isinstance(res, str):
    #     action_str = res
    #     action = json5.loads(action_str,allow_duplicate_keys=False)
    # else:
    #     action = res
    
    # action = json5.loads(res,allow_duplicate_keys=False)
    jsonschema.validate(action, SCHEMA)
    return action

global_executor = ProcessPoolExecutor(max_workers=8)

def _action_schema_check(res:str,solution: dict):
    try:
        action:dict = load_and_validate_action(res)
        if "```json" in res:
            return 0.5
        return 1.0
    except jsonschema.ValidationError as e:
        return 0.3
    except Exception as e:
        return 0.0

def action_schema_check(completions, **kwargs):
    global global_executor
    futures = [global_executor.submit(_action_schema_check,completion[0]["content"],) for completion in completions]
    scores = []
    for future in futures:
        try:
            scores.append(future.result(timeout=5)*0.3)
        except TimeoutError as e:
            print("Timeout while checking schema.")
            scores.append(0.0)

    return scores

def _action_type_check(res:str, solution: dict):
    try:
        action = load_and_validate_action(res)
        if not ("think" in action or res.startswith("//") or res.startswith("/*")):
            raise Exception("No think.")
        action_keys = set(action.keys())
        solution_keys = set(solution.keys())
        if "think" in action_keys:
            action_keys.remove("think")
        if "think" in solution_keys:
            solution_keys.remove("think")
        
        # jaccard_index = len(action_keys & solution_keys) / len(solution_keys.union(action_keys))
        # if jaccard_index < 1:
            # print("Mismatched keys in action, Expected: ", solution_keys, " Got: ", action_keys)
        # score = jaccard_index
        # score = max(0,score)
        
        score = len(action_keys & solution_keys)  == len(solution_keys.union(action_keys))
        
        if "```json" in res:
            return score * 0.9
        return score
    
    except Exception as e:
        return -1
    

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
        # if '```json' in res:
        #     raise Exception("Invalid format")
        
        if not ("think" in action or "//" in res or ("/*" in res and '*/' in res)):
            raise Exception("No think.")

        action_keys = set(action.keys())
        solution_keys = set(solution.keys())
        if "think" in action_keys:
            action_keys.remove("think")
        if "think" in solution_keys:
            solution_keys.remove("think")
            
        if len(action_keys & solution_keys) != len(solution_keys.union(action_keys)):
            recall = len(action_keys & solution_keys) / len(solution_keys)
            # rescale to -0.95 to -0.8
            return -0.95 + (recall * 0.15)
            
    except jsonschema.ValidationError as e:
        return -0.95 
    except Exception as e:
        return -1

    score_penalty = 0.0
    sub_scores = []
    
    for k in solution.keys():
        if k == "think":
            continue
        if k not in action:
            sub_scores.append(-1)
            continue
        sub_score = 0
        match k:
            case "POINT":
                sub_score += calculate_dist_score(action[k], solution[k], reso, bbox[0])
            
            case "duration":
                if action[k] > 150 or action[k] <= 5000:
                    sub_score += 1.0
                else:
                    sub_score -= 1.0
                    print("Invalid duration: ", action[k])
            
            case "TYPE":
                similarity = difflib.SequenceMatcher(None, action[k], solution[k]).ratio()
                sub_score += similarity
                # print("Text: ",solution[k],", Got: ", action[k],". Similarity: ", similarity)
                
            case "to":
                if isinstance(solution[k], list):
                    # point direction
                    if isinstance(action[k],list):
                        sub_score += calculate_dist_score(action[k], solution[k], reso, bbox[1])
                    else:
                        sub_score -= 1.0
                        print(f"Invalid to for direction {solution[k]}: ", action[k])
                    
                else:
                    # text direction
                    if isinstance(action[k],list):
                        sub_score -= 1.0
                        print(f"Invalid to for direction {solution[k]}: ", action[k])
                    else:
                        if action[k] == solution[k]:
                            sub_score += 1.0
                        else:
                            sub_score -= 1.0
                            print("Invalid to: ", action[k])
            
            case _:
                if solution[k] is None:
                    if action[k] is None:
                        sub_score += 1.0
                    else:
                        sub_score -= 1.0
                        # print("Required ", solution[k], ", got: ", action[k])
                else:
                    if action[k] == solution[k]:
                        sub_score += 1.0
                    else:
                        sub_score -= 1.0
                        # print("Required ", solution[k], ", got: ", action[k])
                        
        sub_scores.append(sub_score)
    if not sub_scores:
        print("No args to check.")
        return 0.0
    else:
        return max(sum(sub_scores) / len(sub_scores),-0.8)
    

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


def calculate_manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def calculate_dist_score(pred_loc: list[list[int,int]], gt_loc: list[int,int], res: tuple[int,int], bbox: list[int]):    
    x_ratio = pred_loc[0]/1000
    y_ratio = pred_loc[1]/1000
    
    gt_x, gt_y = gt_loc
    gt_x_ratio = gt_x /1000
    gt_y_ratio = gt_y /1000
    
    origin_res, now_res = res
    origin_w, origin_h = origin_res
    now_w, now_h = now_res
    
    abs_x = int(x_ratio * origin_w)
    abs_y = int(y_ratio * origin_h)
    gt_abs_x = int(gt_x_ratio * origin_w)
    gt_abs_y = int(gt_y_ratio * origin_h)
    
    if bbox is None or not isinstance(bbox, list):
        # print("No bbox provided.")
        # let assume the bbox is 1%x1% windows
        if ((gt_x_ratio - 1e-2) <= x_ratio <= (gt_x_ratio + 1e-2)) and ((gt_y_ratio - 1e-2) <= y_ratio <= (gt_y_ratio + 1e-2)):
            dist_score =  1.0
        elif ((gt_x_ratio - 1e-1) <= x_ratio <= (gt_x_ratio + 1e-1)) and ((gt_y_ratio - 1e-1) <= y_ratio <= (gt_y_ratio + 1e-1)):
            dist_score =  0.3
        else:
            # dist_score = -1
            dist_score = - calculate_manhattan_distance(x_ratio, y_ratio, gt_x_ratio, gt_y_ratio) / 2
    
    else:
        left_top = bbox[0]
        right_bottom = bbox[1]
        
        if (left_top[0] <= abs_x <= right_bottom[0]) and (left_top[1] <= abs_y <= right_bottom[1]):
            dist_score = 0.9
            # remain 0.1 for centering
            max_delta = max(abs(abs_x - (left_top[0] + right_bottom[0]) / 2), abs(abs_y - (left_top[1] + right_bottom[1]) / 2))
            dist_score += 0.1 * ((1 - max_delta / 1000)**3)
        elif ((left_top[0] - 0.1*origin_w) <= abs_x <= (right_bottom[0] + 0.1*origin_w )) and ((left_top[1] -0.1*origin_h) <= abs_y <= right_bottom[1] + 0.1*origin_h):
            dist_score = 0.3
        else:
            # print(f"Point {(x_ratio,y_ratio)} {[abs_x,abs_y]} out of Bbox {[left_top, right_bottom]}, GT: {(gt_x_ratio,gt_y_ratio)} {[gt_abs_x,gt_abs_y]}")
            # dist_score = -1
            dist_score = - calculate_manhattan_distance(x_ratio, y_ratio, gt_x_ratio, gt_y_ratio) / 2
    
    return dist_score
    
    # origin_res, now_res = res
    # origin_w, origin_h = origin_res
    # now_w, now_h = now_res
    
    # x, y = pred_loc
    # gt_x, gt_y = gt_loc
    # gt_x_ratio = gt_x /1000
    # gt_y_ratio = gt_y /1000
    # x_ratio = x / now_w
    # y_ratio = y / now_h
    
    # if x_ratio > 1 or y_ratio > 1:
    #     print("Invalid prediction coordinate: ", pred_loc)
    #     return -1.0
    
    # abs_x = int(x_ratio * origin_w)
    # abs_y = int(y_ratio * origin_h)
    
    
    # if bbox is None or not isinstance(bbox, list):
    #     # print("No bbox provided.")
    #     dist_score = - calculate_manhattan_distance(x_ratio, y_ratio, gt_x_ratio, gt_y_ratio) / 2
        
    # else:
    #     left_top, right_bottom = bbox
    #     if left_top[0] <= abs_x <= right_bottom[0] and left_top[1] <= abs_y <= right_bottom[1]:
    #         dist_score = 0.9
    #         # remain 0.1 for centering
    #         max_delta = max(abs(abs_x - (left_top[0] + right_bottom[0]) / 2), abs(abs_y - (left_top[1] + right_bottom[1]) / 2))
    #         dist_score += 0.1 * ((1 - max_delta / 1000)**3)
    #     else:
    #         print(f"Point {(x_ratio,y_ratio)} {[abs_x,abs_y]} out of Bbox {[left_top, right_bottom]}, GT: {(gt_x_ratio,gt_y_ratio)} {gt_loc}")
    #         dist_score = - calculate_manhattan_distance(x_ratio, y_ratio, gt_x_ratio, gt_y_ratio) / 2
    
    # return dist_score
    
    # 绝对坐标iou
    
    # left = min(pred_loc[0][0], pred_loc[1][0])
    # top = min(pred_loc[0][1], pred_loc[1][1])
    # right = max(pred_loc[0][0], pred_loc[1][0])
    # bottom = max(pred_loc[0][1], pred_loc[1][1])
    
    # origin_res, now_res = res
    # origin_w, origin_h = origin_res
    # now_w, now_h = now_res
    
    # pred_left_top = [int(left/now_w*origin_w),int(top/now_h*origin_h)]
    # pred_right_bottom = [int(right/now_w*origin_w),int(bottom/now_h*origin_h)]
    
    # if pred_left_top[0] >= pred_right_bottom[0] or pred_left_top[1] >= pred_right_bottom[1]:
    #     print("Invalid prediction box: ", pred_left_top, pred_right_bottom)
    #     return -1.0
    
    # if bbox is None or not isinstance(bbox, list):
    #     print("No bbox provided.")
    #     gt_x, gt_y = gt_loc
        
    #     delta_x = abs(gt_x/1000 - (left + right) / (now_w * 2))
    #     delta_y = abs(gt_y/1000 - (top + bottom) / (2 * now_h))
    #     max_delta = max(delta_x,delta_y)
    #     dist_score = - max_delta
    #     return dist_score

    # # calculate CIoU score
    # left_top, right_bottom = bbox
    
    # # Intersection area
    # x1 = max(left_top[0], pred_left_top[0])
    # y1 = max(left_top[1], pred_left_top[1])
    # x2 = min(right_bottom[0], pred_right_bottom[0])
    # y2 = min(right_bottom[1], pred_right_bottom[1])
    # inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # # Compute areas of ground truth and predicted boxes
    # gt_area = max(right_bottom[0] - left_top[0], 0) * max(right_bottom[1] - left_top[1], 0)
    # pred_area = max(pred_right_bottom[0] - pred_left_top[0], 0) * max(pred_right_bottom[1] - pred_left_top[1], 0)
    
    # # IoU calculation with smooth term to avoid division by zero
    # iou = inter_area / (gt_area + pred_area - inter_area + 1e-6)
    
    # # Centers of ground truth and predicted boxes
    # gt_center_x = (left_top[0] + right_bottom[0]) / 2.0
    # gt_center_y = (left_top[1] + right_bottom[1]) / 2.0
    # pred_center_x = (pred_left_top[0] + pred_right_bottom[0]) / 2.0
    # pred_center_y = (pred_left_top[1] + pred_right_bottom[1]) / 2.0
    
    # # Squared distance between the centers
    # center_distance_sq = (pred_center_x - gt_center_x) ** 2 + (pred_center_y - gt_center_y) ** 2
    
    # # Smallest enclosing box
    # enc_left = min(left_top[0], pred_left_top[0])
    # enc_top = min(left_top[1], pred_left_top[1])
    # enc_right = max(right_bottom[0], pred_right_bottom[0])
    # enc_bottom = max(right_bottom[1], pred_right_bottom[1])
    # c_diag_sq = (enc_right - enc_left) ** 2 + (enc_bottom - enc_top) ** 2 + 1e-6  # add smooth term
    
    # # Widths and heights for aspect ratio consistency calculation
    # gt_w = right_bottom[0] - left_top[0]
    # gt_h = right_bottom[1] - left_top[1]
    # pred_w = pred_right_bottom[0] - pred_left_top[0]
    # pred_h = pred_right_bottom[1] - pred_left_top[1]
    
    # # Compute the aspect ratio penalty term v
    # if gt_h == 0 or pred_h == 0:
    #     v = 0.0
    # else:
    #     angle_gt = math.atan(gt_w / (gt_h + 1e-6))
    #     angle_pred = math.atan(pred_w / (pred_h + 1e-6))
    #     v = (4 / (math.pi ** 2)) * (angle_gt - angle_pred) ** 2
    
    # alpha = v / (1 - iou + v + 1e-6)
    # ciou = iou - (center_distance_sq / c_diag_sq) - alpha * v
    
    # return ciou



class GUIRFTDataset(Dataset):
    def __init__(self, jsonl_file_path: str, max_line_res: int|None = None, *args, **kwargs):
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
        self.max_line_res = max_line_res

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        
        try:
            # process the conversation
            user_query = item["conversations"][-2]["content"]
            user_query = re.match(r"<Question>(.*?)</Question>", user_query).group(1)
            action = json.loads(item["conversations"][-1]['content'])
        except:
            print("Error while processing conversation.")
            return self[random.randint(0,len(self.data)-1)]
        
        # if list(action.keys()) == ["POINT"]:
        #     # We should skip this case randomly
        #     if random.random() < 0.5:
        #         return self[random.randint(0,len(self.data)-1)]
        
        for img_id,img_file in item["image"].items():
            try:
                if os.path.exists(img_file):
                    origin_img = Image.open(img_file).convert("RGB")
                else:
                    origin_img = Image.open(os.path.join(self.image_root,img_file)).convert("RGB")
            except:
                print("Error while loading image: ", img_file)
                return self[random.randint(0,len(self.data)-1)]
            w,h = origin_img.size
            # resize the max height and width to 1000
            if self.max_line_res is not None:
                max_line = self.max_line_res
                if h > max_line:
                    w = int(w * max_line / h)
                    h = max_line
                if w > max_line:
                    h = int(h * max_line / w)
                    w = max_line
            img = origin_img.resize((w,h),resample=Image.Resampling.LANCZOS)
            
            resolution = (origin_img.size, img.size)
            break
        
        conv = []
        
        def get_random_coordinate():
            return [random.randint(0,1000),random.randint(0,1000)]
        
        conv.append({"role":"system","content":random.choice(SYSTEM_PROMPTS)})
        conv.append({"role": "user", "content": '\n'.join([
                "以下是一些示例操作，您可以参考这些示例来生成您的操作指令：",
                "1. 点击屏幕上的指定位置",
                '// 当前为桌面，需要打开xx软件',
                '{"POINT":'+str(get_random_coordinate())+'}',
                "2. 向上滑动",
                '// 当前界面未找到关键字，需要继续滑动',
                '{"POINT":'+str(get_random_coordinate())+',"to":"up"}',
                "3. 触发特殊按键",
                '// 需要先退回到桌面',
                '{"PRESS":"HOME"}',
                "4. 向设备键入文本",
                '/* 可以向聊天栏键入文本进行回复 */',
                '{"TYPE":"你好"}',
                "5. 结束任务",
                '// 任务已完成',
                '{"STATUS":"finish"}',
                "6. 组合手势参数",
                '// 需要长按以删除',
                '{"POINT":'+str(get_random_coordinate())+',"duration":3000}',
                "7. 等待响应",
                '// 当前界面正在加载，请等待',
                '{"duration":3000}',
                "",
                "你可以将思考过程写在注释中，以便我们了解你的思考过程。当你准备好后，请输出继续的操作指令。"
            ]),})
        conv.append({"role": "assistant", "content": '/* 了解，我需要在注释中进行批判性思考后以JSON格式输出操作指令。目前只是测试我是否能遵循格式，我需要直接输出继续任务的指令*/\n{}'})
        conv.append({"role": "user", "content": [
            f"<Question>{user_query}</Question>\n当前屏幕截图：",
            img, 
        ]})
        
        return {
            "image":img,
            "fullres_image": origin_img,
            "resolution": resolution,
            "bboxs": [item.get("bbox",None),item.get("bbox2",None)],
            "solution": action,
            "prompt": conv
        }

SFT_PROMPT = """# Role
你是一名熟悉安卓系统触屏GUI操作的智能体，将根据用户的问题，分析当前界面的GUI元素和布局，生成相应的操作。

# Task
针对用户问题，根据输入的当前屏幕截图，输出下一步的操作。

# Rule
- 以紧凑JSON格式输出
- 输出操作必须遵循Schema约束

# Schema
""" + compact_json_dumps(SCHEMA)

SYSTEM_PROMPTS = [
SFT_PROMPT,
f"""# Role
一个擅长思考的通用智能体

# Task
思考，理解用户意图，并根据输入的当前屏幕截图等信息输出下一步的动作

# Rule
- 每轮参考 Example Output，以紧凑JSON格式输出**一个**操作
- 输出的动作必须遵循动作空间Schema约束

# 动作空间 Schema
""" + compact_json_dumps(SCHEMA),

f"""# 身份设定
全能决策型数字助手

# 核心职责
解析视觉信息与用户需求，通过多维度推理生成界面交互指令

# 约束条件
- 每次仅生成符合规范的单操作JSON
- 严格匹配下方操作模板结构

# 操作模板
""" + compact_json_dumps(SCHEMA),

f"""# 角色定位
跨平台界面交互决策引擎

# 功能目标
基于屏幕信息流分析，输出最优操作序列节点

# 规范说明
■ 单次响应只允许包含一个标准动作
■ 严格遵守动作参数架构

# 参数架构
""" + compact_json_dumps(SCHEMA),

'''## 智能体特性
多模态交互决策专家

## 执行流程
1. 视觉语义解析
2. 意图推理
3. 生成合规操作

## 硬性要求
- 输出严格遵循JSON schema
- 保持原子化操作（单动作）

## 指令规范
''' + compact_json_dumps(SCHEMA),

"""// 角色：界面导航AI
// 使命：将视觉输入转化为精确操作

'''操作准则'''
1. 单次仅输出一个规范JSON对象
2. 严格匹配操作数据格式

'''动作格式规范'''
""" + compact_json_dumps(SCHEMA),

f"""🤖 智能体类型：界面操作生成器

📌 核心功能：
- 分析屏幕元素布局
- 推导用户潜在意图
- 生成机械可执行指令

🚦 约束条件：
① 每次仅响应单步操作
② 符合预定义指令格式

📜 指令格式手册：
""" + compact_json_dumps(SCHEMA),

"""<AGENT_PROFILE>
类别：自动化决策AI
版本：交互协议

<EXECUTION_POLICY>
1. 单命令输出原则
2. 严格模式：schema验证

<ACTION_SCHEMA>
""" + compact_json_dumps(SCHEMA),

f"""%% 数字操作员系统配置 %%

:: 核心算法 ::
- 计算机视觉理解
- 认知推理引擎
- 操作编码器

:: 输出协议 ::
1. 原子化操作输出
2. 符合API规范

:: 操作API文档 ::
""" + compact_json_dumps(SCHEMA),

f"""# 角色档案
界面导航策略生成器

▲ 核心能力
- 视觉情景理解
- 操作序列规划
- 指令序列化

▲ 输出规范
⚠ 单步操作原则
⚠ 严格类型检查

▼ 类型定义
""" + compact_json_dumps(SCHEMA),

f"""|| 系统角色 ||
界面操作决策中枢

|| 处理流程 ||
① 接收视觉输入
② 生成操作指令
③ 格式合规检查

|| 硬性约束 ||
- 单指令输出模式
- 通过schema验证

|| 指令结构定义 ||
""" + compact_json_dumps(SCHEMA),

f"""⚙️ 机器角色：界面操作编译器

✦ 核心职责
将视觉信号转化为可执行代码

✧ 编译规则
1. 单语句输出原则
2. 类型安全验证

✶ 指令语法
""" + compact_json_dumps(SCHEMA),
]




if __name__=="__main__":
    import yaml
    # print(yaml.dump(SCHEMA, allow_unicode=True, default_flow_style=True))
    print(compact_json_dumps(SCHEMA))
    # dataset = GUIRFTDataset("/data3/workhome/luyaxi/VCPM-R1/GUIData/bboxdata/tasks.jsonl",1120)
    # from PIL import ImageDraw
    # item = dataset[0]
    # img = item["image"]
    # W,H = item["resolution"]
    # draw = ImageDraw.Draw(img)
    # print(item["bboxs"])
    # print(item["resolution"])
    # draw.rectangle([item["bboxs"][0][0][0]/W*img.size[0], item["bboxs"][0][0][1]/H*img.size[1] , item["bboxs"][0][1][0]/W*img.size[0], item["bboxs"][0][1][1]/H*img.size[1]],outline="red",width=3)
    # img.save("test.png")