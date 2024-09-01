import json
from models import GptIntApi, QwenApi
from models import Conversation
import random
import logging
from tqdm import tqdm
import uuid
import os
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import torch.nn.functional as F
import copy
import numpy as np
import bisect
def get_label(premise, hypothesis):
    input = tokenizer_relation(premise, hypothesis, truncation=True, return_tensors="pt").to(device)
    output = model_relation(input["input_ids"].to(device))
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    label_names = ["包含", "中性", "矛盾"]
    prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
    max_label = max(prediction, key=prediction.get)
    return max_label



# 获取prompt
def get_prompt_data(directory, filename):
    for root, dirs, files in os.walk(directory):
        if filename in files:
            # 构建文件的完整路径
            filepath = os.path.join(root, filename)
            # 打开并读取文件内容
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                # 找到文件后退出循环
                return content
# 获取GPT模型的回答
def _internal_model_generate(model, t_trace_id, p_score_prompt: str):
    prompt_data = {
        "trace_id": t_trace_id,
        "questionText": str([{'user': p_score_prompt}]),
        "type": "gpt-4-turbo",
        "choices": {
            "top_p": 0.85,
            "top_k": 1,
            "temperature": 1,
            "max_new_tokens": 3000
        }
    }
    data = Conversation(prompt_data, gpt_internal_api=True)
    return model.generate_single(data)
# 获取QWEN模型的回答
def _internal_model_generate_QWEN(model, t_trace_id: str, p_score_prompt: list):
    if not t_trace_id:
        t_trace_id = str(uuid.uuid4())
    prompt_data = {
        "trace_id": t_trace_id,
        "system_content":'',
        "messages": p_score_prompt,
        "top_p": 0.85,
        "top_k": 1,
        "temperature": 1,
        "max_new_tokens": 1000
    }
    data = Conversation(prompt_data, gpt_internal_api=False)
    return model.generate_single(data)

# 第一个问题的复杂化
def question_1_complication(text):
    PROMPT_Deepening = get_prompt_data(r'E:\graduate_study2\multi_turn_training_data\PROMPT', 'Deepening')
    PROMPT_Concretizing = get_prompt_data(r'E:\graduate_study2\multi_turn_training_data\PROMPT', 'Concretizing')
    PROMPT_Increased = get_prompt_data(r'E:\graduate_study2\multi_turn_training_data\PROMPT', 'Increased_Reasoning_Steps')
    PROMPT = random.choice([PROMPT_Deepening, PROMPT_Concretizing, PROMPT_Increased])
    prompt = PROMPT.format(origin_instruction=text)
    generated_data = _internal_model_generate(model_gpt, '1', prompt)
    return generated_data

def correct_answer_prompt(origin_answer, inst, history):
    PROMPT = get_prompt_data(r'E:\graduate_study2\multi_turn_training_data\PROMPT', 'correct_answer')
    prompt = PROMPT.format(origin_answer=origin_answer, inst=inst, history=history)
    generated_data = _internal_model_generate(model_gpt, '1', prompt)
    return generated_data


# 后续问题的复杂化
def question_2_complication(text, history):
    PROMPT_Deepening = get_prompt_data(r'E:\graduate_study2\multi_turn_training_data\PROMPT', 'Deepening_2')
    PROMPT_Concretizing = get_prompt_data(r'E:\graduate_study2\multi_turn_training_data\PROMPT', 'Concretizing_2')
    PROMPT_Increased = get_prompt_data(r'E:\graduate_study2\multi_turn_training_data\PROMPT', 'Increased_Reasoning_Steps_2')
    PROMPT = random.choice([PROMPT_Deepening, PROMPT_Concretizing, PROMPT_Increased])
    prompt = PROMPT.format(origin_instruction=text, history=history)
    generated_data = _internal_model_generate_QWEN(model_Qwen, '1', [{"role": "user", "content" : prompt}])
    return generated_data
# 生成候选高级指令合集
def advance_inst_generate(inst, history):
    res = []
    for item in tqdm(inst):
        # 准备输入数据
        inputs = tokenizer(history, item, padding=True, truncation=True, return_tensors="pt")
        # 进行推理预测
        with torch.no_grad():
            outputs = ranker_model(**inputs)
            logits = outputs.logits
            # 获取预测结果
        probabilities = F.softmax(logits, dim=-1)
        # 打印logits、outputs和概率值
        # print(probabilities[0][1])
        if probabilities[0][1] > 0.4:
            res.append(item)
    if len(res) > 5:
        res = random.sample(res, 5)
    return res
# 确定最终生成的高级指令
def confirm_inst(inst, history):
    PROMPT_INS = get_prompt_data(r'E:\graduate_study2\multi_turn_training_data\PROMPT', 'inst')
    prompt = PROMPT_INS.format(Instructional_Strategies=inst, history=history)
    generated_data = _internal_model_generate(model_gpt, "text", prompt)
    try:
        parsed_data = json.loads(generated_data)
        return parsed_data["高级指令策略"], parsed_data["具体指令"]
    except:
        return 0
def question_is_equal(question1, question2):
    PROMPT_Equal = get_prompt_data(r'E:\graduate_study2\multi_turn_training_data\PROMPT', 'Equal')
    prompt = PROMPT_Equal.format(first_instruction=question1, second_instruction=question2)
    if '不' in prompt:
        return True
    return False

# 判断回答和问题是否是上下文关系
def question_is_continue(history, question):
        # 准备输入数据
    inputs = tokenizer(history, question, padding=True, truncation=True, return_tensors="pt")
    # 进行推理预测
    with torch.no_grad():
        outputs = model_text_by(**inputs)
        logits = outputs.logits
        # 获取预测结果
    predictions = torch.argmax(logits, dim=-1)
    if predictions[0]== 1:
        return True
    return False

def question_is_ok(question1, question2, history):
    if question_is_equal(question1, question2) and question_is_continue(history, question2):
        return True
    return False

def process_question(topic):
    PROMPT = get_prompt_data(r'E:\graduate_study2\multi_turn_training_data\PROMPT', 'inst_question_1')
    prompt = PROMPT.format(topic=topic)
    generated_data = _internal_model_generate(model_gpt, "text", prompt)
    return [generated_data]
def inst_merge(inst):
    for i in range(len(inst)):
        for j in range(i + 1, len(inst)):
            if get_label(inst[i], inst[j]) == "矛盾":
                return False
    return True
def open_relation_number():
    with open(r'E:\graduate_study2\multi_turn_training_data\data\relation_10.txt', 'r', encoding='utf-8') as f:
        # 读取所有行
        lines = f.readlines()
    # 处理每一行
    data = []
    for line in lines:
        # 使用strip()去除行尾的换行符，然后使用split('\t')以制表符为分隔符分割每行的数据
        row = line.strip().split('\t')
        # 将分割后的数据（字符串列表）转换为适当的数据类型，这里假设原始数据是整数类型
        row = [int(x) for x in row]
        data.append(row)
    return data

def find_random_index(arr):
    # 步骤1: 创建一个空列表
    valid_indices = []
    # 步骤2: 遍历数组
    for i, value in enumerate(arr):
        if value == 2 or value == 1:
            valid_indices.append(i)
            # 步骤3: 随机选择
    if valid_indices:
        return random.choice(valid_indices)
    else:
        # 返回一个特定的值表示没有找到
        return None

def find_intersection(arr1, arr2):
    # 使用集合来找到交集
    return list(set(arr1) & set(arr2))

def choose_sample(data, number):
    res = []
    number1 = random.randint(0, len(data) - 1)
    res.append(number1)
    arr = [j for j in range(len(data))]
    for i in range(number):
        for item in res:
            arr = find_random_index(data[item])
        if arr:
            res.append(arr)
        else:
            return
    return res

def get_prob(n): # 计算概率分布
    sum0 = 0
    for i in range(n):
        sum0 += (1/(i + 1)**2)
    res = [(1/(i + 1)**2)/sum0 for i in range(n)]
    for i in range(1, n):
        res[i] += res[i - 1]
    return res
def inst_structure(text, data, func):
    text_now = copy.deepcopy(text)
    text_now[-1]['content'] = text_now[-1]['content'] + "\n 注意：本轮回答仅需满足以下要求，请务必忽略前几轮的要求！\n 要求如下：\n"
    for i, item in enumerate(data):
        text_now[-1]['content'] = f"{text_now[-1]['content']}\n{i + 1}. {item}"
    res = _internal_model_generate_QWEN(model_Qwen, "text", text_now)
    func_str = func
    bad_case = []
    for j, item in enumerate(func_str):
        for i, it in enumerate(item):
            exec(it[0], globals())
            # 现在 evaluate 函数已经在当前命名空间定义好了
            # 你可以直接调用它并传入 res 作为参数
            result = evaluate(res)
            if result == True:
                break
            elif i == len(item) - 1:
                bad_case.append(j+1)
    if bad_case:
        return res, text_now, False, bad_case
    return res, text_now, True

def inst_right(answer, func):
    func_str = func
    for item in func_str:
        for i, it in enumerate(item):
            exec(it[0], globals())
            # 现在 evaluate 函数已经在当前命名空间定义好了
            # 你可以直接调用它并传入 res 作为参数
            result = evaluate(answer)
            if result == True:
                break
            elif i == len(item) - 1:
                return False
    return True
def answer_quality(query, response):
    scores = []
    PROMPT_answer_score = get_prompt_data(r'E:\graduate_study2\multi_turn_training_data\PROMPT', 'answer_score')
    prompt = PROMPT_answer_score.format(query=query, response=response)
    generated_data = _internal_model_generate(model_gpt, "text", prompt)
    score = re.findall(r'Score: (\d+?)$', generated_data)
    if not score:
        return False
    scores.append(int(score[0]))
    score = np.mean(scores) if scores else 0
    if score > 7: # quality score
        return True
    return False


if __name__ == '__main__':
    # 加载分词器和模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    ranker_model = BertForSequenceClassification.from_pretrained('./ranker_model/')
    # 加载分词器和模型
    model_text_by = BertForSequenceClassification.from_pretrained('./model_text_by/')
    device = torch.device("cpu")
    tokenizer_relation = AutoTokenizer.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
    model_relation = AutoModelForSequenceClassification.from_pretrained(
        "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7").to(device)
    model_url_qwen = 'http://1411998742277274.cn-wulanchabu.pai-eas.aliyuncs.com/api/predict/rlhf_qwen_2_72b_clone/v1/chat/completions'
    user_token_qwen = 'OGViNTcxOTNiZDUwNzYxM2RjMzJiZDA1ZThlMTU1ZTc0YmEzODNlNg=='
    model_url_gpt = 'http://61.129.116.136:8651/chatCompletion'
    user_token_gpt = '240151_aN8z2gtiudKoinIfvcwd'
    model_Qwen = QwenApi(model_url_qwen, token=user_token_qwen)
    model_gpt = GptIntApi(model_url_gpt, token=user_token_gpt)
    with open(r'E:\graduate_study2\multi_turn_training_data\data\cross_validation_10.jsonl', 'r', encoding='utf-8') as f:
        data_inst_structure = [json.loads(line) for line in f]
    inst = ['获取详细信息并确认', '请求详细解释和定义', '引入新元素并提出问题', '请求具体技术示例', '内容生成与优化', '聚焦核心要素与情感', '详细操作信息交流', '确保内容完整规范', '反馈与正向激励', '获取项目资金需求', '探索艺术的多样表现', '引导信息扩展与推进', '深入探讨与应用', '利用外部资源支持', '寻求优化和解决方案', '寻求详细信息和反馈', '逐层深入探讨', '优化和扩展内容', '明确需求并设定条件', '优化沟通与深入探讨', '澄清需求并寻求具体指导', '拓展与创新思维', '通过示例和反馈提升理解', '分解复杂任务', '突出机会并请求细节', '提炼并保存最终成果', '制定计划并寻求反馈', '激发对能力的思考', '集中资源达成目标', '获取详细信息', '挑战权威以获取真相', '创造性命名策略', '请求恢复先前状态', '问题解决导向对话', '情境分析与需求满足', '逐步深入探讨相关主题', '请求详细信息和背景', '明确需求并寻求详细说明', '引导轻松对话', '探索精神修行路径', '寻求全面和深入的见解', '提升交流与思维', '获取隐私相关信息', '提出新问题和解决方案', '获取详细信息和指导', '寻求解决方案与优化', '寻求学术与学习资源', '澄清需求并提供详细说明', '使用具体示例增强理解', '内容生成与优化', '探索科技公司动态', '扩展知识与视野', '生成主题相关内容', '生成营销推广内容', '寻求紧急健康援助', '验证信息的准确性', '评估方案的潜在缺陷', '促进合作与项目推广', '寻求详细信息和深入分析', '定制化解决方案与优化', '扩展与澄清信息', '明确需求并逐步指导', '资源优化与推广', '制定个性化解决方案', '制定系统化学习计划', '确保对话连贯并详细扩展', '引入新元素并扩展', '请求具体示例和实现方法', '明确需求并提出疑问', '分解任务并提供具体指导', '简化信息呈现', '请求详细解释与示例', '生成创意社交媒体内容', '优先考虑安全与舒适', '阐述项目及其意义', '寻求信息和建议', '调试和验证查询错误', '深入挖掘和扩展信息', '针对特定需求提供解决方案', '寻求信息并澄清概念', '系统化分析与具体执行', '探讨社会责任与影响', '生成优化内容建议', '营造轻松互动', '综合策略与防御', '促进合作与发展', '比较与优化策略', '寻求具体解决方案和实现方法', '请求和提供具体示例', '确保地址栏一致性', '比较区域社会文化', '分析与讨论政策', '细化内容结构', '探索新领域', '根据内容创建表格', '切换话题或回到之前话题', '重复提问以求新的回答', '代词、指示词指代或省略指代词', '生成表格数据']
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    file_handler = logging.FileHandler('output/log', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    with open(r'E:\graduate_study2\multi_turn_training_data\data\question.txt', 'r', encoding='utf-8') as f:
        topic = f.read()
        try:
            topics = eval(topic)
            logging.info("文件内容已成功加载为Python列表格式")
            # 遍历列表并打印每个元素
        except:
            logging.error("文件内容不是有效的Python列表格式")
            exit(1)  # 或者其他错误处理
    for i, topic in enumerate(topics):
        if i <= 35:
            continue
        res = []
        # 生成第一个问题
        instruction = process_question(topic)
        number = random.randint(3, 7)
        logging.info(f"第一个问题迭代次数: {number}")
        for j in tqdm(range(number)):
            k = question_1_complication(instruction[-1])
            instruction.append(k)
        res.append({"role": "user", "content": instruction[-1]})
        # print(res)
        # 开始轮次循环
        T = 0
        round_2 = random.randint(2, 3)
        logging.info(f"轮次数量为: {round_2}")
        while T < round_2:
            inst_round = 1
            while True:
                number_list = choose_sample(open_relation_number(), random.randint(1, 3))
                if not number_list:
                    continue
                data_list = [data_inst_structure[item] for item in number_list]
                inst_room = [item['instruction'] for item in data_list]
                func = [item['eval_func'] for item in data_list]
                m = inst_structure(res, inst_room, func)
                a_q = answer_quality(m[0], m[1])
                if m[2] and a_q:
                    answer = m[0]
                    res = m[1]
                    logging.info(f"question_new{i + 1000}第{inst_round}轮次指令生成成功，指令为{inst_room}")
                    break
                elif a_q:
                    logging.info(f"question_new{i + 1000}第{inst_round}轮次回答合理，但是指令不遵循，不遵循编号为{m[3]}")
                    new_answer = correct_answer_prompt(m[0], inst_room, m[1])
                    answer = new_answer
                    res = m[1]
                    logging.info(f"指令遵循，新回答为{new_answer}")
                    break
                else:
                    logging.info(f"第{inst_round}轮次指令生成失败，指令为{inst_room}")
                    inst_round += 1
                    if inst_round == 5:
                        logging.info(f"该轮指令生成失败，不生成指令！！！！")
                        answer = _internal_model_generate_QWEN(model_Qwen, "text", res)
                        break
            # logging.info(f"此轮指令未生成，使用原始指令。")
            # answer = _internal_model_generate_QWEN(model_Qwen, "text", res)
            # 生成回答
            # 判断回答是否合理
            #
            #
            #
            # 还没做，烦死了这玩意儿
            res.append({"role": "assistant", "content": answer})
            logging.info(f"第{T + 1}轮次开始，当前回答为: {res}")

            # 选择与前面第几轮问答相关
            round_present = int(len(res)//2)
            lst = get_prob(round_present)
            random_sample = random.random()
            index_left = bisect.bisect_left(lst, random_sample)
            chosen_round = round_present - 1 - index_left
            logging.info(f"选择第{chosen_round}轮次的问题")
            question_chosen = res[2 * chosen_round]
            answer_chosen = res[2 * chosen_round + 1]
            answer_present = question_chosen['content'] + answer_chosen['content']

            # 生成高级指令
            advance_inst = advance_inst_generate(inst, answer_present)
            # 生成下一个问题
            final_inst = confirm_inst(advance_inst, answer_present)
            wtf = 0
            while final_inst == 0:
                logging.info(f"-----question{i}出现问题--------")
                final_inst = confirm_inst(advance_inst, answer_present)
                wtf += 1
                if wtf == 5:
                    break
            if wtf == 5:
                break
            question = [final_inst[1]]
            I = 0
            round_I = random.randint(3, 5)
            while I < round_I:
                question.append(question_2_complication(question[-1], answer))
                if not question_is_continue(answer, question[-1]):
                    # question_is_ok(question[-1], question[-2], answer):
                    question.pop()
                I += 1
            res.append({"role": "user", "content": question[-1]})
            T += 1

        # 生成上一个问题的输出格式指令以及回答
        inst_round = 1
        while wtf != 5:
            number_list = choose_sample(open_relation_number(), random.randint(1, 3))
            if not number_list:
                continue
            data_list = [data_inst_structure[item] for item in number_list]
            inst_room = [item['instruction'] for item in data_list]
            func = [item['eval_func'] for item in data_list]
            m = inst_structure(res, inst_room, func)
            a_q = answer_quality(m[0], m[1])
            if m[2] and a_q:
                answer = m[0]
                res = m[1]
                logging.info(f"question_new{i + 10000000086}第{inst_round}轮次指令生成成功，指令为{inst_room}")
                break
            elif a_q:
                logging.info(f"question_new{i + 10000000086}第{inst_round}轮次回答合理，但是指令不遵循，不遵循编号为{m[3]}")
                new_answer = correct_answer_prompt(m[0], inst_room, m[1])
                answer = new_answer
                res = m[1]
                logging.info(f"指令已经遵循，新回答为{new_answer}")
                break
            else:
                logging.info(f"第{inst_round}轮次指令生成失败，指令为{inst_room}")
                inst_round += 1
                if inst_round == 5:
                    logging.info(f"该轮指令生成失败，不生成指令！！！！")
                    answer = _internal_model_generate_QWEN(model_Qwen, "text", res)
                    break
        if wtf != 5:
            res.append({"role": "assistant", "content": answer})
            logging.info(f"最后生成的回答为: {answer}")
        with open(f'./output/question_new1{i + 1000}.json', 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=4)
        logging.info(f"结果已写入文件: ./output/question_new1{i + 1000}.json")