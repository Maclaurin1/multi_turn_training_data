import json
from models import GptIntApi, QwenApi
from models import Conversation
import random
import logging
from tqdm import tqdm
import uuid
import os
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import re
import torch.nn.functional as F
# 加载分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
ranker_model = BertForSequenceClassification.from_pretrained('./ranker_model/')
# 加载分词器和模型
model_text_by = BertForSequenceClassification.from_pretrained('./model_text_by/')


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
            "temperature": 0,
            "max_new_tokens": 3000
        }
    }
    data = Conversation(prompt_data, gpt_internal_api=True)
    return model.generate_single(data)
# 获取QWEN模型的回答
def _internal_model_generate_QWEN(model, t_trace_id: str, p_score_prompt: str):
    if not t_trace_id:
        t_trace_id = str(uuid.uuid4())
    prompt_data = {
        "trace_id": t_trace_id,
        "system_content":'',
        "messages": [{"role": "user", "content" : p_score_prompt}],
        "top_p": 0.85,
        "top_k": 1,
        "temperature": 1,
        "max_new_tokens": 1000
    }
    data = Conversation(prompt_data, gpt_internal_api=False)
    return model.generate_single(data)

# 第一个问题的复杂化
def question_1_complication(text):
    PROMPT_Deepening = get_prompt_data(r'D:\training_complicating_data\PROMPT', 'Deepening')
    PROMPT_Concretizing = get_prompt_data(r'D:\training_complicating_data\PROMPT', 'Concretizing')
    PROMPT_Increased = get_prompt_data(r'D:\training_complicating_data\PROMPT', 'Increased_Reasoning_Steps')
    PROMPT = random.choice([PROMPT_Deepening, PROMPT_Concretizing, PROMPT_Increased])
    prompt = PROMPT.format(origin_instruction=text)
    generated_data = _internal_model_generate(model_gpt, '1', prompt)
    return generated_data

# 后续问题的复杂化
def question_2_complication(text, history):
    PROMPT_Deepening = get_prompt_data(r'D:\training_complicating_data\PROMPT', 'Deepening_2')
    PROMPT_Concretizing = get_prompt_data(r'D:\training_complicating_data\PROMPT', 'Concretizing_2')
    PROMPT_Increased = get_prompt_data(r'D:\training_complicating_data\PROMPT', 'Increased_Reasoning_Steps_2')
    PROMPT = random.choice([PROMPT_Deepening, PROMPT_Concretizing, PROMPT_Increased])
    prompt = PROMPT.format(origin_instruction=text, history=history)
    generated_data = _internal_model_generate_QWEN(model_Qwen, '1', prompt)
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
    PROMPT_INS = get_prompt_data(r'D:\training_complicating_data\PROMPT', 'inst')
    prompt = PROMPT_INS.format(Instructional_Strategies=inst, history=history)
    generated_data = _internal_model_generate(model_gpt, "text", prompt)
    try:
        parsed_data = json.loads(generated_data)
        return parsed_data["高级指令策略"], parsed_data["具体指令"]
    except:
        return 0
def question_is_equal(question1, question2):
    PROMPT_Equal = get_prompt_data(r'D:\training_complicating_data\PROMPT', 'Equal')
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
    PROMPT = get_prompt_data(r'D:\training_complicating_data\PROMPT', 'inst_question_1')
    prompt = PROMPT.format(topic=topic)
    generated_data = _internal_model_generate(model_gpt, "text", prompt)
    return [generated_data]


if __name__ == '__main__':
    model_url_qwen = ''
    user_token_qwen = ''
    model_url_gpt = ''
    user_token_gpt = ''
    model_Qwen = QwenApi(model_url_qwen, token=user_token_qwen)
    model_gpt = GptIntApi(model_url_gpt, token=user_token_gpt)
    inst = ['获取详细信息并确认', '请求详细解释和定义', '引入新元素并提出问题', '请求具体技术示例', '内容生成与优化', '聚焦核心要素与情感', '详细操作信息交流', '确保内容完整规范', '反馈与正向激励', '获取项目资金需求', '探索艺术的多样表现', '引导信息扩展与推进', '深入探讨与应用', '利用外部资源支持', '寻求优化和解决方案', '寻求详细信息和反馈', '逐层深入探讨', '优化和扩展内容', '明确需求并设定条件', '优化沟通与深入探讨', '澄清需求并寻求具体指导', '拓展与创新思维', '通过示例和反馈提升理解', '分解复杂任务', '突出机会并请求细节', '提炼并保存最终成果', '制定计划并寻求反馈', '激发对能力的思考', '集中资源达成目标', '获取详细信息', '挑战权威以获取真相', '创造性命名策略', '请求恢复先前状态', '问题解决导向对话', '情境分析与需求满足', '逐步深入探讨相关主题', '请求详细信息和背景', '明确需求并寻求详细说明', '引导轻松对话', '探索精神修行路径', '寻求全面和深入的见解', '提升交流与思维', '获取隐私相关信息', '提出新问题和解决方案', '获取详细信息和指导', '寻求解决方案与优化', '寻求学术与学习资源', '澄清需求并提供详细说明', '使用具体示例增强理解', '内容生成与优化', '探索科技公司动态', '扩展知识与视野', '生成主题相关内容', '生成营销推广内容', '寻求紧急健康援助', '验证信息的准确性', '评估方案的潜在缺陷', '促进合作与项目推广', '寻求详细信息和深入分析', '定制化解决方案与优化', '扩展与澄清信息', '明确需求并逐步指导', '资源优化与推广', '制定个性化解决方案', '制定系统化学习计划', '确保对话连贯并详细扩展', '引入新元素并扩展', '请求具体示例和实现方法', '明确需求并提出疑问', '分解任务并提供具体指导', '简化信息呈现', '请求详细解释与示例', '生成创意社交媒体内容', '优先考虑安全与舒适', '阐述项目及其意义', '寻求信息和建议', '调试和验证查询错误', '深入挖掘和扩展信息', '针对特定需求提供解决方案', '寻求信息并澄清概念', '系统化分析与具体执行', '探讨社会责任与影响', '生成优化内容建议', '营造轻松互动', '综合策略与防御', '促进合作与发展', '比较与优化策略', '寻求具体解决方案和实现方法', '请求和提供具体示例', '确保地址栏一致性', '比较区域社会文化', '分析与讨论政策', '细化内容结构', '探索新领域', '根据内容创建表格', '切换话题或回到之前话题', '重复提问以求新的回答', '代词、指示词指代或省略指代词', '生成表格数据']
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    file_handler = logging.FileHandler('log', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    with open(r'D:\training_complicating_data\data\question.txt', 'r', encoding='utf-8') as f:
        topic = f.read()
        try:
            topics = eval(topic)
            logging.info("文件内容已成功加载为Python列表格式")
            # 遍历列表并打印每个元素
        except:
            logging.error("文件内容不是有效的Python列表格式")
            exit(1)  # 或者其他错误处理
    for i, topic in enumerate(topics):
        res = []
        # 生成第一个问题
        instruction = process_question(topic)
        number = random.randint(2, 5)
        logging.info(f"第一个问题迭代次数: {number}")
        for j in tqdm(range(number)):
            k = question_1_complication(instruction[-1])
            instruction.append(k)
            if not question_is_equal(instruction[-1], instruction[-2]):
                break
        res.append({"role": "user", "content": instruction[-1]})
        # print(res)
        # 开始轮次循环
        T = 0
        round = random.randint(2, 5)
        logging.info(f"轮次数量为: {round}")
        while T < round:
            # 生成回答
            answer = _internal_model_generate_QWEN(model_Qwen, "text", str(res))
            # 判断回答是否合理
            #
            #
            #
            # 还没做，烦死了这玩意儿
            res.append({"role": "assistant", "content": answer})
            # 生成高级指令
            advance_inst = advance_inst_generate(inst, answer)
            # 生成下一个问题
            final_inst = confirm_inst(advance_inst, answer)
            if final_inst == 0:
                logging.info(f"-----question{i}出现问题--------")
                break
            question = [final_inst[1]]
            I = 0
            round_I = random.randint(6, 8)
            while I < round_I:
                question.append(question_2_complication(question[-1], answer))
                if not question_is_ok(question[-1], question[-2], answer):
                    question.pop()
                I += 1
            res.append({"role": "user", "content": question[-1]})
            T += 1
        answer = _internal_model_generate_QWEN(model_Qwen, "text", str(res))
        # 判断回答是否合理
        #
        #
        # 还没做，烦死了这玩意儿
        res.append({"role": "assistant", "content": answer})
        logging.info(f"最后生成的回答为: {answer}")
        with open(f'./output/question{i + 7}.json', 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=4)
        logging.info(f"结果已写入文件: ./output/question{i + 7}.json")
