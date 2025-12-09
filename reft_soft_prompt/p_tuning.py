# step1 导包
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer

# step2 加载数据集
ds = load_dataset("json", data_files="alpaca_data_zh/alpaca_gpt4_data_zh.json")
ds = ds["train"]
print(ds[0])

# step3 数据集预处理
tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-1b4-zh")
print(tokenizer)

def preprocess_function(example):
    MAX_LENGTH = 256
    input_ids,attention_mask,labels=[],[],[]
    instruction = tokenizer("\n".join(["Human: " + example["instruction"], example["input"]]).strip() + "\n\nAssistant: ")
    response = tokenizer(example["output"]+tokenizer.eos_token)
    input_ids= instruction["input_ids"] + response["input_ids"]
    attention_mask= instruction["attention_mask"] + response["attention_mask"]
    labels= [-100]*len(instruction["input_ids"]) + response["input_ids"]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# 对数据集中的每一条样本进行数据预处理
tokenized_ds = ds.map(preprocess_function, remove_columns=ds.column_names)
# print(tokenized_ds)
# print(tokenized_ds[0])

# 查看 tokenized_ds 第 2 条样本的 input_ids 所代表的文本内容
# （也就是模型最终看到的完整 prompt + answer）
# print(tokenizer.decode(tokenized_ds[2]["input_ids"]))

# labels 中 = -100 的部分是被忽略的（通常是 Human 的指令区域）
# 我们过滤掉 -100，只保留真实的回答部分的 token id，然后 decode 成中文文本
# print(tokenizer.decode(list(filter(lambda x: x != -100, tokenized_ds[2]["labels"]))))

# step4 加载模型
model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-1b4-zh", low_cpu_mem_usage=True) # low_cpu_mem_usage=True 表示在 CPU 上运行时，减少内存使用
# print(model)


# p-tuning
from peft import PromptEncoderConfig,PromptEncoderReparameterizationType,get_peft_model,TaskType

config = PromptEncoderConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=10,
    # encoder_reparameterization_type=PromptEncoderReparameterizationType.LSTM,
    # encoder_dropout=0.1,
    # encoder_num_layers=5
    # encoder_hidden_size=1024,
)

model = get_peft_model(model, config)
print(model)
model.print_trainable_parameters()


# step5 配置训练参数
args = TrainingArguments(
    output_dir="./chatbot",          # 输出文件夹：保存模型、日志、checkpoint 的地方
    per_device_train_batch_size=1,   # 每块设备（每块 GPU 或 CPU）上，一次前向传递的样本数
    gradient_accumulation_steps=8,   # 累积多少次梯度，再做一次反向传播和参数更新
    logging_steps=10,                # 每隔多少个 step 记录一次日志（loss 等）
    num_train_epochs=1,              # 整个数据集要被模型“学”多少遍
)

# step6 创建训练器
trainer = Trainer(
    model=model,                     # 模型
    args=args,                       # 训练参数
    train_dataset=tokenized_ds,      # 预处理好的训练集

    # data_collator 的作用：负责把若干条数据组成“一个 batch”
    # 1. 负责对齐 padding（不同序列长度不一致，需要 pad 到相同长度）
    # 2. 自动创建 batch 的 input_ids / attention_mask / labels tensor
    # 3. 确保 label 中 -100 的保留与 pad 的正确处理
    # 4. 根据 tokenizer 的 pad_token_id 自动填充 PAD
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,         # 使用同样 tokenizer 来处理 padding 和特殊 token
        padding=True                 # 开启自动 padding，使得 batch 内所有序列长度一致
    )
)

# step7 训练模型
trainer.train()

# step8 模型推理
print(model.device)
# 如果目前占用的资源是cpu，可以切换到gpu上
model = model.cuda()

ipt = tokenizer(
    prompt="Human: {}\n\nAssistant: ".format("如何提高学习效率？", "").strip(),
    return_tensors="pt" # 返回的张量类型为pt，即pytorch张量
).to(model.device)

# 把model输出的response结果再次转为文本
print(
    tokenizer.decode(
        model.generate(**ipt, max_length=256, do_sample=True)[0],
        skip_special_tokens=True
    )
)
