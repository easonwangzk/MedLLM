# 模型配置
MODEL_CONFIG = {
    "base_model": "aaditya/Llama3-OpenBioLLM-70B",
    "teacher_model": "aaditya/Llama3-OpenBioLLM-70B",
    "student_model": "aaditya/Llama3-OpenBioLLM-70B",
    "max_length": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "num_beams": 1,
    "do_sample": True,
    "repetition_penalty": 1.1,
}

# 训练配置
TRAINING_CONFIG = {
    "output_dir": "./outputs",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 4,  # 减小batch size
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,   # 增加梯度累积步数
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.05,
    "logging_steps": 20,
    "save_steps": 200,
    "eval_steps": 200,
    "fp16": True,
    "deepspeed": "./configs/ds_config.json",
    "max_grad_norm": 1.0,
    "warmup_steps": 100
}

# 蒸馏配置
DISTILL_CONFIG = {
    "temperature": 2.0,
    "alpha": 0.5,  # 蒸馏损失权重
    "beta": 0.5,   # 任务损失权重
    "distill_layers": [0, 4, 8, 12, 16, 20, 24, 28, 32, 36],  # 要蒸馏的层
    "attention_distill": True,  # 是否蒸馏注意力矩阵
    "hidden_distill": True,    # 是否蒸馏隐藏状态
    "max_length": 256,
    "num_beams": 1,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9
}

# RLAIF配置
RLAIF_CONFIG = {
    "reward_model": "aaditya/Llama3-OpenBioLLM-70B",
    "num_rollouts": 1,
    "rollout_batch_size": 2,
    "reward_threshold": 0.7,
    "kl_coef": 0.1,
    "clip_range": 0.2,
    "entropy_coef": 0.01,
    "max_length": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "num_beams": 1
}

# 数据配置
DATA_CONFIG = {
    "train_files": {
        "pqaa": "data/ori_pqaa.json",
        "pqal": "data/ori_pqal.json",
        "pqau": "data/ori_pqau.json"
    },
    "validation_split": 0.1,
    "max_samples": {
        "pqaa": 500,
        "pqal": 1000,
        "pqau": 200
    },
    "max_length": 256,
    "doc_stride": 128,
    "pad_to_max_length": True
}