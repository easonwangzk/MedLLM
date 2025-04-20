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

TRAINING_CONFIG = {
    "output_dir": "./outputs",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.05,
    "logging_steps": 20,
    "save_steps": 200,
    "eval_steps": 200,
    "fp16": True,
}

DISTILL_CONFIG = {
    "temperature": 2.0,
    "alpha": 0.5,
    "beta": 0.5,
    "distill_layers": [0, 4, 8, 12, 16, 20, 24, 28, 32, 36],
    "attention_distill": True,
    "hidden_distill": True,
}

RLAIF_CONFIG = {
    "reward_model": "aaditya/Llama3-OpenBioLLM-70B",
    "num_rollouts": 1,
    "rollout_batch_size": 2,
    "reward_threshold": 0.7,
    "kl_coef": 0.1,
    "clip_range": 0.2,
    "entropy_coef": 0.01,
}
