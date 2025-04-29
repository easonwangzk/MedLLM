from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

def quantize_model():
    # 配置量化参数
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # 加载模型和tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "./ppo_final_model",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("./dpo_final_model")
    
    # 保存量化后的模型
    model.save_pretrained("./quantized_model")
    tokenizer.save_pretrained("./quantized_model")

def test_model():
    # 加载量化后的模型
    model = AutoModelForCausalLM.from_pretrained(
        "./quantized_model",
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("./quantized_model")
    
    # 测试示例
    prompt = "Context: A 45-year-old male presents with chest pain.\nQuestion: What are the possible causes?\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_length=512,
        temperature=0.7,
        do_sample=True
    )
    
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    quantize_model()
    test_model() 