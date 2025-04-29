from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json
from tqdm import tqdm

def generate_rlaif_data():
    # 加载PubMedQA数据集
    dataset = load_dataset("pubmed_qa", "pqa_labeled")
    train_data = dataset["train"]
    
    # 打印数据集信息
    print(f"Dataset type: {type(train_data)}")
    print(f"Dataset length: {len(train_data)}")
    print(f"First item type: {type(train_data[0])}")
    print(f"First item: {train_data[0]}")
    
    # 配置量化参数
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # 加载BioGPT-Large-PubMedQA模型
    model_name = "microsoft/BioGPT-Large-PubMedQA"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # 思维链提示模板
    cot_template = """Please analyze the following medical case step by step:

Context: {context}

Question: {question}

Let's think through this step by step:

1. First, identify the key information from the context:
2. Then, analyze the specific question being asked:
3. Next, consider the relevant medical concepts:
4. After that, evaluate the possible answers:
5. Finally, provide a comprehensive conclusion:

Answer:"""
    
    # 生成RLAIF数据
    rlaif_data = []
    
    for item in tqdm(train_data):
        try:
            question = item["question"]
            context_data = item["context"]
            long_answer = item["long_answer"]
            
            contexts = context_data["contexts"]
            context = " ".join(contexts)
            
            prompt = cot_template.format(context=context, question=question)
            
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            input_length = inputs.input_ids.shape[1]
            if input_length > 1024:
                continue
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
                length_penalty=1.5,
                num_beams=4,
                early_stopping=True
            )
            
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer_part = answer.split("Answer:")[1].strip() if "Answer:" in answer else answer
            cot_part = answer.split("Answer:")[0].strip() if "Answer:" in answer else ""
            
            rlaif_data.append({
                "prompt": prompt,
                "chosen": {
                    "answer": answer_part,
                    "chain_of_thought": cot_part
                },
                "rejected": {
                    "answer": long_answer,
                    "chain_of_thought": ""
                }
            })
            
            if len(rlaif_data) % 10 == 0:
                with open("rlaif_data.json", "w") as f:
                    json.dump(rlaif_data, f, indent=2)
                
        except Exception as e:
            print(f"Error processing item: {e}")
            continue
    
    # 最终保存数据
    with open("rlaif_data.json", "w") as f:
        json.dump(rlaif_data, f, indent=2)
    
    return rlaif_data

if __name__ == "__main__":
    generate_rlaif_data()