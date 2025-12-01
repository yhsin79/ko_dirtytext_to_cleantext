from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_path = "./version5_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

def clean_text(text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    with torch.no_grad():
        output = model.generate(input_ids, max_length=64, num_beams=5)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 테스트
while True:
    text = input("입력 문장 >> ")
    if text == "exit":
        break
    print("정제 결과:", clean_text(text))
