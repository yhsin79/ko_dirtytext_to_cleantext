import pandas as pd
import re

# 엑셀 불러오기
df = pd.read_excel("ex_15000.xlsx")  # B컬럼이 있는 원본 엑셀
dict_df = pd.read_excel("dict.xlsx")  # 부정어-긍정어 매핑

# 부정어-긍정어 딕셔너리 만들기
neg2pos = dict(zip(dict_df['input'], dict_df['output']))

# 부정어 치환 함수
def replace_negatives(sentence):
    for neg_word, pos_word in neg2pos.items():
        # 단어 단위 치환, 부분 단어 치환 방지
        sentence = re.sub(rf'\b{re.escape(neg_word)}\b', pos_word, sentence)
    return sentence

# C컬럼 생성
df['C'] = df['B'].apply(replace_negatives)

# 결과 저장
df.to_excel("example_result.xlsx", index=False)
print("부정어 치환 완료, 결과 저장됨: example_result.xlsx")
