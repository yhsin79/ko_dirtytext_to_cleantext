import pandas as pd
import random

# ---------------------------------------------
# 1. 욕설 리스트 불러오기 (badwords.xlsx → A컬럼)
# ---------------------------------------------
badword_df = pd.read_excel("badwords.xlsx")

# 컬럼명이 없거나 첫 줄이 데이터라면 자동 처리
if 'A' in badword_df.columns:
    negative_words = badword_df['A'].dropna().tolist()
else:
    # header가 없는 경우
    negative_words = badword_df.iloc[:, 0].dropna().tolist()

# ---------------------------------------------
# 부정 문장 생성 함수
# ---------------------------------------------
def make_negative_sentence(sentence):
    # 랜덤 욕설 1~2개 삽입 (원하면 조절 가능)
    insert_count = random.randint(1, 2)
    words = sentence.split()

    for _ in range(insert_count):
        pos = random.randint(0, len(words))
        bad = random.choice(negative_words)
        words.insert(pos, bad)

    return " ".join(words)

# ---------------------------------------------
# 긍정어 엑셀 읽고 B컬럼에 부정어 생성해서 저장
# ---------------------------------------------
df = pd.read_excel("positive_words.xlsx")  # 기존 긍정어 파일
df['B'] = df['A'].apply(make_negative_sentence)
df.to_excel("negative_output.xlsx", index=False)

print("완료! → negative_output.xlsx 생성됨")
