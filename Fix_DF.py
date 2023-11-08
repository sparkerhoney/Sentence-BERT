import pandas as pd

# CSV 파일을 불러옵니다.
csv_path = './data/embedding.csv'
df = pd.read_csv(csv_path)

# 문자열을 리스트로 변환하는 함수를 정의합니다.
def string_to_list(string):
    # NaN 값인 경우 빈 리스트 반환
    if pd.isna(string):
        return []
    
    # 문자열 앞뒤의 불필요한 공백과 대괄호를 제거합니다.
    string = string.strip(" []")
    
    # 공백으로 구분된 각 숫자를 쉼표로 구분된 형태로 변환합니다.
    string = ','.join(string.split())
    
    # 변환된 문자열을 평가하여 리스트로 변환합니다.
    try:
        # 문자열 내의 모든 'e+'를 'e'로 치환합니다.
        string = string.replace('e+', 'e')
        # 문자열 내의 모든 'e-'를 'e'로 치환합니다.
        string = string.replace('e-', 'e')
        
        # 파이썬 리스트로 변환을 시도합니다.
        return eval('[' + string + ']')
    except SyntaxError:
        # 변환 중 에러가 발생한 경우 오류 메시지를 출력하고 빈 리스트를 반환합니다.
        print(f"Error converting string to list: {string}")
        return []

# 모든 임베딩 컬럼에 대해 문자열을 리스트로 변환하는 함수를 적용합니다.
for col in df.columns:
    if 'embedding' in col:
        df[col] = df[col].apply(string_to_list)

# 변환된 데이터프레임을 새로운 CSV 파일로 저장합니다.
df.to_csv('fixed_embeddings.csv', index=False)

print("CSV 파일 변환 완료!")