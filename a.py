# 원본 배열
A = [3, 1, 4, 2, 5, 6]

# 합 배열 만들기
S = [0] * len(A)
S[0] = A[0]
for i in range(1, len(A)):
    S[i] = S[i-1] + A[i]

# 구간 합 구하기 (인덱스 2부터 4까지)
start, end = 2, 4
result = S[end] - S[start-1]  # 15 - 4 = 11
print(result)