import heapq #우선순위 큐(가장 작은 값부터 꺼내는 큐)

# 주어진 역(u)에서 역간(v) 걸리는시간(w)
edges = [
    ('강남','교대',2), ('교대','고속터미널',3), ('고속터미널','사당',2),
    ('사당','이수',2), ('이수','서울대입구',3), ('서울대입구','신림',2),
    ('신림','신대방',3), ('신대방','여의도',4), ('여의도','당산',2),
    ('당산','합정',2), ('합정','홍대입구',2), ('홍대입구','신촌',2),
    ('신촌','이대',1), ('이대','서강대',1), ('서강대','용산',3),
    ('용산','서울역',3), ('서울역','회현',1), ('회현','명동',1),
    ('명동','충무로',1), ('충무로','종로3가',2), ('종로3가','동대문',2),
    ('동대문','청량리',4), ('종로3가','왕십리',5), ('왕십리','성수',2),
    ('성수','건대입구',2), ('건대입구','선릉',5), ('선릉','삼성',2),
    ('삼성','잠실',3), ('잠실','잠실새내',1), ('교대','선릉',7),
    ('강남','선릉',4), ('당산','용산',5), ('고속터미널','홍대입구',10),
]

# 그래프 생성
graph = {}
for u, v, w in edges: # 양방향 가중치
    graph.setdefault(u, []).append((v, w))
    graph.setdefault(v, []).append((u, w)) 

# TODO 추후 특이사항 시 고려해 볼 수 있는 분기
def heuristic(u, v): # 정확한 예측이아닌 방향성 제공(목표방향 가점 반대방향 감점)
    return 0

# 환승 시간 계산
def transferTime(u, v):
    return 0

# 노션 변경 시간 계산
def isNotSameLine(u, v):
    return True

def a_star(start_station, goal_station):
    current_weight = 0
    queue = [(heuristic(start_station, goal_station), current_weight, start_station, [start_station])]
    visited_record = dict() # 방문기록

    while queue:
        heuristic_weight, current_time, current_station, routes = heapq.heappop(queue)

        if current_station == goal_station: # 성공 시
            return current_time, routes

        if current_station in visited_record and visited_record[current_station] <= current_time: # 이미 최단경로니까 다음으로
            continue
        
        visited_record[current_station] = current_time # 업데이트

        for next, weight in graph.get(current_station, []):
            next_weight = current_time + weight
            
            if isNotSameLine(current_station, next):
                next_weight += transferTime(current_station, next)
            
            if next in visited_record and visited_record[next] <= next_weight:
                continue
            next_heuristic = heuristic(next, goal_station)
            
            # pq
            heapq.heappush(queue, (next_weight + next_heuristic, next_weight, next, routes + [next]))

    return float('inf'), []

if __name__ == "__main__":
    try:
        print("역 이름을 '시작역 도착역' 양식으로 입력")
        start_station, end_station  = input().split(" ")
    except:
        print("역 이름을 '시작역 도착역' 양식으로 입력 하지 않았습니다.")
    total_time, routes = a_star(start_station, end_station)
    if total_time < float('inf'):
        print(f"경로: {total_time}분 | 경로:", " → ".join(routes))
    else:
        print("경로를 찾을 수 없습니다.")
