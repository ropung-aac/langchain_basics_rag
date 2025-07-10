import heapq #우선순위 큐(가장 작은 값부터 꺼내는 큐)
from collections import defaultdict


# 입력 데이터(역1, 역2, 이동시간) -> 함수형데이터
def get_edges():
    return [
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

# 그래프 구성 함수 -> 계산
def build_graph(edges):
    graph = defaultdict(list)
    for u, v, w in edges: # 양방향 가중치
        graph.setdefault(u, []).append((v, w))
        graph.setdefault(v, []).append((u, w)) # 양방향
    return dict(graph)

# 휴리스틱 함수 -> 계산
def heuristic(u, v):
    return 0

# 환승 시간 계산 -> 계산
def transfer_time(u, v):
    return 0

# 같은 라인인지 확인 -> 계산
def is_not_same_line(u, v):
    return True

# A* 알고리즘 함수 -> 계산
def a_star(graph, start_station, goal_station):
    current_weight = 0
    # (예상 총 거리, 현재 거리, 현재 위치, 경로 기록)
    queue = [(heuristic(start_station, goal_station), current_weight, start_station, [start_station])]
    visited = dict()

    while queue:
        heuristic_weight, current_time, current_station, routes = heapq.heappop(queue)

        # 목적지 도착 분기
        if current_station == goal_station:
            return current_time, routes

        # 이미 방문하거나 더 빠른 가중치는 PASS
        if current_station in visited and visited[current_station] <= current_time:
            continue
        
        visited[current_station] = current_time

        for next, weight in graph.get(current_station, []):
            next_weight = current_time + weight
            
            if is_not_same_line(current_station, next):
                next_weight += transfer_time(current_station, next)
            
            if next in visited and visited[next] <= next_weight:
                continue
            
            next_heuristic = heuristic(next, goal_station)
            heapq.heappush(queue, (next_weight + next_heuristic, next_weight, next, routes + [next]))

    return float('inf'), [] # 최단거리를 못 찾는 경우

# 사용자 입력 받기 -> 액션
def get_user_input():
    try:
        print("역 이름을 '시작역 도착역' 양식으로 입력")
        start_station, goal_station = input().strip().split()
        return start_station, goal_station
    except:
        print("올바른 형식이 아닙니다. ex) 강남 선릉")
        return None, None
    
# 실행 함수 ->  계산 + 액션
def run_pathfinder():
    start_station, goal_station = get_user_input()
    if not start_station or not goal_station:
        return
    edges = get_edges()
    graph = build_graph(edges)
    total_time, route = a_star(graph, start_station, goal_station)

    if total_time < float('inf'):
        return print(f"시간: {total_time}분\n경로: {' -> '.join(route)}")
    print("경로를 찾을 수 없습니다.")
    
if __name__ == "__main__":
    run_pathfinder()
    
# [쪼개는 판단 기준]
# 이 코드를 나중에 웹 API나 앱에서 재사용할 계획이 있는가?	
# 계산 결과를 따로 테스트하고 싶은가?	
# 입력/출력 방식을 바꿀 가능성이 있는가?	
# 코드가 더 커지고 복잡해질 예정인가?