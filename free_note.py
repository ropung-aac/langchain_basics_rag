from queue import PriorityQueue

class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}
    
    def find(self, x):
        """경로 압축을 사용한 find 연산"""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 경로 압축
        return self.parent[x]
    
    def union(self, x, y):
        """랭크 기반 union 연산"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return
        
        # 랭크가 낮은 트리를 높은 트리 아래로 합치기
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
    
    def is_connected(self, x, y):
        """두 요소가 같은 집합에 속하는지 확인"""
        return self.find(x) != self.find(y)

# 노선별 유니온 파인드 초기화
line_uf = {}
for line_num in range(1, 10):  # 1호선~9호선
    line_uf[line_num] = UnionFind()

def heuristic(fr, to):
    """유니온 파인드를 사용한 환승 예상 횟수 휴리스틱"""
    if fr not in station_lines or to not in station_lines:
        return 0
    
    # 환승 예상 횟수 계산
    fr_lines = set(station_lines.get(fr, []))
    to_lines = set(station_lines.get(to, []))
    
    # 공통 노선이 있는지 확인
    common_lines = fr_lines & to_lines
    
    # 공통 노선이 있으면서 실제로 연결되어 있는지 유니온 파인드로 확인
    for line in common_lines:
        if not line_uf[line].is_connected(fr, to):  # 같은 집합에 속하면 False 반환
            return 0  # 환승 없이 갈 수 있음
    
    # 공통 노선이 없거나 연결되어 있지 않으면 환승 필요
    return 3  # 환승 1회당 3분 추가

def transferTime(fr, to):
    """실제 환승시간 계산"""
    fr_lines = set(station_lines.get(fr, []))
    to_lines = set(station_lines.get(to, []))
    
    # 공통 노선이 없으면 환승
    if not (fr_lines & to_lines):
        return 3  # 환승시간 3분
    return 0

def isNotSameLine(fr, to, current_line=None):
    """실제 노선 추적을 고려하여 환승인지 확인"""
    fr_lines = set(station_lines.get(fr, []))
    to_lines = set(station_lines.get(to, []))
    possible_lines = fr_lines & to_lines
    
    if not possible_lines:
        return True  # 공통 노선이 없으면 환승
    
    if current_line is None:
        return False  # 현재 노선 정보가 없으면 환승 아님으로 가정
    
    # 현재 타고 있는 노선으로 계속 갈 수 있는지 확인
    return current_line not in possible_lines

def aStar(start,end):
    
    # 우선순위 큐에 시작 노드 추가 (경로: [(역명, 환승여부)], 현재 노선)
    pq.put((heuristic(start,end),0,start,[(start, False)],None,[]))
    
    # 순환 노드 방지를 위하여 방문 노드 체크 및 최단 거리 저장용용
    visited = dict()
    
    # 우선순위 큐가 비어있지 않을 때 까지 반복
    while not pq.empty():
        
        # 우선순위 큐에서 노드 추출
        target = pq.get()
        
        # 현재 노드의 휴리스틱 비용
        heuristicCost = target[0]
        # 현재 노드의 총 비용
        targetWeight = target[1]
        # 현재 노드
        targetNode = target[2]
        # 현재 노드까지의 경로 [(역명, 환승여부)]
        history = target[3]
        # 현재 타고 있는 노선
        current_line = target[4]
        # 환승역 리스트
        transfer_list = target[5]
        
        # 현재 노드가 목표 노드와 같다면 경로와 비용, 환승 정보를 반환
        if targetNode == end:
            route = [station for station, _ in history]
            return targetWeight, route, transfer_list
        
        # 방문했거나 현재 노드까지의 비용이 더 크다면 무시
        if targetNode in visited and visited[targetNode] <= targetWeight:
            continue
        
        # 현재 노드를 방문 처리
        visited[targetNode] = targetWeight
        
        # 현재 노드의 인접 노드 탐색
        for next in adjGraph[targetNode]:
            nextNode = next[0]
            nextWeight = next[1]
            newCost = targetWeight + nextWeight
            
            # 이미 더 좋은 경로로 방문된 노드는 건너뛰기
            if nextNode in visited and visited[nextNode] <= newCost:
                continue
            
            # 다음 노선 결정
            current_lines = set(station_lines.get(targetNode, []))
            next_lines = set(station_lines.get(nextNode, []))
            possible_lines = current_lines & next_lines
            
            if not possible_lines:
                continue  # 연결된 노선이 없으면 건너뛰기
            
            # 현재 노선이 없거나 첫 번째 이동인 경우
            if current_line is None:
                next_line = min(possible_lines)
                is_transfer = False
            else:
                # 현재 노선으로 계속 갈 수 있는지 확인
                if current_line in possible_lines:
                    next_line = current_line
                    is_transfer = False
                else:
                    # 환승 필요
                    next_line = min(possible_lines)
                    is_transfer = True
            
            # 환승시간 추가
            if is_transfer:
                newCost += transferTime(targetNode, nextNode)
            
            # 새로운 환승역 리스트 생성
            new_transfer_list = transfer_list.copy()
            if is_transfer:
                new_transfer_list.append(targetNode)
            
            pq.put((
                newCost + heuristic(nextNode,end),
                newCost,
                nextNode,
                history + [(nextNode, is_transfer)],
                next_line,
                new_transfer_list
            ))
    return float('inf'), [], []

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

# 역별 노선 정보 (edges의 연결 관계를 바탕으로 수정)
station_lines = {
    '강남': [2, 9], '교대': [2, 3], '고속터미널': [3, 7, 6], '사당': [2, 4],
    '이수': [4, 7], '서울대입구': [2], '신림': [2], '신대방': [2],
    '여의도': [5, 9], '당산': [2, 9], '합정': [2, 6], '홍대입구': [2, 6],
    '신촌': [2], '이대': [2], '서강대': [6], '용산': [1, 4],
    '서울역': [1, 4], '회현': [4], '명동': [4], '충무로': [3, 4],
    '종로3가': [1, 3, 5], '동대문': [1, 4], '청량리': [1], '왕십리': [2, 5],
    '성수': [2], '건대입구': [2, 7], '선릉': [2, 9], '삼성': [2],
    '잠실': [2, 8], '잠실새내': [2]
}

adjGraph = {}

pq = PriorityQueue()

for fr,to,weight in edges:
    adjGraph.setdefault(fr, []).append((to,weight))
    adjGraph.setdefault(to, []).append((fr,weight))
    
# 유니온 파인드 초기화 실행
def init_union_find():
    """간선 정보를 바탕으로 각 노선의 Union-Find 구조 초기화"""
    # edges에서 연결된 역들을 각 노선별로 union
    for fr, to, weight in edges:
        fr_lines = set(station_lines.get(fr, []))
        to_lines = set(station_lines.get(to, []))
        
        # 공통 노선이 있는 경우 해당 노선에서 union
        common_lines = fr_lines & to_lines
        for line in common_lines:
            line_uf[line].union(fr, to)

# 유니온 파인드 초기화 실행
init_union_find()

start = '강남'
end = '이대'
dist, route, transfer_stations = aStar(start,end)

if dist < float('inf'):
    print(f"경로 찾음 : {dist}분 | 경로:", " → ".join(route))
    
    # 환승역 정보 출력
    if transfer_stations:
        print(f"환승역: {', '.join(transfer_stations)}")
        print(f"총 환승 횟수: {len(transfer_stations)}회")
    else:
        print("환승 없음")
else:
    print("경로를 찾을 수 없습니다.")