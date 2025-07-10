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