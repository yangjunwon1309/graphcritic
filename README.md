# Graph Critic

## 결론과 범위

BOOST의 그래프 코어(노드, 간선, 진행도 계산)를 도메인과 무관한 `graphcritic` 라이브러리로 분리해 두 레포가 공유합니다.
BOOST와 본 연구는 같은 LCB 가중 그래프 비용 `C_G`를 사용하고, 그래프 가치의 사용처가 다릅니다.
벤치마크는 BOOST 하드웨어 스택과 분리하며, BOOST 실물 실험은 내보낸 transition과 그래프 파일을 입력으로 받습니다.

## 설치 및 테스트

```bash
pixi install -e core
pixi run -e core pytest
```

NumPy 2.x 환경은 `pixi install -e core-numpy2`와 `pixi run -e core-numpy2 pytest`로 확인합니다.

## 디렉터리

- `graphcritic/`: NumPy/SciPy 기반 core 패키지
- `envs/`: 데이터셋 어댑터 스텁
- `third_party/`: FQL/QC fork URL을 받은 뒤 추가할 자리
- `experiments/`: 그래프 빌드, 실행 래퍼, sweep 및 분석 스텁
- `configs/graph/`: 환경별 그래프 설정
- `tests/`: import, 의존성 경계 및 core 회귀 테스트 스텁
- `docs/`: [명세서](docs/spec.md)와 [수식화](docs/math.md)
