# Graph Critic 실험 레포 명세서 (NeurIPS 2027)

_2026년 10월 31일 · @JUNWON YANG_

## 0. 결론과 범위

가능하다. 전제는 BOOST의 그래프 코어(노드, 간선, 진행도 계산)를 도메인과 무관한 라이브러리 graphcritic으로 분리해 두 레포가 공유하는 것이다.

- BOOST도 LCB 가중치 그래프 비용 C_G를 그대로 쓴다. 두 논문은 그래프와 간선 가중치까지 같은 코드를 쓰고, 그래프 가치의 사용처만 다르다.
- 벤치마크 실험은 BOOST 하드웨어 스택과 분리된 JAX 레포에서 한다. FQL과 Q-chunking(QC) 공식 구현을 fork해 critic 손실만 교체한다.
- BOOST 실물 실험은 BOOST가 내보낸 transition과 그래프 파일을 받아 가중치만 계산해 돌려준다. 실험 레포는 BOOST 코드를 import하지 않는다.
범위 밖: 이미지 관측 벤치마크, OGBench 100M 규모 데이터, 대규모 online sim 실험. 1차 결과가 나온 뒤 재검토한다.

## 1. BOOST 그래프와 본 논문 방법의 대응

두 방법은 그래프 구성과 간선 가중치(LCB)까지 같고, 그래프 가치의 사용처만 다르다.

| 구성 | BOOST (IROS) | 본 논문 (NeurIPS) | core에서의 처리 |
| --- | --- | --- | --- |
| 노드 | n 프레임 keyframe, 매칭 상태 s^m, 세그먼트 k, 위상 φ, 정규화 시간 τ | ψ 공간에서 반경 r로 병합, 대표는 medoid | 노드 빌더를 플러그인으로 둠. BOOST는 자체 빌더 사용 |
| 간선 | 시연 순서 + 유사도(Tol) 간선, 같은 접촉 b와 세그먼트 k, τ 단조 | 데이터 chunk 전이, run-length 압축 | 간선 빌더 + EdgeFilter. BOOST의 제약은 필터로 표현 |
| 간선 가중치 | w_e = −log ℓ_e + c·κ_e | 같음 | WeightPolicy=lcb. hop과 time은 ablation |
| 그래프 가치 | C_G(v): 역방향 다중출발 Dijkstra | 같음 | 같은 solver |
| 실패 rollout | goal 간선 없이 편입. 도달 불가 노드는 C_max로 클리핑 | 같음 | 같음 |
| transition 가치 | −y (한 간선 = 한 keyframe) | critic target y | targets.py |
| 사용처 | GSR식 가중치로 손 expert 가중 SFT, guidance subgoal | 비용 도메인 critic 증류 후 기존 actor | 각 레포 |

BOOST의 transition 가치는 본 논문 target y에 부호만 바꾼 것이다. 기존 BOOST 계획의 hop 진행도는 c = 1, h = 1, ℓ_e ≡ 1인 특수 경우로, ablation과 회귀 테스트로만 남긴다.

$$
\tilde Q(v_i) = -\bigl(1 + c(v_{i+1})\bigr) = -\,y\,\big|_{\,c=1,\ h=1,\ \ell_e \equiv 1}
$$

따라서 BOOST의 graph/value.py는 WeightPolicy=lcb으로 core solver를 호출하는 얇은 래퍼가 된다. 위 hop 등식은 core의 회귀 테스트로 고정한다(4.4절).

## 2. 레포 전략

새 레포 graph-critic(가칭)을 만들고, 그 안의 core 패키지 graphcritic을 BOOST가 tag로 pin해서 쓴다.

### 의존 방향

- graphcritic(core)은 numpy와 scipy만 쓴다. jax와 torch를 import하지 않는다.
- BOOST는 pixi git 의존성으로 core를 tag(core-v0.1 등)에 고정해 쓴다.
- 실험 코드(envs/, experiments/)는 core와 fork된 베이스라인(third_party/fql, third_party/qc)에 의존한다.
- 실험 레포는 BOOST를 import하지 않는다. BOOST가 내보낸 파일(6.3절 boost_export)만 읽는다.
실선은 코드 의존, 점선은 파일 교환이다. 실험 레포는 BOOST 코드를 import하지 않는다.

### BOOST 안에 두지 않는 이유

- BOOST는 GR00T, FoundationPose, 실물 드라이버에 묶인 하드웨어 스택이다. 벤치마크는 FQL/QC가 요구하는 jax, ogbench, gymnasium 고정 버전이 필요하다.
- BOOST develop은 "항상 sim에서 동작" 규칙을 지킨다. 대규모 벤치마크 sweep을 같은 브랜치에 섞으면 그 규칙이 깨진다.
- IROS 마감(2027-03-01) 전후로 BOOST는 동결 구간이 생긴다. 벤치마크 일정이 그에 묶이면 안 된다.
### core를 복사하지 않고 패키지로 공유하는 이유

BOOST와 본 논문의 C_G는 같은 가중치와 같은 solver에서 나와야 한다. 복사본이 갈라지면 BOOST 실물 실험(E6)의 그래프 가치가 벤치마크와 다른 코드에서 나오게 된다.

### NumPy 호환

BOOST의 hand 환경은 NumPy 1.26, thor 환경은 NumPy 2를 쓴다. core는 두 버전에서 모두 테스트한다(8절 CI).

### IROS와 NeurIPS의 기여 경계

- IROS(BOOST): LCB 가중치 그래프 가치 C_G와 GSR식 가중치로 손 expert 가중 SFT와 guidance subgoal을 만든다. 그래프 가치 계산은 두 원고가 공유한다.
- NeurIPS: 그래프 가치로 actor-critic의 TD critic을 대체하는 비용 도메인 증류, 근사최적성 이론, 벤치마크(E0–E5)가 기여다. LCB 그래프 가치는 BOOST를 인용해 선행 구성요소로 둔다.
LCB 그래프 가치가 두 원고에 모두 들어가므로, NeurIPS의 새로움은 critic 대체, 이론, 벤치마크에 있어야 한다. 투고 시점에 두 학회의 이중 투고 규정을 확인한다.

## 3. 디렉터리 구조

최상위는 core(graphcritic/), 데이터 어댑터(envs/), fork된 베이스라인(third_party/), 실험 래퍼(experiments/)로 나눈다.

```text
graph-critic/
├── README.md
├── pixi.toml / pixi.lock           # 환경: core, bench, (kitchen)
├── docs/
│   ├── spec.md                     # 이 문서의 레포 사본
│   └── math.md                     # 정식화: 노테이션, 명제, 정리
├── graphcritic/                    # core (numpy/scipy만, BOOST와 공유)
│   ├── types.py                    # TransitionSet, NodeSet, EdgeSet, GraphArtifact
│   ├── featurize.py                # Featurizer 프로토콜, 표준화
│   ├── nodes.py                    # 반경 병합, medoid 대표, 후속 노드 일관성 검사
│   ├── edges.py                    # chunk 전이 → 간선, run-length 압축, EdgeFilter
│   ├── stats.py                    # Beta 사후분포, LCB, Clopper–Pearson
│   ├── weights.py                  # WeightPolicy: hop | time | lcb
│   ├── solve.py                    # 역방향 다중출발 Dijkstra, 다중 결과 간선용 VI
│   ├── targets.py                  # critic target y, C_max 클리핑, 단위 변환
│   ├── sft_weights.py              # 비율 advantage 가중치 (BOOST 가중 SFT용)
│   ├── online.py                   # transition 추가, 부분 재계산
│   ├── io.py                       # GraphArtifact 저장/로드 (npz + json 메타)
│   └── diagnostics.py              # 단조성, 커버리지, 연결 성분, 순위 상관
├── envs/                           # 데이터 어댑터: 원본 → TransitionSet
│   ├── robomimic.py
│   ├── ogbench.py
│   ├── kitchen.py                  # 선택 (Minari)
│   ├── toy.py                      # 결정론적 사슬/격자, 정답 d* 계산 가능
│   └── boost_export.py             # BOOST가 내보낸 파일 읽기
├── third_party/
│   ├── fql/                        # submodule: 본인 fork, branch graph-critic
│   └── qc/                         # submodule: 본인 fork, branch graph-critic
├── experiments/
│   ├── build_graph.py              # 데이터셋 → GraphArtifact + targets.npz
│   ├── launch.py                   # fork의 main.py / main_online.py 호출 래퍼
│   ├── sweeps/                     # yaml: 과제 × 방법 × seed
│   └── analysis/                   # 표, 그림, 진단 플롯
├── configs/graph/<env>.yaml        # ψ, r, h, γ, 비관성 설정, C_max
├── tests/                          # core 단위 테스트, toy 정리 검증, BOOST 등식 테스트
└── outputs/                        # gitignored
```

BOOST 쪽 변경은 두 가지뿐이다. boost/graph/value.py를 core 호출로 바꾸고, scripts/export_transitions.py를 추가한다.

## 4. core 라이브러리 명세

core는 데이터셋을 받아 그래프 artifact와 critic target 배열을 만드는 순수 함수들의 묶음이다. 학습 코드는 이 출력만 읽는다.

### 4.1 타입

스키마 v0.1은 행 하나를 환경 step `(s_t, a_t, s_{t+1})`로 정의한다. 행 순서는 원본 데이터셋(특히 QC의 평탄 배열)과 같고, chunk 값은 저장하지 않고 target 계산 때 만든다. 이 행 기반 정의는 앞서 적힌 chunk 행 스키마를 대체한다.

| 타입 | 필드 (dtype, shape) | 의미 |
| --- | --- | --- |
| `TransitionSet` | `feat`, `next_feat`: float32 `(N,p)`; `episode_id`, `t`, `row_idx`: int64 `(N,)`; `success`, `terminal`: bool `(N,)` | 현재/다음 특징, 에피소드와 step 번호, 다음 상태의 성공 여부, 마지막 행 표시, 원본 행 번호 |
| `NodeSet` | `rep_feat`: float32 `(V,p)`; `rep_ref`: int64 `(V,2)`; `assign_cur`, `assign_next`: int64 `(N,)`; `is_goal`: bool `(V,)` | 대표 특징과 출처 `(row, role)` (`0=cur`, `1=next`; BOOST 같은 외부 빌더는 row `-1` 허용), 행별 현재/다음 상태 노드, 성공 노드 |
| `EdgeSet` | `src`, `dst`, `kappa`, `n_pos`, `n_neg`: int64 `(E,)`; `kind`: int8 `(E,)` | 방향 간선, 지속시간 상계(스텝), 성공/실패 occurrence 수, `0=data`, `1=bridge` (v0.1에서는 미사용) |
| `GraphArtifact` | `schema_version`, `config`, `feature_stats`, `nodes`, `edges`, `weights`, `cost`, `succ`, `meta`, `content_hash` | 그래프 및 계산 설정. `cost`는 nats 단위이며 도달 불가는 `inf`, `succ`는 goal/도달 불가에서 `-1` |
| `TargetSet` | `y`: float32 `(N,)`; `valid`: bool `(N,)`; `h`, `unit`, `graph_hash`, `dataset_id` | QC sampler와 같은 chunk 시작 행 기준 critic target, 유효성, 단위 및 출처 |

모든 스키마 생성 시 배열 dtype을 명시적으로 맞추고 `validate()`를 실행한다. 스키마 불일치는 `SchemaError`다. 같은 에피소드 행은 연속되고 `t`는 0부터 1씩 증가하며, 각 에피소드 마지막 행만 `terminal=True`다. 비종료 행 `i`에서는 `assign_next[i] == assign_cur[i+1]`다. 성공 상태는 성공 행의 `s_{t+1}`이며, 다음 행이 같은 에피소드에 있으면 그 행의 `s_t`도 같은 성공 상태다. goal 노드에는 성공 상태만 할당한다.

`EdgeSet`은 자기 간선을 허용하지 않으며 `(src,dst,kind)` 키가 유일해야 한다. `kappa >= 1`, occurrence 수는 음수가 아니다. Graph의 weights는 유한하고 양수, cost는 음수가 아니며 goal cost는 0이다. `GraphConfig`는 frozen dataclass이며 `gamma`, `radius`, feature 설정, 압축/일관성 설정, weight policy, pessimism 사전분포와 `c_max_steps`를 담는다. v0.1에서는 `consistency=True`를 지원하지 않는다. `TargetConfig`는 chunk 길이 `h`와 `steps`/`nats` 단위를 담는다.

### 4.2 API

- `build_nodes(trans, cfg) -> NodeSet`: 각 행의 current state와 각 에피소드 마지막 행의 next state를 군집화한다. 비종료 행의 다음 상태는 다음 행의 current 상태와 같은 상태로 취급한다.
- `build_edges(trans, nodes, cfg, edge_filter=None) -> EdgeSet`: episode 내 연속 상태 노드 전이로 간선을 만들며, 압축 시 occurrence duration의 최댓값을 `kappa`로 저장한다.
- `edge_weights(edges, cfg) -> weights`: `hop`, `time`, `lcb` 정책으로 양수 비용을 계산한다.
- `solve(nodes, edges, weights) -> (cost, succ)`: 역방향 다중 출발 Dijkstra로 goal까지 최소 비용과 후계자를 구한다.
- `critic_targets(trans, nodes, cost, gamma, c_max_steps, cfg) -> TargetSet`: QC의 chunk 유효성 규칙에 맞춰 시작 행별 target을 만든다.
- `sft_weights(trans, nodes, cost, targets, beta, w_max) -> weights`: BOOST의 cost-domain advantage 가중치를 만든다.
- `online.update(artifact, new_trans) -> artifact`: v0.1은 새 데이터로 전체 재계산한다.
### 4.3 구현하는 수식

시간 비용 상수는 c = −log γ다. 간선 성공확률 하한 ℓ_e는 Beta 사후분포의 하위 분위수이고, edge별 수준은 δ_E = δ / |E|다(|E|는 그래프 전체 간선 수). 구현은 scipy.special.betaincinv.

$$
\ell_e = F^{-1}_{\mathrm{Beta}(\alpha_0 + n_e^{+},\ \beta_0 + n_e^{-})}(\delta_E), \qquad w_e = -\log \ell_e + c\,\kappa_e
$$

offline 데이터만 있는 경우(β₀ = 1, 실패 0회)는 닫힌 꼴이다. `mode="union"`은 분위수 `q=δ/|E|`, `mode="fixed"`는 분위수 `q=exp(-λ)`를 쓴다.

$$
-\log \ell_e = \frac{\log(1/q)}{\alpha_0 + n_e^{+}}
$$

그래프 비용은 아래 방정식의 해이고, 가중치가 비음수이므로 Dijkstra가 정확한 해를 준다.

$$
C_G(v) = 0\ \ (v \in \mathcal V_{\mathcal G}), \qquad C_G(u) = \min_{e = (u \to v)} \bigl[\, w_e + C_G(v) \,\bigr]
$$

행 `i`에서 시작하는 `h`-step chunk가 유효한 조건은 마지막 행이 데이터 안에 있고, 모든 행이 한 에피소드에 있으며, 마지막 행 이전에는 terminal이 없는 것이다. QC `sample_sequence`와 같은 `valid[..., -1]` 규칙을 쓴다. chunk 안 첫 성공 위치를 `k_i`라 하면 성공 target은 `k_i+1` step이다. 성공이 없으면 마지막 다음 상태의 graph cost를 이어 붙인다. graph cost는 nats로 저장하고 target에서만 단위 변환한다. 도달 불가 비용은 `c_max_steps`로 제한한다.

$$
y_i\,[\mathrm{steps}] = \begin{cases} k_i+1, & k_i\text{ exists} \\ h + \min\bigl(C_G(\mathrm{assign\_next}[i+h-1])/c,\ c_{\max}^{\mathrm{steps}}\bigr), & \text{otherwise} \end{cases}
$$

무효 행은 `valid=False`, `y=0`으로 저장한다. 손실에서 valid mask를 곱해도 NaN이 전파되지 않도록 무효 target에 NaN을 넣지 않는다. `unit="nats"`이면 step target에 `c`를 곱한다. `h=1`은 FQL용 target이다.

BOOST 가중 SFT용 가중치는 비용 도메인 advantage의 지수다. 이는 (Q/V)^β와 같고, 거리에 대해 스케일 불변이다.

$$
w(s,a) = \mathrm{clip}\Bigl(\exp\bigl[\beta\,(C_G(\nu(s)) - y_{\mathrm{nats}})\bigr],\ 0,\ w_{\max}\Bigr)
$$

### 4.4 테스트 (core 태그 조건)

- [ ] BOOST 등식: policy=hop에서 cost가 BOOST의 기존 BFS 진행도 c(v)와 모든 노드에서 일치. lcb 설정에서는 BOOST와 실험 레포의 C_G가 같은 artifact에서 일치
- [ ] toy 사슬에서 C_G = c·d* 정확 일치, 벽이 있는 격자에서 우회 경로 비용 일치
- [ ] LCB 닫힌 꼴과 betaincinv 수치 일치
- [ ] 같은 입력과 seed에서 artifact 해시 동일 (결정론)
- [ ] NumPy 1.26과 2.x 양쪽에서 통과
## 5. 베이스라인 통합

두 공식 구현을 fork하고, critic 학습 신호만 바꾸는 플래그 하나를 추가한다. 나머지(하이퍼파라미터, seed, 평가)는 원본 그대로 둔다.

| 베이스라인 | 원본 | 프레임워크 | 이번 실험에서의 역할 |
| --- | --- | --- | --- |
| FQL | seohongpark/fql | JAX. jax ≥ 0.4.26, ogbench == 1.1.0, gymnasium == 0.29.1 | 단일 스텝 actor-critic. one-step 정책이 Q 최대화 + flow BC 증류 |
| QC | ColinQiyangLi/qc | JAX, main.py(offline) / main_online.py | chunk critic. best-of-N 추출은 argmax만 써서 비용 도메인 변환에 불변 |

fork 규칙: 각 fork에 graph-critic 브랜치를 만들고, submodule은 특정 commit에 고정한다. upstream에는 push하지 않는다.

### 5.1 patch 지점

1. 데이터셋: 각 transition에 graph_target 필드를 추가한다. QC는 chunk 시작 인덱스 기준으로 정렬한다. experiments/build_graph.py가 만든 targets.npz를 로드 시점에 붙인다.
2. 설정: agent.critic_mode ∈ {td, graph}, agent.graph_target_unit = steps를 추가한다.
3. critic 손실: graph 모드면 손실은 (C_θ(s,a) − y)²이고 target network는 쓰지 않는다. ensemble 집계는 비용 도메인에서 max(원본의 min Q에 해당)를 쓴다.
4. actor: Q := −C_θ로 바꾼다. FQL의 Q 정규화 옵션은 원본 동작 그대로 적용한다. QC best-of-N은 argmin C로 고른다.
5. online (main_online.py): K env step마다 새 transition을 graphcritic.online.update에 넣고, replay buffer의 graph_target을 갱신한다.

### 5.2 재현 gate

critic_mode=td로 원본 수치를 먼저 재현한다. 재현이 신뢰구간 안에 들어오기 전에는 graph 모드 실험을 시작하지 않는다(실험 E1).

### 5.3 확인할 것 (코드를 읽고 이 절에 기록)

- QC sampler가 에피소드 끝을 넘는 chunk를 어떻게 처리하는지. targets.py의 k 처리를 똑같이 맞춘다.
- 각 fork가 성공 시점에 에피소드를 종료(mask)하는지. 우리 target의 흡수 가정과 맞아야 한다.
- 확인 결과에 따라 decoupled Q-chunking(DQC)을 세 번째 베이스라인으로 추가할지 정한다.
## 6. 벤치마크와 데이터 어댑터

주 벤치는 robomimic MH와 OGBench singletask 조작 과제다. 둘 다 QC 공식 구현이 이미 지원하므로 베이스라인 수치와 바로 비교된다.

| 벤치 | 과제 | 데이터 | 보상 설정 | 상태 표현 ψ | 역할 |
| --- | --- | --- | --- | --- | --- |
| robomimic MH | lift, can, square | ~/.robomimic/<task>/mh/low_dim_v15.hdf5 (QC 경로) | 이진 성공 (QC 설정) | low-dim 관측 표준화: 물체 pose, eef, gripper | 주 벤치. 시연 품질 편차가 있어 그래프 가치가 의미를 가짐 |
| OGBench singletask | cube-double, cube-triple, scene, puzzle-3x3 | 공식 데이터셋 | scene, puzzle은 QC의 --sparse=True. cube는 성공만 남기는 래퍼 필요 여부 확인 | state 관측 표준화 | 장거리 stitching |
| Franka Kitchen (선택) | mixed, partial | Minari D4RL/kitchen | 4개 subtask 모두 완료 = 성공 | 59차원 관측 | 다단계. gymnasium 버전 충돌 시 제외 |
| toy | 결정론적 사슬, 벽 있는 격자 | 생성 | 성공만 | 정확한 상태 | 정리 검증. 정답 d*를 계산할 수 있음 |
| BOOST | 포장 | BOOST export | meta.success | BOOST의 s^m (그래프는 BOOST가 생성) | 실물 (E6) |

### 6.1 어댑터 계약

모든 어댑터는 TransitionSet, 성공 라벨, 에피소드 경계를 돌려준다. chunk 정렬은 fork sampler와 같은 규칙을 따른다(5.3).

### 6.2 보상 공정성

우리 방법은 성공 라벨만 쓰고, 베이스라인은 각 논문이 쓴 보상을 쓴다. 두 가지 표를 함께 보고한다.

- 표 A: 베이스라인은 원 논문 보상, 우리는 성공 라벨만. 우리 쪽이 정보가 적은 조건이다.
- 표 B: 모든 방법을 성공 보상만으로 통일한다.
### 6.3 BOOST export 형식

BOOST의 scripts/export_transitions.py가 아래를 쓴다. 실험 레포는 이 파일만 읽는다.

- transitions.npz: TransitionSet 필드. feat은 BOOST s^m을 벡터화한 값
- graph.npz: BOOST가 만든 NodeSet, EdgeSet (접촉 세그먼트 제약 적용 후)
- meta.json: 데이터셋 ID, BOOST commit, 설정 해시, hand_spec
실험 레포는 가중치 파일 weights.npz를 돌려준다. BOOST가 이것으로 손 expert를 학습한다.

## 7. 실험 프로토콜

실험은 E0부터 E6까지 일곱 개이고, E1(재현 gate)을 통과해야 E2 이후를 시작한다.

| ID | 내용 | 비교 조건 | 지표 | 데이터 |
| --- | --- | --- | --- | --- |
| E0 | 정리 검증 | 비용 도메인 vs 가치 도메인 target, 주입 오차 크기 | 거리 d별 순위 정확도, 진전 조건(오차 < h 스텝) 성립 여부, ‖∇ₐQ‖ 대 d | toy |
| E1 | 재현 gate | FQL, QC (td 모드) | 원 논문 수치와의 차이 | robomimic 3과제 + OGBench 2과제 |
| E2 | offline 본 실험 | FQL, FQL-G, QC, QC-G | 학습 종료 시 성공률 (평균 ± 표준오차) | robomimic, OGBench |
| E3 | offline → online | 같은 4개 | 환경 step 대비 성공률 곡선 | 같음 |
| E4 | 진단 | td vs graph critic | 성공 궤적에서 critic 값과 남은 스텝 수의 순위 상관, critic 발산 횟수, seed 간 분산, d별 ‖∇ₐQ‖ | E2 체크포인트 |
| E5 | ablation | 가중치 정책(hop / time / lcb), 가치 vs 비용 도메인 target, 비관성 λ, 반경 r, chunk 길이 h, ensemble 집계 | E2와 같음 | robomimic square, OGBench 1과제 |
| E6 | BOOST 실물 | 손 expert 가중 SFT 가중치: lcb 그래프(BOOST 현행), hop 그래프(ablation), TD critic, 균등 | teleop에 없던 물체 pose에서 포장 성공률 | BOOST 라운드 데이터 |

### 규칙

- offline step 수, seed 수, 평가 에피소드 수는 각 베이스라인 논문의 설정을 그대로 쓴다. E1에서 fork 코드의 기본값을 읽어 이 절에 기록한다.
- 모든 실행은 (레포 commit, fork commit, 데이터셋 ID, 그래프 artifact 해시, 설정 해시)를 기록한다.
- E6의 TD critic은 같은 transition에서 critic_mode=td로 학습한다. 손 expert 학습 자체는 BOOST 레포에서 한다.
- 계산량은 E1의 실행당 GPU 시간을 재서 E2–E5 전체 규모를 추정한다. 추정 전에는 sweep 크기를 확정하지 않는다.
## 8. 환경(pixi)과 git 규칙

환경은 BOOST와 같은 방식(pixi feature/environment, pixi.lock 커밋)으로 관리하고, git 규칙도 BOOST를 그대로 따른다.

### 8.1 pixi 환경

| 환경 | 내용 | 용도 |
| --- | --- | --- |
| core | numpy, scipy, pytest. NumPy 1.26과 2.x 두 matrix | core 테스트, BOOST 호환 확인 |
| bench | core + fork들의 requirements.txt 고정 버전 (jax, flax, ogbench == 1.1.0, gymnasium == 0.29.1, robomimic 계열) | E0–E5 |
| kitchen (선택) | minari, gymnasium-robotics | Kitchen. gymnasium 버전이 bench와 충돌하면 별도 환경으로 둔다 |

- bench의 버전은 QC fork의 requirements.txt를 기준으로 한다. 두 fork가 충돌하면 fork별 환경으로 나눈다.
- core는 jax와 torch에 의존하지 않는다. BOOST의 Thor 환경(CUDA 13, NumPy 2)에서도 그대로 설치돼야 한다.
### 8.2 git 규칙 (BOOST와 동일)

| 브랜치 / tag | 용도 |
| --- | --- |
| main | README와 문서만 |
| develop | 기본 작업 브랜치. core 테스트가 항상 통과하는 상태 유지 |
| feat/<name> | 단기 작업, develop으로 merge 후 삭제 |
| tag <feature>-v<maj>.<min> + snap/<feature>-v<maj>.<min> | 기능 완성 시점. snap 브랜치는 수정 금지 |

### 계획된 tag:

| tag | 완료 조건 |
| --- | --- |
| core-v0.1 | 4.4절 테스트 전부 통과. BOOST가 이 tag를 pin |
| repro-v0.1 | E1 재현 gate 통과 |
| offline-v0.1 | E2 완료 |
| o2o-v0.1 | E3 완료 |
| boost-exp-v0.1 | E6 가중치 전달과 결과 수집 |
| paper-v1.0 | 투고 시점 스냅샷 |

- 커밋 금지: 데이터, 체크포인트, 그래프 artifact. .gitignore에 data/, outputs/, checkpoints/.
- core의 공개 API를 바꾸면 minor 버전을 올린다. BOOST는 새 tag로 올릴 때만 반영한다.
## 9. 일정과 마일스톤

core-v0.1은 BOOST 그래프 단계가 시작하는 11/24 전에 끝내야 한다. BOOST가 이 tag를 pin해서 그래프 단계를 시작하기 때문이다.

E2는 IROS 마감(2027-03-01) 전에 백그라운드로 돌리고, 2월에는 새 개발 없이 예약된 sweep만 유지한다. NeurIPS 2027 마감일은 아직 확인하지 않았으므로 작성 구간의 끝(5/14)은 가정이다.

## 10. 위험과 미결정 사항

가장 큰 위험은 베이스라인 재현 실패와 데이터 구성(대부분 성공 궤적)이다. 둘 다 E1 단계에서 드러난다.

| 항목 | 위험 | 대응 | 결정 시점 |
| --- | --- | --- | --- |
| 베이스라인 재현 | td 모드 수치가 원 논문과 다르면 비교 전체가 무의미 | E1 gate. 통과 전 graph 실험 금지 | E1 |
| OGBench 보상 의미 | cube 과제의 기본 보상이 성공만이 아닐 수 있음 | fork 코드에서 --sparse 적용 범위 확인, 필요하면 성공 래퍼 추가 | E1 |
| chunk와 에피소드 경계 | QC sampler와 우리 target의 k 처리가 다르면 target이 어긋남 | 5.3절 확인 후 targets.py에 반영 | core-v0.1 이후 |
| 성공 궤적 위주 데이터 | 간선이 모두 성공으로 이어지면 가치 차이가 거리에서만 나옴 | robomimic MH의 품질 편차 활용. 실패 rollout을 추가하면 베이스라인에도 동일하게 제공 | E2 |
| 노드 병합 metric | 저차원 표준화 ψ가 도달 가능성과 어긋날 수 있음 | 반경 r ablation(E5), 병합 오류 진단 | E5 |
| 그래프 규모 | OGBench 데이터 수백만 transition에서 병합 비용 | KD-tree 기반 반경 병합. 100M 데이터는 범위 밖 | E2 |
| 두 원고의 기여 중복 | IROS와 NeurIPS의 그래프 가치 기여가 겹침 | LCB 그래프 가치는 BOOST를 인용해 선행 구성요소로 두고, NeurIPS 기여는 critic 대체, 이론, 벤치마크로 한정. 이중 투고 규정 확인 | BOOST 원고 작성 전 |
| 계산 예산 | 과제 × 방법 × seed 규모 미정 | E1에서 실행당 GPU 시간 측정 후 확정 | E1 |
| Kitchen 의존성 | gymnasium 버전 충돌 | 별도 환경 또는 제외 | E1 |
| NeurIPS 2027 마감일 | 아직 확인하지 않음 | 공식 발표 확인 후 9절 일정 조정 | 2027-01 |
