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

스키마 v0.3은 행 하나를 환경 step `(s_t, a_t, s_{t+1})`로 정의한다. 행 순서는 QC의 평탄 배열과 같고, chunk 값은 저장하지 않고 target 계산 때 만든다. current/next 성공 라벨에 더해 mode와 행동을 저장하며, 시뮬레이터 reset 상태는 별도 `sim_state.npz`에 보관한다. v0.1/v0.2 아티팩트는 배열 계약이 달라 명확한 버전 오류로 거부한다.

| 타입 | 필드 (dtype, shape) | 의미 |
| --- | --- | --- |
| `TransitionSet` | `feat`, `next_feat`, 선택 `action`: float32; `episode_id`, `t`, `row_idx`, `mode`, `mode_next`: int64; `success_cur`, `success`, `terminal`: bool | 현재/다음 특징과 행동, sampler segment/time, 원본 행, 현재/다음 상태 mode 및 성공 라벨 |
| `SimStateSet` | `qpos`, `qvel`: float32 `(M,dim)`; 선택 `button_states`: int64 `(M,dim)` | `state_table` 순서에 정렬된 MuJoCo reset sidecar |
| `NodeSet` | `rep_feat`: float32 `(V,p)`; `rep_ref`: int64 `(V,2)`; `assign_cur`, `assign_next`: int64 `(N,)`; `is_goal`: bool `(V,)` | `graph_kind=state`에서는 모든 상태 발생을 보존하고 identity 배정. `node`에서는 v0.1 대표점/군집 배정 사용 |
| `EdgeSet` | `src`, `dst`, `kappa`, `n_pos`, `n_neg`, `anchor`: int64; `kind`: int8; `dist`: float32 | `kind=0` 시간 간선은 확실한 데이터 replay, `kind=1`은 명시적 bridge. bridge 거리와 빌려 쓴 행동의 anchor를 기록 |
| `GraphArtifact` | `schema_version`, `config`, `feature_stats`, `nodes`, `edges`, `weights`, `cost`, `succ`, `meta`, `content_hash` | 그래프 및 계산 설정. `cost`는 nats 단위이며 도달 불가는 `inf`, `succ`는 goal/도달 불가에서 `-1` |
| `TargetSet` | `y`: float32 `(N,)`; `valid`: bool `(N,)`; `source`: uint8 `(N,)`; `h`, `unit`, `graph_hash`, `dataset_id` | target 값과 유효성, 출처 코드 `0=chunk success`, `1=finite graph`, `2=cap/unreachable`, `3=invalid` |

모든 스키마 생성 시 배열 dtype을 명시적으로 맞추고 `validate()`를 실행한다. 스키마 불일치는 `SchemaError`다. `episode_id`는 원본 시연 ID가 아니라 QC sampler가 연속된 유효 chunk로 취급하는 segment ID다. 같은 segment 행은 연속되고 `t`는 0부터 1씩 증가하며, 각 segment 마지막 행만 `terminal=True`다. QC `dones`가 연속해서 참인 robomimic post-success 행은 각 done 행에서 segment를 나누되 모든 원본 행과 `row_idx`를 보존한다. 이때 `success_cur`를 별도 기록해 segment 경계 뒤의 이미 성공한 상태도 goal로 남긴다. OGBench는 QC loader가 만든 전이 순서를 그대로 쓰고, 원본 terminal 관측 행은 transition으로 포함하지 않는다. 비종료 행 `i`에서는 `assign_next[i] == assign_cur[i+1]` 및 `success[i] == success_cur[i+1]`이어야 한다. `success_cur`는 `s_t`, `success`는 `s_{t+1}`의 성공 여부이며, goal 노드는 current/next 어느 쪽의 성공 상태도 비성공 상태와 병합하지 않는다.

`EdgeSet`은 자기 간선을 허용하지 않으며 `(src,dst,kind)` 키가 유일해야 한다. 시간 간선은 `kappa=1`, `dist=0`, `anchor=-1`; bridge는 `kappa=L`이고 유효 anchor가 있어야 한다. `GraphConfig.graph_kind`의 기본은 `state`, v0.1 비교와 BOOST 호환에는 `node`를 쓴다. state graph의 시간 비용은 `c·kappa`이고 비관성은 bridge에만 적용된다. `TargetConfig`는 chunk 길이 `h`와 `steps`/`nats` 단위를 담는다.

### 4.2 API

- `StateFeaturizer.fit(trans, feature_cfg)`와 `state_table(trans) -> (X, ref, success, mode)`: identity/select/TDR 거리 입력을 준비하고 현재 상태 및 terminal next 상태를 나열한다.
- `build_state_graph(trans, cfg, phi) -> (NodeSet, EdgeSet, meta)`: occurrence별 상태 정점, 확실한 temporal 간선, 거리·mode·landing 조건을 만족한 명시적 bridge를 만든다.
- kNN `auto`는 낮은 차원에서 cKDTree를 쓰고, 고차원에서는 FAISS가 있으면 우선 사용한다. FAISS가 없고 JAX GPU가 있으면 정확한 GPU top-k를 쓰며 후보 거리를 float64로 다시 계산한다. 둘 다 없으면 cKDTree로 돌아간다.
- `build_nodes`와 `build_edges`: `graph_kind=node`에서 v0.1 군집 그래프와 비교에 사용한다.
- `contract_segments(...) -> SegmentGraph`: bridge 출발/착지, mode 변화, goal, episode 경계와 길이 제한 사이의 temporal chain을 정확히 압축한다.
- `edge_weights(edges, cfg, d_step, meta) -> weights`: state graph는 temporal 비용 `c·kappa`와 bridge-only prior/data-calibrated/sim-verified 비용을 계산한다. node graph는 legacy 정책을 쓴다.
- `solve(nodes, edges, weights) -> (cost, succ)`: 중복 간선을 최소 비용으로 접고 역방향 다중 출발 Dijkstra로 goal까지 비용과 후계자를 구한다.
- `build_artifact(trans, cfg, *, dataset_id, source, extra_meta=None) -> GraphArtifact`: 표준화 통계부터 노드, 간선, 비용, 해시, 메타데이터까지 구성하고 검증한다.
- `rebuild_from_external(trans, nodes, edges, cfg, *, dataset_id, source, extra_meta=None) -> GraphArtifact`: 외부에서 만든 BOOST 스타일 그래프에 가중치와 최단 경로 비용을 붙인다.
- `critic_targets(trans, artifact, cfg) -> TargetSet`: QC chunk 유효성에 맞는 시작 행별 target을 만들고 artifact의 해시와 dataset ID를 붙인다.
- `sft_weights(trans, artifact, targets, beta, w_max) -> weights`: 같은 비용 단위에서 BOOST의 cost-domain advantage 가중치를 만든다.
- `report(trans, artifact, targets=None) -> dict`: 도달성, bridge 사용·switch 수, stitching gain, mode/segment 통계, target provenance를 JSON 호환 값으로 반환한다.

`online.update(artifact, new_trans)`는 아직 구현되지 않았다. artifact에는 기존 전체 전이 집합이 저장되지 않으므로, 증분 갱신 계약을 정하기 전에는 정확한 전체 재계산을 할 수 없다.
### 4.3 구현하는 수식

시간 비용 상수는 `c = −log γ`다. v0.2 state graph에서 확실한 temporal replay는 성공확률 불확실성이 없다고 보고 `c·kappa`만 낸다. prior bridge는 거리 기반 prior를 더한다. data-calibrated bridge는 행동 일치 데이터의 거리 bin에서 성공확률 하한을 공유한다.

$$
w_e=\begin{cases}
c\,\kappa_e, & e\in E_T,\\
c\,\kappa_e+c\left(\Lambda_0+\Lambda_1(d_e/r)^2\right), & e\in E_B\text{ (prior)},\\
c\,\kappa_e-\log\ell_{b(e)}, & e\in E_B\text{ (data-calibrated)}.
\end{cases}
$$

For bin `b`, `ell_b` is the Beta posterior lower quantile at `delta / n_bins`, using that distance bin's action-matched landing successes and failures. Bins with too few samples warn; bridges over `lambda_cap_steps` are dropped. The lower-confidence statement applies to a distance-bin average under the stated representativeness assumption, not to each bridge individually. The legacy node graph still supports the earlier per-edge LCB formula:

$$
\ell_e = F^{-1}_{\mathrm{Beta}(\alpha_0 + n_e^{+},\ \beta_0 + n_e^{-})}(\delta_E), \qquad w_e = -\log \ell_e + c\,\kappa_e.
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

`TargetSet.source`는 `0=chunk success`, `1=finite graph cost`, `2=capped/unreachable`, `3=invalid`로 각 target의 출처를 기록한다.

BOOST 가중 SFT용 가중치는 비용 도메인 advantage의 지수다. 이는 (Q/V)^β와 같고, 거리에 대해 스케일 불변이다.

$$
w(s,a) = \mathrm{clip}\Bigl(\exp\bigl[\beta\,(C_G(\nu(s)) - y_{\mathrm{nats}})\bigr],\ 0,\ w_{\max}\Bigr)
$$

### 4.4 테스트 (core 태그 조건)

- [x] 무작위 DAG에서 hop 비용이 BFS hop 수와 일치
- [x] toy 사슬, 벽이 있는 격자 및 stitched 사슬의 비용 검증
- [x] LCB 닫힌 꼴, posterior 단조성, edge 가중치 및 중복 간선 검증
- [x] target의 행 루프 기준값과 QC terminal validity 비교, SFT 가중치 경계 검증
- [x] 같은 입력에서 artifact 해시와 toy CLI 출력이 결정론적
- [x] core 테스트를 NumPy 1.26과 2.x 환경에서 각각 통과

검증 명령은 두 환경의 Python을 직접 호출해 실행했다. 현재 셸에서는 `pixi` 실행 파일을 찾지 못했다.
## 5. 베이스라인 통합

두 공식 구현을 fork하고, critic 학습 신호만 바꾸는 플래그 하나를 추가한다. 나머지(하이퍼파라미터, seed, 평가)는 원본 그대로 둔다.

| 베이스라인 | 원본 | 프레임워크 | 이번 실험에서의 역할 |
| --- | --- | --- | --- |
| FQL | seohongpark/fql | JAX. jax ≥ 0.4.26, ogbench == 1.1.0, gymnasium == 0.29.1 | 단일 스텝 actor-critic. one-step 정책이 Q 최대화 + flow BC 증류 |
| QC | ColinQiyangLi/qc | JAX, main.py(offline) / main_online.py | chunk critic. best-of-N 추출은 argmax만 써서 비용 도메인 변환에 불변 |

fork 규칙: 각 fork에 graph-critic 브랜치를 만들고, submodule은 특정 commit에 고정한다. upstream에는 push하지 않는다.

### 5.1 fork 통합 상태

1. 데이터셋: QC runner는 `--graph_targets`에서 schema v0.3 targets.npz/JSON 쌍을 읽고 샘플러 행에 붙인다. `h`, 단위, dataset ID, 행 수, QC chunk 유효성 배열이 모두 일치해야 한다.
2. 설정: `agent.critic_mode ∈ {td, graph}`와 `agent.graph_target_unit=steps`를 추가했다.
3. critic 손실: graph 모드는 softplus 비용 예측과 정답 비용의 제곱 오차를 사용하고 TD target network를 사용하지 않는다. Actor의 비용 집계는 ensemble max다.
4. actor: graph 모드의 Q는 `-max(C_θ)`이며, best-of-N은 최소 비용 action chunk를 선택한다. TD 모드는 QC 원본 집계를 유지한다.
5. online (main_online.py): graph 모드는 아직 지원하지 않으며 `--online_steps=0`이 필요하다.

graph target runner는 schema v0.3 target 쌍을 읽어 dataset ID, horizon, 행 수, validity를 검증하고 row-aligned 배열을 학습 데이터에 붙인다. graph critic은 현재 `softplus` 비용 예측을 steps target에 회귀한다. 그래프와 target을 만드는 `experiments/build_graph.py` 및 core pipeline은 구현되어 있다.

### 5.2 재현 gate

critic_mode=td로 원본 수치를 먼저 재현한다. 재현이 신뢰구간 안에 들어오기 전에는 graph 모드 실험을 시작하지 않는다(실험 E1).

### 5.3 확인할 것 (코드를 읽고 이 절에 기록)

- QC sampler는 chunk 시작 row부터 연속 h개를 읽고, 마지막 endpoint의 `valid`가 참이면 loss에 포함한다. chunk 끝 row의 terminal은 허용하고 그보다 앞선 terminal은 무효다. `main.py`의 graph target validator와 `targets.py`는 이 규칙을 공유해야 한다.
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
