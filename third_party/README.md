# Third-party forks

FQL and QC are tracked as Git submodules on their `graph-critic` branches. The superproject pins the following commits:

| Fork | URL | Branch | Pinned commit |
| --- | --- | --- | --- |
| FQL | <https://github.com/yangjunwon1309/fql.git> | `graph-critic` | `e8cd16eb490332924dfa2492097219f181765933` |
| QC | <https://github.com/yangjunwon1309/qc.git> | `graph-critic` | `48283b4f662bd1c127ec9fa80647b49759f10653` |

The submodule working trees are not initialized in a fresh checkout. Initialize the pinned commits with:

```bash
git submodule update --init --recursive
```

Do not push experimental changes to upstream repositories. Work on the `graph-critic` branches of the configured forks and update the superproject gitlinks only when intentionally changing the pinned commits.
