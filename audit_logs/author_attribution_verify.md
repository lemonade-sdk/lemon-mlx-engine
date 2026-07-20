# Author attribution verification (read-only)

**Date:** 2026-07-19  
**Targets:** lemonade-sdk/lemonade#2751, lemonade-sdk/lemon-mlx-engine#62  
**Method:** Parent-session `gh api repos/.../pulls/{n}/commits` after filter-branch force-push; local remote-tracking tips.

## Verdict: PASS

GitHub **no longer** attributes these PR commits to user **ANTmi**.  
All checked commit `author.login` values are **antmikinka**, with email **antmikinka@gmail.com**.

## Commits API evidence (post-rewrite)

From session terminal log after force-push (`call-7ce2720c-…-190.log`):

| SHA (short) | GitHub `author` | commit author | email |
|-------------|-----------------|---------------|--------|
| fef445e | antmikinka | antmikinka | antmikinka@gmail.com |
| c13e3d9 | antmikinka | antmikinka | antmikinka@gmail.com |
| d5ae122 | antmikinka | antmikinka | antmikinka@gmail.com |
| 3842434 | antmikinka | antmikinka | antmikinka@gmail.com |
| 66c0353 | antmikinka | antmikinka | antmikinka@gmail.com |
| 52b061b | antmikinka | antmikinka | antmikinka@gmail.com |
| dbc5740 | antmikinka | antmikinka | antmikinka@gmail.com |

**ANTmi count: 0**

## Local remote tips (match rewritten heads)

| PR | Branch | Tip |
|----|--------|-----|
| lemonade #2751 | `origin/fix/mlx-stream-tool-hygiene` | `38424347c814bd3b34867b39e28a55277f9948de` |
| engine #62 | `origin/feat/openai-tools-server` | `dbc574015c1ddf880da3cfbed9cc66a54bea6326` |

Local rewrite log (filter-branch) rewrote all prior `antmi/*` commits to `antmikinka <antmikinka@gmail.com>` on both branches before force-push.

## Root cause (brief)

1. **OS user** on this machine is `antmi` (uid 1000).
2. Early agent/commits ran **without** a proper `user.name` / `user.email`, so git used defaults:
   - `antmi <antmi@local>`
   - `antmi <antmi@users.noreply.github.com>`
3. GitHub linked `antmi@users.noreply.github.com` / name `antmi` to **unrelated** account [ANTmi](https://github.com/ANTmi) (id 6091099), not [antmikinka](https://github.com/antmikinka) (id 67480807).
4. **Fix applied:** `git filter-branch --env-filter` rewrote author/committer to `antmikinka` / `antmikinka@gmail.com`, then force-pushed draft branches. Repo-local config now sets both identities correctly.

## Before → after (sample)

**Before (local log, pre-rewrite):**
- lemonade: `antmi@local`, `antmi@users.noreply.github.com` on P3/P4/backend commits
- engine: `antmi@local` on Qwen XML tools fix

**After:** all PR commits `antmikinka <antmikinka@gmail.com>` only.

## Note on PR author vs commit author

PR *openers* were already `antmikinka`. The issue was **commit-level** co-author / contribution graph attribution to ANTmi via bad emails — fixed by history rewrite + force-push.
