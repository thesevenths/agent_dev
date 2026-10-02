#!/usr/bin/env bash
# 在 flink 服务器上拉起本地 embedding 服务（OpenAI 兼容），供 multi-agent 长期记忆检索调用。
#
#   bash start_embed_server.sh                 # 后台（setsid + 崩溃自动重启），日志 /root/embed_server.log
#   bash start_embed_server.sh --stop          # 按端口停掉（只杀 python embed_server_light.py）
#   bash start_embed_server.sh --fg            # 前台（调试用，Ctrl+C 停）
#   tail -f /root/embed_server.log
#   curl http://127.0.0.1:8100/health
#
# 关于「当前 session 断了，API 会不会跟着断」：
#   默认分支用 setsid + nohup，进程脱离「会话 + 进程组」，SSH 断开只发 SIGHUP、nohup 直接忽略，
#   所以服务会继续活着。真正会断的只有三种情况：
#     1) 你用了 --fg 前台跑（Ctrl+C / 关窗口即死）；
#     2) 有东西对整个进程组发 SIGTERM/SIGKILL（systemd KillMode=control-group、运维自杀脚本、OOM killer）；
#     3) 机器重启 —— 这个只能靠 systemd 开机自启（见 embed-server.service，可选）。
#   tmux 能再加一层「随时 attach 回去看实时日志」和「手滑 Ctrl+C 也不丢」，但不是必需的。
set -euo pipefail

MODEL_PATH="${EMBED_MODEL_PATH:-/root/model/Qwen3-Embedding-0.6B}"
PORT="${EMBED_PORT:-8100}"
HOST="${EMBED_HOST:-0.0.0.0}"
LOG="${EMBED_LOG:-/root/embed_server.log}"
VENV_CONDActivate="${EMBED_CONDA_ACTIVATE:-}"   # 例：source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
NO_SUPERVISE="${EMBED_NO_SUPERVISE:-0}"         # 设成 1 = 不起监督循环，纯单进程 nohup
MAX_RESTART="${EMBED_MAX_RESTART:-10}"          # 监督循环最多重启几次

HERE="$(cd "$(dirname "$0")" && pwd)"

# detach：彻底脱离「登录会话 + 进程组」。setsid 在 util-linux 里（CentOS 7 自带），
# 万一没有就退回纯 nohup —— nohup 忽略 SIGHUP 已能扛住 SSH 断开，只是少一层进程组隔离。
detach() {
  if command -v setsid >/dev/null 2>&1; then
    setsid nohup "$@" </dev/null >>"${LOG}" 2>&1 &
  else
    nohup "$@" </dev/null >>"${LOG}" 2>&1 &
  fi
}

# --- --stop：按端口反查 PID 精确杀，避免 killall python 误伤同机其他 python ---
if [[ "${1:-}" == "--stop" ]]; then
  pids=""
  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -ti tcp:"${PORT}" 2>/dev/null || true)"
  fi
  if [[ -z "${pids}" ]] && command -v ss >/dev/null 2>&1; then
    pids="$(ss -ltnpH "sport = :${PORT}" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u || true)"
  fi
  if [[ -z "${pids}" ]]; then
    echo "[start_embed_server] 端口 ${PORT} 上没有监听中的进程，无需停止。"
  else
    echo "[start_embed_server] kill pid(s): ${pids}"
    # 先 TERM 给 5s，还在就 KILL（模型进程收 TERM 基本都会自己退）
    kill -TERM ${pids} 2>/dev/null || true
    for _ in $(seq 1 5); do sleep 1; done
    if command -v lsof >/dev/null 2>&1; then
      left="$(lsof -ti tcp:"${PORT}" 2>/dev/null || true)"
    else
      left=""
    fi
    if [[ -n "${left}" ]]; then
      kill -KILL ${left} 2>/dev/null || true
      sleep 1
    fi
    echo "[start_embed_server] stopped."
  fi
  exit 0
fi

FG=0
if [[ "${1:-}" == "--fg" ]]; then FG=1; fi

# --- 端口占用预检：重复起两份会各复制一份 1.2G 权重，抢内存最容易炸 ---
port_busy() {
  if command -v ss >/dev/null 2>&1 && ss -ltnpH "sport = :${PORT}" 2>/dev/null | grep -q LISTEN; then
    return 0
  fi
  if command -v lsof >/dev/null 2>&1 && lsof -ti "tcp:${PORT}" >/dev/null 2>&1; then
    return 0
  fi
  return 1
}
if port_busy; then
  echo "[start_embed_server] port ${PORT} 已被占用（疑似已有实例在跑），先确认再决定是否继续："
  curl -s "http://127.0.0.1:${PORT}/health" || true
  echo
  echo "想停掉旧实例：bash $(basename "$0") --stop"
  echo "或手工：      lsof -ti tcp:${PORT} | xargs -r kill"
  exit 1
fi

if [[ -n "${VENV_CONDActivate}" ]]; then
  eval "${VENV_CONDActivate}"
fi
# CPU 线程数：多给几个线程能明显提速 0.6B 的 encode（默认取 4，留余量给同机其他进程）
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# 监督循环里子进程写日志用的路径（不能依赖外层变量展开，bash -c 是单引号）
export EMBED_CHILD_LOG="${LOG}"

cd "${HERE}"
echo "[start_embed_server] model=${MODEL_PATH} port=${PORT} log=${LOG}"
echo "[start_embed_server] 提示：本脚本后台模式已脱离会话，SSH 断开不会被带走；只有机器重启才会没。"

PY_ARGS=(embed_server_light.py --model-path "${MODEL_PATH}" --host "${HOST}" --port "${PORT}")

if [[ "${FG}" -eq 1 ]]; then
  python "${PY_ARGS[@]}"
  exit $?
fi

if [[ "${NO_SUPERVISE}" == "1" ]]; then
  # 纯后台单进程：脱离会话/进程组，nohup 忽略 SIGHUP
  detach python "${PY_ARGS[@]}"
  echo "[start_embed_server] pid=$! (无监督循环，EMBED_NO_SUPERVISE=1)  logs: ${LOG}"
else
  # 后台 + 监督循环：进程被意外杀掉（OOM / SIGKILL / system cleanup）3s 内自动拉起，
  # 连续重启超 MAX_RESTART 次才放弃，避免死循环刷日志。
  detach bash -c '
    i=0
    while true; do
      i=$((i+1))
      if [ "$i" -gt "${EMBED_MAX_RESTART}" ]; then
        echo "[supervisor] 累计重启 ${EMBED_MAX_RESTART} 次仍起不来，放弃" >>"${EMBED_CHILD_LOG}"
        break
      fi
      "${@}" >>"${EMBED_CHILD_LOG}" 2>&1
      rc=$?
      echo "[supervisor] python exit rc=${rc}（第 ${i} 次），3s 后重启" >>"${EMBED_CHILD_LOG}"
      sleep 3
    done
  ' _ python "${PY_ARGS[@]}"
  echo "[start_embed_server] supervisor pid=$! (崩溃自动重启)  logs: ${LOG}"
fi

# --- 等 health 起来，最多 120s（0.6B 冷加载几十秒，不然 Windows 侧会连上还没加载完的模型） ---
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[start_embed_server] ready:"; curl -s "http://127.0.0.1:${PORT}/health"; echo; exit 0
  fi
  sleep 1
done
echo "[start_embed_server] 120s 内未 ready，最近日志："; tail -n 30 "${LOG}"; exit 1
