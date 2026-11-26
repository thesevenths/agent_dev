import os

# 固定 Julia 环境（避免重新解析）
os.environ["PYTHON_JULIACALL_EXE"] = r"D:\ProgramData\anaconda3\julia_env\pyjuliapkg\install\bin\julia.exe"
os.environ["PYTHON_JULIACALL_PROJECT"] = r"D:\ProgramData\anaconda3\julia_env"

# 关闭多线程，避免 Windows+JuliaCall 的线程调度问题
os.environ["PYTHON_JULIACALL_THREADS"] = "1"
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

import juliacall      # 一定要在 torch 前
import numpy as np
from pysr import PySRRegressor

X = np.random.randn(200, 4)
y = X[:,0] + X[:,1]

print("Start fit...")

model = PySRRegressor(
    niterations=3,
    parallelism="serial",
    deterministic=True,
    random_state=42,
    progress=True,
)

model.fit(X, y)

print(model)