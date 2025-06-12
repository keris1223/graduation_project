import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

df = pd.read_csv("d:/graduation_project/FaderatedLearningStudy/data/time동적epoch.csv")

# 라운드별 평균 loss 계산
avg_loss_epoch = df.groupby("round")["loss"].mean()

# 이동 평균 (윈도우 크기 = 5)
smoothed_loss = uniform_filter1d(avg_loss_epoch.values, size=12)

# 변화량 계산
delta = np.abs(np.diff(smoothed_loss))
threshold = 0.01  # 수렴 판단 기준 (변경 가능)

# 연속 5번 이상 변화량이 작으면 수렴으로 간주
converged_round = None
for i in range(len(delta) - 3):
    if np.all(delta[i:i+4] < threshold):
        converged_round = avg_loss_epoch.index[i + 4]
        break

# 그래프 시각화
plt.figure(figsize=(10, 5))
plt.plot(avg_loss_epoch.index, avg_loss_epoch.values, label="loss", alpha=0.4)
plt.plot(avg_loss_epoch.index, smoothed_loss, label="Smoothed Loss", linewidth=2)
if converged_round:
    plt.axvline(x=converged_round, color='red', linestyle='--', label=f"Converged at Round {converged_round}")
plt.title("Dynamic Epoch by time")
plt.xlabel("Round")
plt.ylabel("Average Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
