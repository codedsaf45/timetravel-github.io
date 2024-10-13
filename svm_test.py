import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# 데이터 포인트 설정 (선형 분리가 불가능한 1차원 데이터)
x = np.array([-3, -2, -1, 1, 2, 3, 4, 5, 6])
y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])

# 데이터 형태를 2차원으로 변환 (SVM은 2차원 이상의 데이터를 필요로 함)
X = x[:, np.newaxis]

# RBF 커널을 사용한 SVM 분류기 생성
clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale')

# 모델 학습
clf.fit(X, y)

# 새로운 데이터를 생성하여 예측 수행
X_test = np.linspace(-5, 7, 400).reshape(-1, 1)
y_pred = clf.predict(X_test)

# 그래프 시각화
plt.scatter(x, y, c=y, s=50, cmap='bwr', edgecolors='k')
plt.plot(X_test, y_pred, color='black', lw=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('SVM with RBF Kernel for Non-linear Classification')
plt.show()