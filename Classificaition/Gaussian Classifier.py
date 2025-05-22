import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def train_gaussian_nb(data_matrix, n_samples, k, labels, test_size=0.2, random_state=42):
    """
    주어진 데이터 행렬을 사용하여 Gaussian Naïve Bayes 모델을 학습하고 평가하는 함수

    :param data_matrix: (n_samples x (k+1)) 크기의 데이터 행렬
    :param n_samples: 샘플 개수
    :param k: neighbour 개수
    :param labels: 라벨 리스트 (18개의 고유 라벨)
    :param test_size: 테스트 데이터 비율 (기본값 0.2)
    :param random_state: 랜덤 시드 (기본값 42)
    :return: 모델의 정확도
    """
    # 특징(X)과 라벨(y) 분리
    X = data_matrix[:, :-1].astype(float)  # 마지막 열을 제외한 특징 데이터 (숫자로 변환)
    y_raw = data_matrix[:, -1]  # 마지막 열이 라벨 (현재는 문자열 또는 객체)

    # 라벨 인코딩 (문자형 라벨을 숫자로 변환)
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    y = label_encoder.transform(y_raw)  # 기존에 문자열이었던 라벨을 숫자로 변환

    # 데이터 분할 (훈련: 80%, 테스트: 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Gaussian Naïve Bayes 모델 생성 및 학습
    model = GaussianNB()
    model.fit(X_train, y_train)

    # 예측 수행
    y_pred = model.predict(X_test)

    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# 예제 데이터 생성 및 함수 실행
n_samples = 100
k = 5
labels = [
    "Humor", "Family", "Romance", "Parody", "Crime", "Tragedy", "Suspense", "Supernatural",
    "Spiritual", "Angst", "Hurt-Comfort", "Horror", "Sci-Fi", "Fantasy", "Adventure",
    "Mystery", "Friendship", "Drama"
]

# 랜덤으로 18개 라벨 중 하나를 선택해 라벨링
data_matrix = np.random.rand(n_samples, k)  # (n_samples x k) 크기의 특징 데이터
random_labels = np.random.choice(labels, n_samples)  # 라벨 데이터
data_matrix = np.column_stack((data_matrix, random_labels))  # (n_samples x (k+1)) 크기로 조합

accuracy = train_gaussian_nb(data_matrix, n_samples, k, labels, test_size=0.2)
print(f"모델 정확도: {accuracy:.2f}")
