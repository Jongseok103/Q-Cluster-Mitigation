import numpy as np
from collections import Counter

class QClusterMitigator:
    """
    Hamming distance와 Qubit-wise Majority Vote(QMV)를 이용한
    Q-Cluster Error Mitigation 알고리즘 클래스
    """
    def __init__(self, k_clusters=2, max_iters=10, random_state=42):
        self.k = k_clusters
        self.max_iters = max_iters
        self.centroids = []
        self.clusters = {}
        np.random.seed(random_state)


    def _hamming_distance(self, s1, s2):
        """
        두 비트열 간의 Hamming distance 계산
        """
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))


    def _get_majority_centroid(self, cluster_samples):
        """
        Qubit-wise Majority Vote로 새로운 중심점 계산
        """
        if not cluster_samples: return ""
        n_qubits = len(cluster_samples[0]) # 비트열 길이
        new_centroid = [] # 새로운 중심점 계산
        for i in range(n_qubits):   # 각 비트 위치별로
            ith_bits = [s[i] for s in cluster_samples] # 해당 위치의 모든 비트 수집
            # 과반수 투표 (Majority Vote) 
            if ith_bits.count('1') > len(cluster_samples) / 2:
                new_centroid.append('1')
            else:
                new_centroid.append('0') # 과반수 투표 결과 반영
        return "".join(new_centroid)

    def fit(self, bit_strings):
        """
        주어진 비트열 데이터에 대해 클러스터링을 수행하여 Centroid를 탐색.
        """

        # 초기화: 랜덤하게 k개의 비트열을 초기 중심점으로 선택
        self.centroids = list(np.random.choice(bit_strings, self.k, replace=False))
        
        for _ in range(self.max_iters): # 최대 반복 횟수만큼 반복
            self.clusters = {j: [] for j in range(self.k)} # 클러스터 초기화
            
            # 할당 단계 (Assignment) 
            for s in bit_strings:
                dists = [self._hamming_distance(s, c) for c in self.centroids] # 각 중심점과의 거리 계산
                closest = np.argmin(dists) # 가장 가까운 중심점 인덱스 찾기
                self.clusters[closest].append(s) # 해당 클러스터에 할당
                
            # 업데이트 단계 (Update with QMV)
            new_centroids = []
            for j in range(self.k): # 각 클러스터별로
                if self.clusters[j]: # 클러스터에 샘플이 있으면
                    new_centroids.append(self._get_majority_centroid(self.clusters[j])) # QMV로 새로운 중심점 계산
                else:
                    new_centroids.append(self.centroids[j])
            
            if new_centroids == self.centroids: # 중심점이 더 이상 변하지 않으면 종료
                break
            self.centroids = new_centroids
            # 중심점 업데이트 완료
        return self

    def mitigate(self, bit_strings):
        """
        Noisy 비트열을 학습된 Centroid로 매핑하여 에러를 완화.
        """
        mitigated_bitstrings = []
        # 현재 저장된 클러스터 정보를 바탕으로 Hard Assignment 수행
        # (주의: fit 이후 바로 호출한다고 가정하거나, 새로운 데이터에 대해 predict 로직 추가 가능)
        # 여기서는 fit 수행 시 생성된 클러스터를 그대로 활용합니다.
        for idx, samples in self.clusters.items():
            mitigated_bitstrings.extend([self.centroids[idx]] * len(samples))
            
        return Counter(mitigated_bitstrings)
    
    def get_centroids(self):
        return self.centroids