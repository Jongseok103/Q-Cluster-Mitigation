import numpy as np
from collections import Counter

# 결과를 구조적으로 반환하기 위한 클래스
class MitigationResult:
    def __init__(self, mitigated_counts, centroids, clusters):
        self.mitigated_counts = mitigated_counts
        self.centroids = centroids
        self.clusters = clusters

class QClusterMitigator:
    """
    Hamming distance와 Qubit-wise Majority Vote(QMV)를 이용한
    Q-Cluster Error Mitigation 알고리즘 클래스
    """
    def __init__(self, k_clusters=2, max_iters=10, random_state=42):
        self.k = k_clusters  # 내부 변수는 self.k로 저장
        self.max_iters = max_iters
        self.centroids = []
        self.clusters = {}
        np.random.seed(random_state)

    def _hamming_distance(self, s1, s2):
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    def _get_majority_centroid(self, cluster_samples):
        if not cluster_samples: return ""
        n_qubits = len(cluster_samples[0])
        new_centroid = []
        for i in range(n_qubits):
            ith_bits = [s[i] for s in cluster_samples]
            if ith_bits.count('1') > len(cluster_samples) / 2:
                new_centroid.append('1')
            else:
                new_centroid.append('0')
        return "".join(new_centroid)

    def fit(self, bit_strings):
        """클러스터링 수행"""
        # 데이터가 k보다 적을 경우 예외처리
        if len(bit_strings) < self.k:
             self.centroids = bit_strings
             return self

        self.centroids = list(np.random.choice(bit_strings, self.k, replace=False))
        
        for _ in range(self.max_iters):
            self.clusters = {j: [] for j in range(self.k)}
            
            for s in bit_strings:
                dists = [self._hamming_distance(s, c) for c in self.centroids]
                closest = np.argmin(dists)
                self.clusters[closest].append(s)
                
            new_centroids = []
            for j in range(self.k):
                if self.clusters[j]:
                    new_centroids.append(self._get_majority_centroid(self.clusters[j]))
                else:
                    new_centroids.append(self.centroids[j])
            
            if new_centroids == self.centroids:
                break
            self.centroids = new_centroids
        return self

    def mitigate(self, bit_strings):
        """Hard Assignment를 통한 에러 완화"""
        mitigated_bitstrings = []
        for idx, samples in self.clusters.items():
            mitigated_bitstrings.extend([self.centroids[idx]] * len(samples))
        return Counter(mitigated_bitstrings)
    
    # [추가됨] main 함수에서 호출하는 run 메서드 구현
    def run(self, bit_strings):
        """
        fit과 mitigate를 순차적으로 실행하고 결과를 반환하는 편의 메서드
        """
        self.fit(bit_strings)
        counts = self.mitigate(bit_strings)
        
        # 결과 객체 반환
        return MitigationResult(
            mitigated_counts=counts, 
            centroids=self.centroids, 
            clusters=self.clusters
        )