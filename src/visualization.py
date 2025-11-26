import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from qiskit.visualization import plot_histogram

def plot_mitigation_result(counts_ideal, counts_noisy, counts_mitigated):
    """
    Ideal, Noisy, Mitigated 결과를 히스토그램으로 비교 시각화.
    """
    legend = ['Ideal (No Noise)', 'Noisy', 'Q-Cluster Mitigated']
    fig = plot_histogram([counts_ideal, counts_noisy, counts_mitigated], 
                   legend=legend, 
                   figsize=(12, 6),
                   title="Q-Cluster Mitigation Performance")
    plt.show()
    return fig

def plot_pca_clusters(bitstrings, centroids, counts, n_qubits):
    """
    비트열 데이터를 PCA로 차원 축소하여 클러스터링 결과를 시각화.
    """
    # 데이터 전처리
    def bitstring_to_vector(bitstring_list):
        return np.array([[int(bit) for bit in s] for s in bitstring_list])

    X_noisy = bitstring_to_vector(bitstrings)
    
    # PCA 수행
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_noisy)
    
    # Centroid 변환
    X_centroids = bitstring_to_vector(centroids)
    C_pca = pca.transform(X_centroids)
    
    # 각 샷에 대한 클러스터 라벨링 (거리 기반)
    # 시각화 함수 내부에서 간단히 재계산
    shot_labels = []
    for s_vec in X_noisy:
        # Hamming distance 대신 Euclidean 사용 (PCA 공간 매핑을 위해 단순화) 또는
        # 원래 bitstring 기반으로 매칭. 여기서는 bitstring index로 매칭
        # 편의상 시각화에서는 단순히 산점도만 그립니다.
        pass 

    plt.figure(figsize=(12, 8))
    
    # Jitter 추가
    jitter_strength = 0.05
    jitter_x = np.random.normal(0, jitter_strength, size=len(X_pca))
    jitter_y = np.random.normal(0, jitter_strength, size=len(X_pca))

    plt.scatter(X_pca[:, 0] + jitter_x, X_pca[:, 1] + jitter_y, 
                alpha=0.3, s=30, label='Noisy Shots (Jittered)')

    plt.scatter(C_pca[:, 0], C_pca[:, 1], 
                c='red', marker='X', s=300, edgecolors='black', linewidth=2,
                label='Centroids')

    # 주요 비트열 라벨링
    unique_bitstrings = list(set(bitstrings))
    unique_coords = pca.transform(bitstring_to_vector(unique_bitstrings))
    
    for i, txt in enumerate(unique_bitstrings):
        count = counts.get(txt, 0)
        if count > 50: # Threshold
            plt.annotate(f"{txt}\n({count})", 
                         (unique_coords[i, 0], unique_coords[i, 1]),
                         xytext=(0, 10), textcoords='offset points',
                         ha='center', fontsize=9, weight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.title(f"2D PCA Visualization of {n_qubits}-Qubit Q-Cluster", fontsize=15)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()