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

def bitstring_to_vector(bitstring_list):
    """비트열 리스트를 numpy 수치 배열로 변환 ('01' -> [0, 1])"""
    if not bitstring_list:
        return np.array([])
    return np.array([[int(bit) for bit in s] for s in bitstring_list])

def plot_3x1_histogram(ideal, noisy, mitigated, title=""):
    """
    Ideal vs Noisy vs Mitigated 비교 히스토그램
    (00...0 부터 11...1 까지 모든 Basis를 표시)
    """
    # 1. 큐비트 수(n) 추론
    sample_key = next(iter(ideal)) if ideal else (next(iter(noisy)) if noisy else None)
    if sample_key is None:
        # 데이터가 아예 없는 경우 처리
        print("Warning: No data to plot histogram.")
        return
    
    n_qubits = len(sample_key)
    
    # 2. 모든 가능한 Basis 상태 생성 (00000 ~ 11111)
    full_basis = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]
    
    # 3. 각 딕셔너리에서 값 추출
    vals_ideal = [ideal.get(k, 0) for k in full_basis]
    vals_noisy = [noisy.get(k, 0) for k in full_basis]
    vals_mitigated = [mitigated.get(k, 0) for k in full_basis]
    
    x = np.arange(len(full_basis))
    width = 0.6
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    configs = [
        (axes[0], vals_ideal, '#1f77b4', "1. Ideal Distribution"),
        (axes[1], vals_noisy, '#ff7f0e', "2. Noisy Distribution"),
        (axes[2], vals_mitigated, '#2ca02c', "3. Q-Cluster Mitigated")
    ]
    
    for ax, vals, color, sub_title in configs:
        ax.bar(x, vals, width, color=color, alpha=0.9)
        ax.set_title(sub_title, fontsize=14, fontweight='bold')
        ax.set_ylabel("Counts")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        max_val = max(vals) if vals else 0
        if max_val > 0:
            ax.set_ylim(0, max_val * 1.2)

    axes[2].set_xlabel(f"All Basis States ({len(full_basis)} states)", fontsize=12)
    plt.xticks(x, full_basis, rotation=90, ha='center', fontsize=8)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_pca_clusters(bitstrings, counts, centroids, clusters, title=""):
    """
    비트열 분포와 클러스터링 결과를 2D PCA로 시각화
    
    Args:
        bitstrings (list): Noisy shot들의 비트열 리스트
        counts (dict): Noisy shot들의 빈도수 딕셔너리
        centroids (list): Q-Cluster 결과인 중심점 비트열 리스트
        clusters (dict): 클러스터링 결과 {cluster_idx: [bitstrings...]}
        title (str): 그래프 제목
    """
    # 데이터 벡터화
    X_noisy = bitstring_to_vector(bitstrings)
    if len(X_noisy) == 0:
        print("No noisy data to plot.")
        return

    # PCA 수행
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_noisy)
    
    # Centroid 좌표 변환
    X_centroids = bitstring_to_vector(centroids)
    C_pca = pca.transform(X_centroids)
    
    # 각 샷(Shot)에 대한 클러스터 라벨 생성 (색상 구분용)
    # bitstrings 리스트 순서대로 해당 비트열이 어느 클러스터에 속하는지 매핑
    
    # 1. 비트열 -> 클러스터 인덱스 매핑 테이블 생성
    # (주의: 동일한 비트열이라도 거리에 따라 다른 클러스터일 수 있으나, 
    # Q-Cluster는 Hard Assignment를 하므로 가장 가까운 Centroid로 매핑된다고 가정)
    
    # Q-Cluster 로직 재현: 각 비트열을 가장 가까운 Centroid에 할당
    def hamming_distance(s1, s2):
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))
        
    shot_labels = []
    for s in bitstrings:
        dists = [hamming_distance(s, c) for c in centroids]
        shot_labels.append(np.argmin(dists))

    plt.figure(figsize=(12, 8))
    
    # Jitter 추가 (이산적인 비트열 데이터 겹침 방지)
    jitter = np.random.normal(0, 0.05, size=X_pca.shape)
    
    # Scatter Plot (Noisy Shots)
    plt.scatter(X_pca[:, 0] + jitter[:, 0], 
                X_pca[:, 1] + jitter[:, 1], 
                c=shot_labels, cmap='viridis', alpha=0.3, s=30, label='Noisy Shots')

    # Centroids Plot
    plt.scatter(C_pca[:, 0], C_pca[:, 1], c='red', marker='X', s=300, 
                edgecolors='black', linewidth=2, label='Centroids')

    # 주요 비트열 텍스트 라벨링 (전체 샷의 1% 이상 빈도)
    unique_bitstrings = list(set(bitstrings))
    unique_coords = pca.transform(bitstring_to_vector(unique_bitstrings))
    
    for i, txt in enumerate(unique_bitstrings):
        count = counts.get(txt, 0)
        if count > len(bitstrings) * 0.01: 
            plt.annotate(f"{txt}\n({count})", 
                         (unique_coords[i, 0], unique_coords[i, 1]),
                         xytext=(0, 10), textcoords='offset points',
                         ha='center', fontsize=9, weight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.title(title, fontsize=15)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()