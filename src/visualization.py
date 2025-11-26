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
    # 1. 큐비트 수(n) 추론 (키의 길이를 통해 확인)
    # 데이터가 비어있을 경우를 대비해 ideal, noisy 등에서 확인
    sample_key = next(iter(ideal)) if ideal else (next(iter(noisy)) if noisy else None)
    if sample_key is None:
        raise ValueError("Data is empty. Cannot determine n_qubits.")
    
    n_qubits = len(sample_key)
    
    # 2. 모든 가능한 Basis 상태 생성 (00000 ~ 11111)
    full_basis = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]
    
    # 3. 각 딕셔너리에서 값 추출 (없으면 0)
    vals_ideal = [ideal.get(k, 0) for k in full_basis]
    vals_noisy = [noisy.get(k, 0) for k in full_basis]
    vals_mitigated = [mitigated.get(k, 0) for k in full_basis]
    
    x = np.arange(len(full_basis))
    width = 0.6
    
    # 그래프 크기 조정 (Basis가 많아지므로 가로를 더 넓게)
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    # Plotting Configs
    configs = [
        (axes[0], vals_ideal, '#1f77b4', "1. Ideal Distribution"),
        (axes[1], vals_noisy, '#ff7f0e', "2. Noisy Distribution (Readout Error)"),
        (axes[2], vals_mitigated, '#2ca02c', "3. Q-Cluster Mitigated")
    ]
    
    for ax, vals, color, sub_title in configs:
        ax.bar(x, vals, width, color=color, alpha=0.9)
        ax.set_title(sub_title, fontsize=14, fontweight='bold')
        ax.set_ylabel("Counts")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # y축 한계 설정 (최대값보다 조금 높게 여유 두기)
        max_val = max(vals) if vals else 0
        if max_val > 0:
            ax.set_ylim(0, max_val * 1.2)

    # X축 설정 (모든 Basis 표시)
    axes[2].set_xlabel(f"All Basis States ({len(full_basis)} states)", fontsize=12)
    plt.xticks(x, full_basis, rotation=90, ha='center', fontsize=8) # 90도 회전하여 겹침 방지
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_2d_pca(bitstrings_noisy, counts_noisy, centroids, clusters, title=""):
    """비트열 분포와 클러스터링 결과를 2D PCA로 시각화"""
    # 데이터 벡터화
    X_noisy = bitstring_to_vector(bitstrings_noisy)
    if len(X_noisy) == 0:
        print("No noisy data to plot.")
        return

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_noisy)
    
    # 샷별 클러스터 라벨링 재계산 (centroid 기준)
    def hamming_distance(s1, s2):
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))
        
    shot_labels = []
    for s in bitstrings_noisy:
        dists = [hamming_distance(s, c) for c in centroids]
        shot_labels.append(np.argmin(dists))

    # Centroid 좌표 변환
    X_centroids = bitstring_to_vector(centroids)
    C_pca = pca.transform(X_centroids)

    plt.figure(figsize=(10, 8))
    
    # Jitter 추가 (점 겹침 방지)
    jitter = np.random.normal(0, 0.05, size=X_pca.shape)
    
    # Scatter Plot
    plt.scatter(X_pca[:, 0] + jitter[:, 0], 
                X_pca[:, 1] + jitter[:, 1], 
                c=shot_labels, cmap='viridis', alpha=0.3, s=30, label='Noisy Shots')

    # Centroids Plot
    plt.scatter(C_pca[:, 0], C_pca[:, 1], c='red', marker='X', s=300, 
                edgecolors='black', linewidth=2, label='Centroids')

    # 주요 비트열 텍스트 라벨링 (빈도수 높은 것만)
    unique_bitstrings = list(set(bitstrings_noisy))
    unique_coords = pca.transform(bitstring_to_vector(unique_bitstrings))
    
    for i, txt in enumerate(unique_bitstrings):
        count = counts_noisy.get(txt, 0)
        # 전체 샷의 1% 이상인 경우에만 라벨 표시 (기준 완화)
        if count > len(bitstrings_noisy) * 0.01: 
            plt.annotate(f"{txt}\n({count})", 
                         (unique_coords[i, 0], unique_coords[i, 1]),
                         xytext=(0, 10), textcoords='offset points',
                         ha='center', fontsize=8, weight='bold',
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    plt.title(f"2D PCA Visualization: {title}", fontsize=15)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()