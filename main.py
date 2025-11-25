import numpy as np
from qiskit import transpile
from qiskit.circuit.library import real_amplitudes as RealAmplitudes
from qiskit_aer.primitives import SamplerV2

# 우리가 만든 모듈 임포트
from src.noise_models import build_custom_noise_model
from src.qcluster import QClusterMitigator
from src.visualization import plot_mitigation_result, plot_pca_clusters

def main():
    # ---------------------------------------------------------
    # 1. 설정 및 회로 생성
    # ---------------------------------------------------------
    n_qubits = 5
    n_shots = 2048
    
    ansatz = RealAmplitudes(num_qubits=n_qubits, reps=1, entanglement='linear')
    ansatz.measure_all()
    
    # 파라미터 고정
    np.random.seed(42)
    fixed_params = np.random.rand(ansatz.num_parameters) * 2 * np.pi
    qc = ansatz.assign_parameters(fixed_params)
    qc_transpiled = transpile(qc, optimization_level=0)

    # ---------------------------------------------------------
    # 2. 시뮬레이션 실행 (Ideal vs Noisy)
    # ---------------------------------------------------------
    # [Ideal]
    sampler_ideal = SamplerV2()
    result_ideal = sampler_ideal.run([qc_transpiled], shots=n_shots).result()
    counts_ideal = result_ideal[0].data.meas.get_counts()
    
    # [Noisy] - Readout Error를 강하게(10%) 설정
    noise_model = build_custom_noise_model(p_depol=0.0, p_readout=0.10)
    sampler_noisy = SamplerV2()
    sampler_noisy.options.simulator = {"noise_model": noise_model}
    
    result_noisy = sampler_noisy.run([qc_transpiled], shots=n_shots).result()
    bitstrings_noisy = result_noisy[0].data.meas.get_bitstrings()
    counts_noisy = result_noisy[0].data.meas.get_counts()
    
    print(f"Simulation Complete: {len(bitstrings_noisy)} shots collected.")

    # ---------------------------------------------------------
    # 3. Q-Cluster Mitigation 적용
    # ---------------------------------------------------------
    # 클러스터 개수(k)는 문제 상황에 따라 튜닝 필요 (여기선 임의로 6 설정)
    mitigator = QClusterMitigator(k_clusters=6, max_iters=10)
    
    # 학습 및 완화 수행
    mitigator.fit(bitstrings_noisy)
    counts_mitigated = mitigator.mitigate(bitstrings_noisy)
    
    print("Mitigation Complete.")

    # ---------------------------------------------------------
    # 4. 결과 시각화
    # ---------------------------------------------------------
    # (1) 히스토그램 비교
    plot_mitigation_result(counts_ideal, counts_noisy, counts_mitigated)
    
    # (2) PCA 클러스터링 분석
    centroids = mitigator.get_centroids()
    plot_pca_clusters(bitstrings_noisy, centroids, counts_noisy, n_qubits)

if __name__ == "__main__":
    main()