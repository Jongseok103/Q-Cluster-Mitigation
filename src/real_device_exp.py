import numpy as np
from qiskit import transpile

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as RuntimeSampler
from qiskit_aer.primitives import SamplerV2 as AerSampler

from src.qcluster import QClusterMitigator
from src.visualization import plot_3x1_histogram, plot_pca_clusters
from src.circuits import get_circuit

def run_real_device_experiment(api_token: str, 
                               circuit_type: str = 'ghz', 
                               n_qubits: int = 4, 
                               k_clusters: int = 2, 
                               shots: int = 2048):
    """
    IBM Quantum 실제 하드웨어에서 실험을 수행하고 Q-Cluster 완화를 적용합니다.
    """
    print(f"\n🚀 Real Device Experiment: {circuit_type.upper()} (n={n_qubits}, k={k_clusters})")

    # 1. IBM Quantum 서비스 연결 및 백엔드 선택
    try:
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_token)
    except Exception as e:
        print(f"❌ Error initializing service: {e}")
        return

    # Backend 선택: 최소 큐비트 수를 만족하는 가장 바쁜 백엔드 선택
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=n_qubits)
    
    # 백엔드 상태 출력
    status_msg = backend.status().status_msg
    print(f"   Target Backend: {backend.name} (Status: {status_msg})")

    # 2. 회로 생성 및 변환 (Transpilation)
    qc = get_circuit(circuit_type, n_qubits)
    
    # 실제 하드웨어의 연결성(Topology)에 맞춰 회로 변환
    qc_transpiled = transpile(qc, backend=backend, optimization_level=1)
    
    # 비교를 위한 Ideal 회로 (로컬 시뮬레이터용)
    qc_ideal = transpile(qc, optimization_level=0)

    # 3. [Ideal] 로컬 시뮬레이션 실행 (정답지 생성)
    print("   Running Ideal Simulation (Local)...")
    sampler_ideal = AerSampler()
    result_ideal = sampler_ideal.run([qc_ideal], shots=shots).result()
    counts_ideal = result_ideal[0].data.meas.get_counts()

    # 4. [Noisy] 실제 하드웨어 실행
    print(f"   Submitting Job to {backend.name}...")
    sampler_real = RuntimeSampler(mode=backend)
    job = sampler_real.run([qc_transpiled], shots=shots)
    
    print(f"   Job ID: {job.job_id()}")
    print("   Waiting for result... (This may take a while depending on the queue)")
    
    try:
        # 결과를 기다림 (Blocking)
        result_real = job.result()
    except Exception as e:
        print(f"❌ Job execution failed: {e}")
        return

    # 데이터 추출
    pub_result = result_real[0]
    bitstrings_noisy = pub_result.data.meas.get_bitstrings()
    counts_noisy = pub_result.data.meas.get_counts()
    print(f"   ✅ Job Finished! Collected {len(bitstrings_noisy)} shots.")

    # 5. Q-Cluster 완화 적용 (모듈화된 클래스 사용)
    print("   Applying Q-Cluster Mitigation...")
    
    # QClusterMitigator 인스턴스 생성 및 실행
    mitigator = QClusterMitigator(k_clusters=k_clusters)
    mitigation_result = mitigator.run(bitstrings_noisy)
    
    counts_mitigated = mitigation_result.mitigated_counts
    centroids = mitigation_result.centroids
    clusters = mitigation_result.clusters

    # 6. 결과 시각화
    print("   Generating Plots...")
    
    # (1) 3단 히스토그램 (Ideal vs Real vs Mitigated)
    plot_3x1_histogram(
        counts_ideal, 
        counts_noisy, 
        counts_mitigated, 
        title=f"Real Device ({backend.name}): {circuit_type.upper()}"
    )
    
    # (2) 2D PCA 클러스터링 시각화
    plot_pca_clusters(
        bitstrings_noisy, 
        counts_noisy, 
        centroids, 
        clusters,
        title=f"PCA Clustering Analysis ({backend.name})"
    )
