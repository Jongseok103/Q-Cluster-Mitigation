import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes as RealAmplitudes
from qiskit.circuit import ParameterVector


def get_ghz_circuit(n_qubits=4):
    qc = QuantumCircuit(n_qubits)
    
    # 1. 첫 번째 큐비트에 중첩 생성 (Hadamard)
    qc.h(0)
    
    # 2. 나머지 큐비트들에 얽힘 생성 (CNOT: 0->1, 1->2, ...)
    for i in range(n_qubits - 1):
        qc.cx(i, i+1)
        
    # 측정 추가
    qc.measure_all()
    return qc


def get_sparse_block_ansatz(n_qubits=4):
    """Sparse Block Ansatz: 4개의 상태만 생성"""
    qc = QuantumCircuit(n_qubits)
    theta = ParameterVector('θ', n_qubits // 2)
    for i in range(0, n_qubits - 1, 2):
        param_idx = i // 2
        qc.ry(theta[param_idx], i)
        qc.cx(i, i+1)
    qc.measure_all()
    params = [1.5, 0.5] if (n_qubits // 2) >= 2 else [1.5] * (n_qubits//2)
    return qc.assign_parameters(params[:len(theta)])

def get_dense_ansatz(n_qubits=4):
    """RealAmplitudes: Dense Distribution"""
    ansatz = RealAmplitudes(num_qubits=n_qubits, reps=1, entanglement='full')
    ansatz.measure_all()
    np.random.seed(42)
    random_params = np.random.uniform(0, 2*np.pi, ansatz.num_parameters)
    return ansatz.assign_parameters(random_params)

# --- 🆕 새로 추가된 회로들 ---

def get_bv_circuit(n_qubits=4):
    """
    1. Bernstein-Vazirani (BV)
    - 비밀 비트열(Secret String) 's'를 찾는 회로
    - 정답: '11...1' (모두 1인 경우로 설정함) -> 단 하나의 정답
    """
    qc = QuantumCircuit(n_qubits)
    
    # 모든 큐비트 중첩
    qc.h(range(n_qubits))
    
    # Oracle: '11...1'이 정답인 경우 (모든 큐비트에 Z 게이트 적용 효과)
    # 실제 구현: 상태 |x>에 대해 (-1)^{s*x} 위상 킥백
    # 간단히 s='11...1'이라 가정하면 모든 큐비트에 Z를 건 것과 결과가 같음 (HZH = X)
    # 여기서는 결과적으로 측정시 '11...1'이 나오도록 설계
    for i in range(n_qubits):
        qc.z(i) # Phase Oracle for s='11...1'
        
    # 다시 하다마드
    qc.h(range(n_qubits))
    
    qc.measure_all()
    return qc

def get_w_state_circuit(n_qubits=4):
    """
    2. W-State
    - |100..0> + |010..0> + ... + |000..1>
    - 정답 개수: n_qubits 개 (해밍 무게가 1인 상태들)
    - 논문 벤치마크에 포함된 회로
    """
    qc = QuantumCircuit(n_qubits)
    
    # W-state 생성 로직 (F. Vatan and C. Williams)
    # 루트 상태 생성
    qc.ry(2 * np.arccos(1 / np.sqrt(n_qubits)), 0)
    
    # 제어형 회전 및 CNOT 사다리
    for i in range(n_qubits - 1):
        # 제어 큐비트(i)가 0일 때 타겟(i+1) 회전
        theta = 2 * np.arccos(1 / np.sqrt(n_qubits - (i + 1)))
        # Cry 구현 (Control-0)
        qc.x(i)
        qc.cry(theta, i, i+1)
        qc.x(i)
        
        qc.cx(i+1, i)
        
    qc.x(0) # 마지막 보정
    
    qc.measure_all()
    return qc

def get_simple_qaoa(n_qubits=4):
    """
    3. Simple QAOA (MaxCut on Linear Graph)
    - 선형 그래프 0-1-2-3... 의 MaxCut 문제
    - 정답: 인접한 비트가 서로 다른 상태 (0101..., 1010...)
    """
    qc = QuantumCircuit(n_qubits)
    
    # 초기 상태: |+>
    qc.h(range(n_qubits))
    
    # 파라미터 (임의 설정 for p=1)
    gamma = 1.2
    beta = 0.8
    
    # Cost Layer (ZZ interaction)
    for i in range(n_qubits - 1):
        qc.cx(i, i+1)
        qc.rz(2 * gamma, i+1)
        qc.cx(i, i+1)
        
    # Mixer Layer (Rx)
    for i in range(n_qubits):
        qc.rx(2 * beta, i)
        
    qc.measure_all()
    return qc

# --- 팩토리 함수 업데이트 ---
def get_circuit(circuit_type, n_qubits=4):
    if circuit_type == 'ghz':
        return get_ghz_circuit(n_qubits)
    elif circuit_type == 'sparse':
        return get_sparse_block_ansatz(n_qubits)
    elif circuit_type == 'dense':
        return get_dense_ansatz(n_qubits)
    elif circuit_type == 'bv':        # NEW
        return get_bv_circuit(n_qubits)
    elif circuit_type == 'w_state':   # NEW
        return get_w_state_circuit(n_qubits)
    elif circuit_type == 'qaoa':      # NEW
        return get_simple_qaoa(n_qubits)
    else:
        raise ValueError("Invalid circuit_type. Choose: 'ghz', 'sparse', 'dense', 'bv', 'w_state', 'qaoa'")