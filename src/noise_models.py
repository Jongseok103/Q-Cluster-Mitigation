import numpy as np
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

def build_custom_noise_model(p_depol=0.01, p_readout=0.05):
    """
    사용자 정의 노이즈 모델을 생성.
    
    Args:
        p_depol (float): 게이트 탈분극 에러 확률 (Depolarizing Error) -> 0.0 ~ 1.0 사이 값
        p_readout (float): 측정 에러 확률 (Readout Error) -> 0.0 ~ 1.0 사이 값
        
    Returns:
        NoiseModel: 생성된 Qiskit NoiseModel 객체
    """
    noise_model = NoiseModel()
    
    # 1. Depolarizing Error (게이트 에러)
    error_gate1 = depolarizing_error(p_depol, 1)
    noise_model.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3", "rx", "ry", "rz"])
    
    # 2. Readout Error (측정 에러)
    # P(0|0), P(1|0)
    # P(0|1), P(1|1)
    readout_matrix = [
        [1 - p_readout, p_readout],   
        [p_readout, 1 - p_readout]    
    ]
    error_readout = ReadoutError(readout_matrix)
    noise_model.add_all_qubit_readout_error(error_readout, ["measure"])
    
    return noise_model