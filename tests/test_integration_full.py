from src.quantum_hardware.backend_manager import QuantumBackendManager


def test_full_pipeline_smoke():
    backend = QuantumBackendManager("ibm")
    devices = backend.list_available_devices()
    assert devices
