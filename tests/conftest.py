def pytest_configure(config):
    config.addinivalue_line("markers", "real_weights: needs cached pretrained weights; not run by verify_equivalence.sh")
