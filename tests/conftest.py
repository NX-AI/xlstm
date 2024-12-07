import pytest 
import torch 

if not torch.cuda.is_available():
    pytest.skip("Skipping all tests, since no GPU available")