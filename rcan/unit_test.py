import unittest
import torch
import numpy as np
from model import UNetSR  

class TestUNetSR(unittest.TestCase):
    def setUp(self):
        self.model = UNetSR()

    def tearDown(self):
        del self.model

    def test_forward_pass(self):
        batch_size, channels, height, width = 1, 3, 64, 64
        input_tensor = torch.randn(batch_size, channels, height, width)

        output = self.model(input_tensor)

        self.assertEqual(output.shape, torch.Size([batch_size, channels, height * self.model.upscale_factor, width * self.model.upscale_factor]))


if __name__ == '__main__':
    unittest.main()
