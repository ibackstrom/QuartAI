import unittest

import torch

from qllm.layers import ComplexLinear, QuaternionLinear, RealLinear


class HypercomplexLayerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_unit_quaternion_product_has_unit_norm(self):
        a = torch.randn(32, 4); a /= a.norm(dim=-1, keepdim=True)
        b = torch.randn(32, 4); b /= b.norm(dim=-1, keepdim=True)
        ar, ax, ay, az = a.unbind(-1); br, bx, by, bz = b.unbind(-1)
        product = torch.stack((ar*br-ax*bx-ay*by-az*bz, ar*bx+ax*br+ay*bz-az*by,
                               ar*by-ax*bz+ay*br+az*bx, ar*bz+ax*by-ay*bx+az*br), -1)
        torch.testing.assert_close(product.norm(dim=-1), torch.ones(32), atol=1e-5, rtol=1e-5)

    def test_quaternion_layer_matches_component_reference(self):
        layer = QuaternionLinear(12, 8, bias=False)
        inputs = torch.randn(5, 12)
        ir, ix, iy, iz = inputs.chunk(4, dim=-1)
        r, x, y, z = layer.weight_r, layer.weight_x, layer.weight_y, layer.weight_z
        # Weight quaternion multiplied by input quaternion (order matters).
        expected = torch.cat((ir@r.T-ix@x.T-iy@y.T-iz@z.T, ir@x.T+ix@r.T-iy@z.T+iz@y.T,
                              ir@y.T+ix@z.T+iy@r.T-iz@x.T, ir@z.T-ix@y.T+iy@x.T+iz@r.T), -1)
        torch.testing.assert_close(layer(inputs), expected)

    def test_exact_weight_parameter_ratios(self):
        real = sum(p.numel() for p in RealLinear(16, 24, bias=False).parameters())
        complex_count = sum(p.numel() for p in ComplexLinear(16, 24, bias=False).parameters())
        quat = sum(p.numel() for p in QuaternionLinear(16, 24, bias=False).parameters())
        self.assertEqual(complex_count * 2, real)
        self.assertEqual(quat * 4, real)

    def test_three_layer_gradient_flow(self):
        stack = torch.nn.Sequential(QuaternionLinear(16, 16), torch.nn.GELU(),
                                    QuaternionLinear(16, 16), torch.nn.GELU(), QuaternionLinear(16, 16))
        stack(torch.randn(4, 16)).square().mean().backward()
        for parameter in stack.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
