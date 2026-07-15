import unittest
from pathlib import Path

import torch

from qbench.data import dataset_hash, story_sets, sha256, TOKENIZER_SHA256, TRAIN_SHA256
from qbench.run import initialize_model
from qllm.layers import QuaternionLinear
from qllm.model.transformer import ModelConfig, TransformerLM, parameter_count
from qllm.train import set_seed


def tiny(kind="real", dropout=0.0):
    return TransformerLM(ModelConfig(vocab_size=16, context_length=8, n_layers=1,
                                      n_heads=2, width=8, mlp_ratio=2,
                                      dropout=dropout, layer_kind=kind))


class ModelTests(unittest.TestCase):
    def test_phase_pair_exact_parameter_counts(self):
        real = TransformerLM(ModelConfig(1024, 64, 2, 4, 64, dropout=.1, layer_kind="real"))
        quat = TransformerLM(ModelConfig(1024, 64, 2, 4, 100, dropout=.1, layer_kind="quaternion"))
        self.assertEqual(parameter_count(real), 168576)
        self.assertEqual(parameter_count(quat), 169800)
        self.assertLessEqual(abs(parameter_count(real)-parameter_count(quat))/parameter_count(real), .01)

    def test_hamilton_identities_and_fused_reference(self):
        def product(a, b):
            ar, ai, aj, ak = a; br, bi, bj, bk = b
            return torch.tensor([ar*br-ai*bi-aj*bj-ak*bk, ar*bi+ai*br+aj*bk-ak*bj,
                                 ar*bj-ai*bk+aj*br+ak*bi, ar*bk+ai*bj-aj*bi+ak*br])
        one, i, j, k = [torch.eye(4)[n] for n in range(4)]
        for actual, expected in ((product(i, i), -one), (product(j, j), -one), (product(k, k), -one),
                                 (product(i, j), k), (product(j, k), i), (product(k, i), j)):
            torch.testing.assert_close(actual, expected)
        layer = QuaternionLinear(12, 8); inputs = torch.randn(5, 12)
        torch.testing.assert_close(layer(inputs), layer.naive_forward(inputs), atol=1e-5, rtol=1e-5)

    def test_scalar_real_embedding_is_repeated_block_diagonal(self):
        layer = QuaternionLinear(12, 8, bias=False)
        with torch.no_grad(): layer.weight_x.zero_(); layer.weight_y.zero_(); layer.weight_z.zero_()
        inputs = torch.randn(7, 12)
        # A scalar-real quaternion map is four repeated (3 -> 2) real maps,
        # not an arbitrary dense (12 -> 8) real layer.
        expected = torch.cat([part @ layer.weight_r.T for part in inputs.chunk(4, -1)], -1)
        torch.testing.assert_close(layer(inputs), expected)

    def test_float64_gradcheck(self):
        layer = QuaternionLinear(8, 8).double()
        self.assertTrue(torch.autograd.gradcheck(layer, (torch.randn(2, 8, dtype=torch.float64, requires_grad=True),)))

    def _curve(self, kind):
        set_seed(1); torch.use_deterministic_algorithms(True); model = tiny(kind, .1); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        generator = torch.Generator().manual_seed(11); curves = []
        for _ in range(200):
            x = torch.randint(0, 16, (4, 8), generator=generator); y = torch.roll(x, -1, 1)
            optimizer.zero_grad(); loss = model(x, y)[1]; loss.backward(); optimizer.step(); curves.append(loss.item())
        return curves

    def test_deterministic_200_steps(self):
        for kind in ("real", "quaternion"):
            self.assertEqual(self._curve(kind), self._curve(kind))

    def test_gaussian_ablation_shares_every_non_quaternion_parameter(self):
        config = ModelConfig(16, 8, 1, 2, 8, dropout=.1, layer_kind="quaternion")
        parcollet = initialize_model(config, 7)
        gaussian = initialize_model(config, 7, True)
        differed = False
        for (name, left), (_, right) in zip(parcollet.named_parameters(), gaussian.named_parameters()):
            quaternion = any(key in name for key in ("weight_r", "weight_x", "weight_y", "weight_z"))
            if quaternion: differed = differed or not torch.equal(left, right)
            else: self.assertTrue(torch.equal(left, right), name)
        self.assertTrue(differed)

    def test_both_models_overfit_fixed_64_sequence_batch(self):
        x = torch.arange(8).repeat(64, 1) % 16; y = torch.roll(x, -1, 1)
        for kind in ("real", "quaternion"):
            set_seed(3); model = tiny(kind); optimizer = torch.optim.AdamW(model.parameters(), lr=.03, weight_decay=0)
            for _ in range(2000):
                optimizer.zero_grad(); loss = model(x, y)[1]; loss.backward(); optimizer.step()
                if loss.item() < .1: break
            self.assertLess(loss.item(), .1, kind)

    def test_data_hash_and_story_disjointness(self):
        cache = Path("qbench/data")
        self.assertTrue((cache / "metadata.json").exists(), "run python -m qbench.data first")
        train, valid = story_sets(cache); self.assertTrue(train.isdisjoint(valid)); self.assertEqual(len(dataset_hash(cache)), 64)
        self.assertEqual(sha256(cache / "tokenizer.json"), TOKENIZER_SHA256)
        self.assertEqual(sha256(cache / "train.bin"), TRAIN_SHA256)


if __name__ == "__main__": unittest.main()
