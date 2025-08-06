import unittest
import torch
from codonmoe.codon_moe import LayerNorm, MixtureOfExperts, CodonMoE, mRNAModel

class TestCodonMoE(unittest.TestCase):
    def setUp(self):
        self.input_dim = 768
        self.batch_size = 2
        self.seq_len = 50
        self.num_experts = 4

    def test_layer_norm(self):
        layer_norm = LayerNorm(self.input_dim)
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        output = layer_norm(x)
        self.assertEqual(output.shape, x.shape)
        self.assertAlmostEqual(output.mean().item(), 0, places=6)
        # Allow for a small tolerance in the standard deviation
        self.assertAlmostEqual(output.std().item(), 1, places=2)

    def test_mixture_of_experts(self):
        moe = MixtureOfExperts(self.input_dim * 3, self.num_experts)
        x = torch.randn(self.batch_size, self.seq_len // 3, self.input_dim * 3)
        output = moe(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len // 3, self.input_dim))

    def test_codon_moe(self):
        codon_moe = CodonMoE(self.input_dim, self.num_experts)
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        output = codon_moe(x)
        self.assertEqual(output.shape, (self.batch_size, 1))

    def test_mrna_model(self):
        class DummyBaseModel(torch.nn.Module):
            def forward(self, input_ids):
                return torch.randn(input_ids.shape[0], 50, 768)

        base_model = DummyBaseModel()
        codon_moe = CodonMoE(self.input_dim, self.num_experts)
        mrna_model = mRNAModel(base_model, codon_moe)

        input_ids = torch.randint(0, 1000, (self.batch_size, 100))
        output = mrna_model(input_ids)
        self.assertEqual(output.shape, (self.batch_size, 1))

if __name__ == '__main__':
    unittest.main()