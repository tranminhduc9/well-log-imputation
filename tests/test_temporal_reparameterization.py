"""Affine branch fusion must preserve boundary samples, biases and predictions."""
import unittest
import torch

from src.models.conv_saits.config import ConvSAITSConfig
from src.models.conv_saits.network import ConvSAITSNetwork
from src.models.conv_saits.reparameterize import fold_temporal_branches


class TemporalReparameterizationTests(unittest.TestCase):
    def test_max_kernel_preserves_full_network_in_double_precision(self):
        torch.manual_seed(912)
        config=ConvSAITSConfig(seq_len=17,n_features=3,encoder_channels=2,
                              d_model=12,d_inner=24,n_heads=3,n_layers=1,
                              multi_scale_kernels=(3,7,15),dropout=0.,attn_dropout=0.,device='cpu')
        network=ConvSAITSNetwork(config).double().eval()
        values=torch.randn(2,17,3,dtype=torch.float64)
        observed=torch.randint(0,2,values.shape).double()
        random_state=torch.random.get_rng_state()
        folded=fold_temporal_branches(network)
        self.assertTrue(torch.equal(random_state,torch.random.get_rng_state()))
        with torch.no_grad():
            before=network(values*observed,observed)
            after=folded(values*observed,observed)
        for x,y in zip(before,after):
            torch.testing.assert_close(x,y,rtol=1e-11,atol=1e-12)
        self.assertIsNotNone(network.encoder.temporal_branches)
        self.assertEqual(folded.encoder.temporal.kernel_size,(15,))
        self.assertIsNone(folded.encoder.temporal_branches)


if __name__=='__main__':
    unittest.main()
