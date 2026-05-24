import unittest

import torch

from mesd.mesd_trainer import ESDConfig, StableDiffusionESDAdapter, select_parameter_names

try:
    from diffusers import UNet2DConditionModel
except ModuleNotFoundError:
    UNet2DConditionModel = None


SD14_MODEL_ID = "CompVis/stable-diffusion-v1-4"
EXACT_TARGET = "up_blocks.3.attentions.2.transformer_blocks.0.ff.net.2"
PARENT_TARGET = "up_blocks.3.attentions"
OUTSIDE_TARGET_PREFIX = "down_blocks.0"


@unittest.skipIf(UNet2DConditionModel is None, "diffusers is not installed")
class TestSpecificLayerSelectionSD14(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.unet = UNet2DConditionModel.from_pretrained(
                SD14_MODEL_ID,
                subfolder="unet",
                torch_dtype=torch.float32,
            )
        except Exception as exc:
            raise unittest.SkipTest(f"Could not load SD 1.4 UNet: {exc}") from exc

    @classmethod
    def tearDownClass(cls):
        del cls.unet

    def setUp(self):
        self.adapter = StableDiffusionESDAdapter()

    def make_config(self, target_layers=None):
        return ESDConfig(
            family="sd",
            base_model_id=SD14_MODEL_ID,
            erase_concept="cat",
            erase_from=None,
            train_method="specific-layer",
            iterations=1,
            lr=None,
            negative_guidance=1.0,
            num_inference_steps=10,
            guidance_scale=7.5,
            batch_size=1,
            resolution=None,
            save_path="/tmp",
            device="cpu",
            target_layers=target_layers,
        )

    def test_normalize_method_accepts_specific_layer(self):
        self.assertEqual(self.adapter.normalize_train_method("specific-layer"), "specific-layer")

    def test_real_unet_contains_expected_sd14_target_module(self):
        module_names = dict(self.unet.named_modules())
        self.assertIn(EXACT_TARGET, module_names)
        self.assertIsInstance(module_names[EXACT_TARGET], torch.nn.Linear)

    def test_select_parameter_names_utility_filters_exact_real_sd14_layer(self):
        selected = select_parameter_names(
            self.unet,
            lambda module_name: module_name == EXACT_TARGET,
        )

        self.assertEqual(
            selected,
            [
                f"{EXACT_TARGET}.weight",
                f"{EXACT_TARGET}.bias",
            ],
        )

    def test_adapter_selects_only_requested_real_sd14_subtree(self):
        config = self.make_config(target_layers=[PARENT_TARGET])

        selected = self.adapter.select_parameter_names(
            self.unet,
            "specific-layer",
            config,
        )

        self.assertTrue(selected)
        self.assertTrue(any(name.startswith(f"{PARENT_TARGET}.0") for name in selected), selected)
        self.assertTrue(any(name.startswith(EXACT_TARGET) for name in selected), selected)
        self.assertFalse(any(name.startswith("up_blocks.2.attentions") for name in selected), selected)
        self.assertFalse(any(name.startswith(OUTSIDE_TARGET_PREFIX) for name in selected), selected)

    def test_adapter_requires_target_layers_for_specific_layer(self):
        config = self.make_config(target_layers=None)

        with self.assertRaisesRegex(ValueError, "requires config.target_layers"):
            self.adapter.select_parameter_names(self.unet, "specific-layer", config)


if __name__ == "__main__":
    unittest.main()
