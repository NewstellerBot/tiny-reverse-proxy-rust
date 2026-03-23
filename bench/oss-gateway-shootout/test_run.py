import importlib.util
import pathlib
import sys
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("run.py")
SPEC = importlib.util.spec_from_file_location("oss_gateway_shootout_run", MODULE_PATH)
RUN = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = RUN
SPEC.loader.exec_module(RUN)


class TinyProxyConfigTests(unittest.TestCase):
    def test_prompt_cache_configs_use_family_and_surfaces(self) -> None:
        config = RUN.make_tiny_proxy_config_for_scenario("prompt-cache")
        self.assertIn('family = "openai"', config)
        self.assertIn("[providers.surfaces]", config)
        self.assertIn("[providers.surfaces.prompt_cache]", config)
        self.assertNotIn("capabilities =", config)

    def test_tool_round_trip_configs_use_family_and_surfaces(self) -> None:
        config = RUN.make_tiny_proxy_config_for_scenario("tool-round-trip")
        self.assertIn('family = "openai"', config)
        self.assertIn("[providers.surfaces]", config)
        self.assertIn('tools = "openai"', config)
        self.assertNotIn("tool_protocol =", config)


if __name__ == "__main__":
    unittest.main()
