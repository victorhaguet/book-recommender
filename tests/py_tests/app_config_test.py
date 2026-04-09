"""Tests for Hydra-backed application configuration helpers."""

import unittest
from unittest.mock import MagicMock, patch

from src import app_config


class TestAppConfig(unittest.TestCase):
    def test_load_settings_uses_local_deployment_by_default(self):
        """Test that load_settings defaults to the 'local' deployment when APP_ENV is not set, and that it correctly loads and composes the configuration using Hydra."""
        composed_config = MagicMock()
        hydra_context = MagicMock()
        hydra_context.__enter__.return_value = None
        hydra_context.__exit__.return_value = None

        with patch("src.app_config.load_dotenv") as mock_load_dotenv, patch(
            "src.app_config.os.getenv",
            return_value="local",
        ) as mock_getenv, patch(
            "src.app_config.GlobalHydra.instance"
        ) as mock_global_hydra, patch(
            "src.app_config.initialize_config_dir",
            return_value=hydra_context,
        ) as mock_initialize_config_dir, patch(
            "src.app_config.compose",
            return_value=composed_config,
        ) as mock_compose:
            settings = app_config.load_settings()

        self.assertIs(settings, composed_config)
        mock_load_dotenv.assert_called_once_with()
        mock_getenv.assert_called_once_with("APP_ENV", "local")
        mock_global_hydra.return_value.clear.assert_called_once_with()
        mock_initialize_config_dir.assert_called_once_with(
            version_base=None,
            config_dir=str(app_config._CONFIG_DIR),
        )
        mock_compose.assert_called_once_with(
            config_name="config",
            overrides=["deployment=local"],
        )

    def test_load_settings_merges_overrides_and_normalizes_blank_env(self):
        """Test that load_settings normalizes a blank APP_ENV to 'local', and that it correctly merges any provided overrides with the deployment override."""
        composed_config = MagicMock()
        hydra_context = MagicMock()
        hydra_context.__enter__.return_value = None
        hydra_context.__exit__.return_value = None

        with patch("src.app_config.load_dotenv"), patch(
            "src.app_config.os.getenv",
            return_value="   ",
        ), patch("src.app_config.GlobalHydra.instance") as mock_global_hydra, patch(
            "src.app_config.initialize_config_dir",
            return_value=hydra_context,
        ), patch(
            "src.app_config.compose",
            return_value=composed_config,
        ) as mock_compose:
            settings = app_config.load_settings(
                overrides=["frontend.timeout_seconds=12", "rag.top_k=5"]
            )

        self.assertIs(settings, composed_config)
        mock_global_hydra.return_value.clear.assert_called_once_with()
        mock_compose.assert_called_once_with(
            config_name="config",
            overrides=[
                "deployment=local",
                "frontend.timeout_seconds=12",
                "rag.top_k=5",
            ],
        )

    def test_load_settings_uses_app_env_when_set(self):
        """Test that load_settings uses the APP_ENV environment variable when it is set."""
        composed_config = MagicMock()
        hydra_context = MagicMock()
        hydra_context.__enter__.return_value = None
        hydra_context.__exit__.return_value = None

        with patch("src.app_config.load_dotenv"), patch(
            "src.app_config.os.getenv",
            return_value="docker",
        ), patch("src.app_config.GlobalHydra.instance"), patch(
            "src.app_config.initialize_config_dir",
            return_value=hydra_context,
        ), patch(
            "src.app_config.compose",
            return_value=composed_config,
        ) as mock_compose:
            app_config.load_settings()

        mock_compose.assert_called_once_with(
            config_name="config",
            overrides=["deployment=docker"],
        )


if __name__ == "__main__":
    unittest.main()
