import unittest
from unittest import mock
from importlib.metadata import PackageNotFoundError

from modules.app_version import resolve_app_version


class AppVersionTest(unittest.TestCase):
    def test_environment_override_has_priority(self):
        self.assertEqual(
            resolve_app_version({"ASMR_APP_VERSION": "2.0.0-preview"}),
            "2.0.0-preview",
        )

    def test_packaged_runtime_uses_fallback_without_distribution_metadata(self):
        with mock.patch(
            "modules.app_version.version",
            side_effect=PackageNotFoundError,
        ):
            self.assertEqual(resolve_app_version({}), "1.5.7")


if __name__ == "__main__":
    unittest.main()
