"""Install the dependencies declared by a plugin.

Usage:
    python tools/install_plugin_deps.py <plugin_name>

The plugin's `dependencies` class attribute lists pip requirements.
This script resolves the plugin from either the supervised or recommender
registry and installs whatever it declares.
"""
import subprocess
import sys

sys.path.insert(0, "src")

from plugins import get_plugin, get_recommender_plugin, list_plugins


def main():
    if len(sys.argv) != 2:
        print("usage: install_plugin_deps.py <plugin_name>", file=sys.stderr)
        sys.exit(1)

    name = sys.argv[1]
    try:
        plugin = get_plugin(name) if name in list_plugins() else get_recommender_plugin(name)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    deps = plugin.dependencies
    if not deps:
        print(f"{name}: no extra dependencies declared.")
        return

    print(f"{name}: installing {deps}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + deps)


if __name__ == "__main__":
    main()
