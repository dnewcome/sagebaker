"""Single entry point for synthetic training data generation.

Usage
-----
    python prepare.py --plugin fillrate [--output-dir data] [--seed 42] [-- --rows 20000]
    python prepare.py --plugin clh      [--output-dir data] [--seed 42] [-- --creators 1000]

The ``--plugin`` name is resolved against both the supervised and recommender
registries. Any arguments after ``--`` are forwarded verbatim to the plugin's
``prepare_data()`` so each plugin can define its own CLI surface without
polluting this script.
"""
import argparse
import os
import sys

# Allow running from the repo root without installing the package.
# This file lives in prep/, so src/ is one directory up.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from plugins import get_plugin, get_recommender_plugin, list_plugins, list_recommender_plugins


def resolve_plugin(name: str):
    """Try supervised registry first, then recommender registry."""
    try:
        return get_plugin(name)
    except ValueError:
        pass
    try:
        return get_recommender_plugin(name)
    except ValueError:
        pass
    all_plugins = list_plugins() + list_recommender_plugins()
    raise SystemExit(
        f"Unknown plugin '{name}'. Available: {all_plugins}\n"
        "Register new plugins in src/plugins/__init__.py."
    )


def main():
    # Split on '--' so plugin-specific flags don't confuse our parser.
    argv = sys.argv[1:]
    if "--" in argv:
        split = argv.index("--")
        our_argv = argv[:split]
        plugin_argv = argv[split + 1:]
    else:
        our_argv = argv
        plugin_argv = []

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--plugin", required=True,
        help=f"plugin name. supervised: {list_plugins()}  recommender: {list_recommender_plugins()}",
    )
    parser.add_argument(
        "--output-dir", default="data",
        help="directory to write the dataset into (default: data/)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="random seed (default: 42)",
    )
    args = parser.parse_args(our_argv)

    plugin = resolve_plugin(args.plugin)
    print(f"plugin:     {plugin.name}")
    print(f"output-dir: {args.output_dir}")
    print(f"seed:       {args.seed}")
    if plugin_argv:
        print(f"plugin args:{plugin_argv}")
    print()

    plugin.prepare_data(
        output_dir=args.output_dir,
        seed=args.seed,
        extra_args=plugin_argv,
    )


if __name__ == "__main__":
    main()
