#!/usr/bin/env python3
"""
Coverage mapping generator for Isaac Sim.

This script reads .pycov files and creates a new coverage.py rc file that maps
each file referenced in the pycov files to their corresponding source folders.
This enables proper coverage reporting by mapping build artifacts back to their
original source locations.

The script analyzes package file paths from _build/packages/ directories and
attempts to map them to their corresponding source files in ./source/extensions/,
./source/deprecated/, and ./source/python_packages/ directories.
"""

import argparse
import glob
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set

import coverage


def extract_file_paths_from_pycov(pycov_file_path: Path) -> List[str]:
    """Extract all file paths from a .pycov coverage data file.

    Args:
        pycov_file_path: Path to the .pycov file to read.

    Returns:
        List of file paths found in the coverage data. Returns empty list on error.

    Example:

    .. code-block:: python

        >>> from pathlib import Path
        >>> paths = extract_file_paths_from_pycov(Path("test.pycov"))
        >>> len(paths) > 0
        True
    """
    try:
        # Create a coverage object with the specific data file
        cov = coverage.Coverage(data_file=str(pycov_file_path))
        cov.load()
        data = cov.get_data()

        # Get all measured files
        file_paths = list(data.measured_files())
        return file_paths

    except Exception as e:
        print(f"Error reading {pycov_file_path}: {e}")
        return []


def find_pycov_files(pattern: str) -> List[Path]:
    """Find .pycov files matching a file path or glob pattern.

    Args:
        pattern: File path or glob pattern for .pycov files.

    Returns:
        Sorted list of Path objects for .pycov files matching the pattern.

    Example:

    .. code-block:: python

        >>> files = find_pycov_files("*.pycov")
        >>> all(f.suffix == ".pycov" for f in files)
        True
    """
    # If it's a direct file path, return it
    if Path(pattern).exists() and Path(pattern).is_file():
        return [Path(pattern)]

    # Otherwise, treat it as a glob pattern
    pycov_files = []
    for file_path in glob.glob(pattern, recursive=True):
        path_obj = Path(file_path)
        if path_obj.is_file() and path_obj.suffix == ".pycov":
            pycov_files.append(path_obj)

    return sorted(pycov_files)


def filter_file_paths(file_paths: List[str], exclude_prefixes: List[str]) -> List[str]:
    """Filter out file paths that start with specified prefixes.

    Args:
        file_paths: List of file paths to filter.
        exclude_prefixes: List of prefix strings to exclude from file paths.

    Returns:
        Filtered list of file paths with excluded prefixes removed.

    Example:

    .. code-block:: python

        >>> paths = ["/path/extscache/file.py", "/path/normal/file.py"]
        >>> filtered = filter_file_paths(paths, ["extscache"])
        >>> len(filtered)
        1
    """
    if not exclude_prefixes:
        return file_paths

    filtered_paths = []
    excluded_count = 0

    for file_path in file_paths:
        should_exclude = False
        for prefix in exclude_prefixes:
            if file_path.startswith(prefix) or f"/{prefix}" in file_path:
                should_exclude = True
                excluded_count += 1
                break

        if not should_exclude:
            filtered_paths.append(file_path)

    if excluded_count > 0:
        print(f"Filtered out {excluded_count} files matching exclude prefixes: {exclude_prefixes}")

    return filtered_paths


def collect_all_file_paths(pycov_files: List[Path], exclude_prefixes: List[str] = None) -> Set[str]:
    """Collect all unique file paths from multiple .pycov files.

    Args:
        pycov_files: List of Path objects pointing to .pycov files.
        exclude_prefixes: Optional list of prefix strings to exclude from file paths.

    Returns:
        Set of unique file paths found across all coverage data, with excluded prefixes filtered out.

    Example:

    .. code-block:: python

        >>> from pathlib import Path
        >>> files = [Path("test1.pycov"), Path("test2.pycov")]
        >>> paths = collect_all_file_paths(files, ["extscache"])
        >>> isinstance(paths, set)
        True
    """
    all_paths = set()

    for pycov_file in pycov_files:
        print(f"Reading: {pycov_file}")
        file_paths = extract_file_paths_from_pycov(pycov_file)

        # Apply filtering if exclude_prefixes is provided
        if exclude_prefixes:
            file_paths = filter_file_paths(file_paths, exclude_prefixes)

        all_paths.update(file_paths)
        print(f"  Found {len(file_paths)} files in this pycov")

    return all_paths


def create_path_mappings(package_file_paths: List[str]) -> Dict[str, str]:
    """Create mappings from package build paths to source paths.

    Analyzes package file paths from _build/packages/ directories and attempts
    to map them to their corresponding source files using glob patterns.
    Handles multiple hash folders under _build/packages/.

    Args:
        package_file_paths: List of file paths from package build directories.

    Returns:
        Dictionary mapping build folder paths to source folder paths.

    Example:

    .. code-block:: python

        >>> paths = ["/path/_build/packages/hash/exts/ext.name/file.py"]
        >>> mappings = create_path_mappings(paths)
        >>> isinstance(mappings, dict)
        True
    """
    if not package_file_paths:
        return {}

    path_mappings = {}
    found_files = 0

    # Group files by hash to handle multiple package folders
    hash_groups = {}
    for file_path in package_file_paths:
        if "_build/packages/" in file_path:
            hash_value = file_path.split("_build/packages/")[1].split("/")[0]
            if hash_value not in hash_groups:
                hash_groups[hash_value] = []
            hash_groups[hash_value].append(file_path)

    print(f"Found {len(hash_groups)} package hash folders: {list(hash_groups.keys())}")

    # Process each hash group separately
    for hash_value, hash_file_paths in hash_groups.items():
        print(f"Processing hash folder: {hash_value} ({len(hash_file_paths)} files)")

        for file_path in hash_file_paths:
            mapped_path = file_path.replace(f"_build/packages/{hash_value}/", "")
            extension_name = mapped_path.split("/")[1]
            file_name = mapped_path.split("/")[-1]

            # Let's start with a specific glob first
            specific_glob = glob.glob(f"source/*/{'/'.join(mapped_path.split('/')[1:])}")
            if len(specific_glob) == 1:
                build_folder = "/".join(file_path.split("/")[:-1]) + "/"
                source_folder = "/".join(specific_glob[0].split("/")[:-1]) + "/"
                path_mappings[build_folder] = source_folder
                found_files += 1
            else:
                # Try to find the file using glob pattern
                glob_pattern = f"source/*/{extension_name}/**/{file_name}"
                glob_matches = glob.glob(glob_pattern, recursive=True)

                if len(glob_matches) == 1:
                    # Extract folders for both sides
                    build_folder = "/".join(file_path.split("/")[:-1]) + "/"
                    source_folder = "/".join(glob_matches[0].split("/")[:-1]) + "/"
                    path_mappings[build_folder] = source_folder
                    found_files += 1
                elif len(glob_matches) > 1:
                    # Try with more specific pattern including parent folder
                    parent_folder = mapped_path.split("/")[-2]
                    specific_pattern = f"source/*/{extension_name}/**/{parent_folder}/{file_name}"
                    specific_matches = glob.glob(specific_pattern, recursive=True)

                    if len(specific_matches) == 1:
                        build_folder = "/".join(file_path.split("/")[:-1]) + "/"
                        source_folder = "/".join(specific_matches[0].split("/")[:-1]) + "/"
                        path_mappings[build_folder] = source_folder
                        found_files += 1
                    else:
                        print(
                            f"Unable to uniquely map file: {mapped_path} (extension: {extension_name}, file: {file_name}), matches {len(specific_matches)}"
                        )
                else:
                    print(f"Unable to find file: {mapped_path} (extension: {extension_name}, file: {file_name})")

    print(f"Successfully mapped {found_files} out of {len(package_file_paths)} package files")
    return path_mappings


def generate_coverage_rc_file(path_mappings: Dict[str, str], output_file: str) -> None:
    """Generate a coverage.py rc file with path mappings.

    Creates a new coverage rc file by copying the existing .coveragerc and
    appending path mappings for coverage.py to properly map build artifacts
    to their source locations.

    Args:
        path_mappings: Dictionary mapping build paths to source paths.
        output_file: Path to the output coverage rc file.

    Raises:
        FileNotFoundError: If .coveragerc base file does not exist.

    Example:

    .. code-block:: python

        >>> mappings = {"/build/path/": "/source/path/"}
        >>> generate_coverage_rc_file(mappings, "test.coveragerc")
    """
    try:
        shutil.copyfile(".coveragerc", output_file)
        print(f"Copied base configuration from .coveragerc to {output_file}")
    except FileNotFoundError:
        print("Warning: .coveragerc not found, creating new file")
        with open(output_file, "w") as f:
            f.write("# Generated coverage configuration\n")

    with open(output_file, "a") as f:
        f.write("\n# Auto-generated path mappings\n")
        for index, (build_path, source_path) in enumerate(path_mappings.items()):
            f.write(f"mapping_{index:05d} = \n   {source_path}\n   {build_path}\n")

    print(f"Added {len(path_mappings)} path mappings to {output_file}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace containing pycov_pattern and output_file.

    Example:

    .. code-block:: python

        >>> import sys
        >>> sys.argv = ["script.py", "*.pycov"]
        >>> args = parse_arguments()
        >>> args.pycov_pattern
        '*.pycov'
    """
    parser = argparse.ArgumentParser(
        description="Generate coverage.py rc file with path mappings from .pycov files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s _testoutput/pycov/py_cov.exttest_2025-10-06_17-25-37-632200.pycov
  %(prog)s "_testoutput/pycov/*.pycov"
  %(prog)s "_testoutput/pycov/py_cov.exttest*.pycov" --output custom.coveragerc
  %(prog)s "*.pycov" --exclude-prefix extscache --exclude-prefix kit""",
    )

    parser.add_argument("pycov_pattern", help="Path to .pycov file or glob pattern for multiple .pycov files")

    parser.add_argument(
        "--output", "-o", default=".coveragemergerc", help="Output coverage rc file (default: .coveragemergerc)"
    )

    parser.add_argument(
        "--exclude-prefix",
        action="append",
        dest="exclude_prefixes",
        help="Exclude file paths containing this prefix (can be used multiple times)",
    )

    return parser.parse_args()


def main():
    """Main entry point for the coverage mapping generator.

    Parses command line arguments, processes .pycov files with optional filtering,
    and generates a coverage.py rc file with path mappings.
    """
    args = parse_arguments()

    # Get list of pycov files
    pycov_files = find_pycov_files(args.pycov_pattern)

    if not pycov_files:
        print(f"Error: No .pycov files found matching pattern: {args.pycov_pattern}")
        sys.exit(1)

    print(f"Found {len(pycov_files)} .pycov files matching pattern: {args.pycov_pattern}")
    print("=" * 60)

    # Read all file paths from all pycov files
    all_file_paths = collect_all_file_paths(pycov_files, args.exclude_prefixes)

    # Filter for package file paths that need mapping
    package_file_paths = [fp for fp in all_file_paths if "_build/packages/" in fp]

    if not package_file_paths:
        print("No package file paths found in coverage data")
        sys.exit(1)

    print(f"Found {len(package_file_paths)} package file paths to map")
    print("=" * 60)

    # Create path mappings from build paths to source paths
    path_mappings = create_path_mappings(package_file_paths)

    if not path_mappings:
        print("Warning: No path mappings could be created")
        sys.exit(1)

    # Generate coverage.py rc file with path mappings
    generate_coverage_rc_file(path_mappings, args.output)

    print(f"\nCoverage mapping generation complete!")
    print(f"Output file: {args.output}")
    print(f"Total mappings: {len(path_mappings)}")


if __name__ == "__main__":
    main()
