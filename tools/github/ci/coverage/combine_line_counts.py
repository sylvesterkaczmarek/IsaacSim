#!/usr/bin/env python3

import argparse
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description="Combine line counts from multiple coverage reports")
    parser.add_argument("coverage_reports", nargs="+", help="Coverage report files to combine")
    args = parser.parse_args()

    total_line_count = 0
    total_covered_line_count = 0

    for coverage_report in args.coverage_reports:
        with open(coverage_report, "r") as f:
            coverage_data = ET.parse(f).getroot()
            total_line_count += int(coverage_data.attrib["lines-valid"])
            total_covered_line_count += int(coverage_data.attrib["lines-covered"])

    if total_line_count == 0:
        percentage_string = "N/A"
    else:
        percentage_string = f"{total_covered_line_count / total_line_count * 100:.2f}%"

    print("Processed coverage reports, total information:")
    print("--------------------------------")
    print(f"TOTAL                             {total_covered_line_count} {total_line_count}  {percentage_string}")


if __name__ == "__main__":
    main()
