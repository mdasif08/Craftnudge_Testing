#!/usr/bin/env python3
"""
Comprehensive Test Runner for CraftNudge Microservices

This script runs all unit tests with 100% code coverage and quality checks.
It provides detailed reporting and ensures all tests pass before proceeding.
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class TestRunner:
    """Comprehensive test runner with coverage and quality checks."""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_dir = self.project_root / "tests"
        self.coverage_dir = self.project_root / "htmlcov"
        self.results = {
            "start_time": datetime.now().isoformat(),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "coverage_percentage": 0.0,
            "quality_score": 0.0,
            "errors": [],
            "warnings": [],
        }

    def run_command(
        self, command: List[str], capture_output: bool = True
    ) -> subprocess.CompletedProcess:
        """Run a command and return the result."""
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=capture_output,
                text=True,
                timeout=300,  # 5 minutes timeout
            )
            return result
        except subprocess.TimeoutExpired:
            raise Exception(f"Command timed out: {' '.join(command)}")
        except Exception as e:
            raise Exception(f"Command failed: {' '.join(command)} - {str(e)}")

    def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed."""
        print("🔍 Checking dependencies...")

        required_packages = [
            "pytest",
            "pytest-cov",
            "pytest-asyncio",
            "pytest-mock",
            "coverage",
            "black",
            "flake8",
            "mypy",
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            print(f"❌ Missing packages: {', '.join(missing_packages)}")
            print("Please install missing packages:")
            print(f"pip install {' '.join(missing_packages)}")
            return False

        print("✅ All dependencies are installed")
        return True

    def run_linting(self) -> bool:
        """Run code linting with flake8."""
        print("\n🔍 Running code linting...")

        try:
            result = self.run_command(
                [
                    "python",
                    "-m",
                    "flake8",
                    "--max-line-length=100",
                    "--ignore=E203,W503",
                    "--exclude=venv,__pycache__,*.pyc",
                    ".",
                ]
            )

            if result.returncode == 0:
                print("✅ Code linting passed")
                return True
            else:
                print("❌ Code linting failed:")
                print(result.stdout)
                print(result.stderr)
                self.results["errors"].append("Code linting failed")
                return False

        except Exception as e:
            print(f"❌ Linting error: {str(e)}")
            self.results["errors"].append(f"Linting error: {str(e)}")
            return False

    def run_type_checking(self) -> bool:
        """Run type checking with mypy."""
        print("\n🔍 Running type checking...")

        try:
            result = self.run_command(
                [
                    "python",
                    "-m",
                    "mypy",
                    "--ignore-missing-imports",
                    "--disallow-untyped-defs",
                    "--disallow-incomplete-defs",
                    "--check-untyped-defs",
                    "--disallow-untyped-decorators",
                    "--no-implicit-optional",
                    "--warn-redundant-casts",
                    "--warn-unused-ignores",
                    "--warn-return-any",
                    "--warn-unreachable",
                    "--strict-equality",
                    ".",
                ]
            )

            if result.returncode == 0:
                print("✅ Type checking passed")
                return True
            else:
                print("❌ Type checking failed:")
                print(result.stdout)
                print(result.stderr)
                self.results["warnings"].append("Type checking issues found")
                return False

        except Exception as e:
            print(f"❌ Type checking error: {str(e)}")
            self.results["warnings"].append(f"Type checking error: {str(e)}")
            return False

    def run_code_formatting_check(self) -> bool:
        """Check code formatting with black."""
        print("\n🔍 Checking code formatting...")

        try:
            result = self.run_command(["python", "-m", "black", "--check", "--diff", "."])

            if result.returncode == 0:
                print("✅ Code formatting is correct")
                return True
            else:
                print("❌ Code formatting issues found:")
                print(result.stdout)
                print(result.stderr)
                self.results["warnings"].append("Code formatting issues found")
                return False

        except Exception as e:
            print(f"❌ Formatting check error: {str(e)}")
            self.results["warnings"].append(f"Formatting check error: {str(e)}")
            return False

    def run_unit_tests(self) -> bool:
        """Run all unit tests with coverage."""
        print("\n🧪 Running unit tests with coverage...")

        try:
            result = self.run_command(
                [
                    "python",
                    "-m",
                    "pytest",
                    "--cov=.",
                    "--cov-report=term-missing",
                    "--cov-report=html:htmlcov",
                    "--cov-report=xml:coverage.xml",
                    "--cov-fail-under=100",
                    "--verbose",
                    "--tb=short",
                    "--maxfail=10",
                    "tests/",
                ]
            )

            if result.returncode == 0:
                print("✅ All unit tests passed")

                # Parse coverage information
                coverage_lines = result.stdout.split("\n")
                for line in coverage_lines:
                    if "TOTAL" in line and "%" in line:
                        try:
                            percentage = float(line.split("%")[0].split()[-1])
                            self.results["coverage_percentage"] = percentage
                            break
                        except (ValueError, IndexError):
                            pass

                return True
            else:
                print("❌ Unit tests failed:")
                print(result.stdout)
                print(result.stderr)
                self.results["errors"].append("Unit tests failed")
                return False

        except Exception as e:
            print(f"❌ Unit test error: {str(e)}")
            self.results["errors"].append(f"Unit test error: {str(e)}")
            return False

    def run_specific_test_modules(self) -> Dict[str, bool]:
        """Run tests for specific modules and return results."""
        print("\n🧪 Running specific test modules...")

        test_modules = [
            "test_commit_tracker",
            "test_commit_quality_coaching",
            "test_shared_models",
            "test_shared_events",
            "test_shared_database",
            "test_config_settings",
            "test_cli_tools",
        ]

        results = {}

        for module in test_modules:
            print(f"  Testing {module}...")
            try:
                result = self.run_command(
                    ["python", "-m", "pytest", f"tests/{module}.py", "--verbose", "--tb=short"]
                )

                success = result.returncode == 0
                results[module] = success

                if success:
                    print(f"    ✅ {module} passed")
                else:
                    print(f"    ❌ {module} failed")
                    print(f"    {result.stdout}")
                    print(f"    {result.stderr}")

            except Exception as e:
                print(f"    ❌ {module} error: {str(e)}")
                results[module] = False

        return results

    def calculate_quality_score(self) -> float:
        """Calculate overall quality score based on test results."""
        score = 100.0

        # Deduct points for errors
        score -= len(self.results["errors"]) * 10

        # Deduct points for warnings
        score -= len(self.results["warnings"]) * 2

        # Deduct points for coverage below 100%
        if self.results["coverage_percentage"] < 100:
            score -= (100 - self.results["coverage_percentage"]) * 0.5

        # Deduct points for failed tests
        if self.results["tests_failed"] > 0:
            score -= self.results["tests_failed"] * 5

        return max(0.0, score)

    def generate_report(self) -> str:
        """Generate a comprehensive test report."""
        self.results["end_time"] = datetime.now().isoformat()
        self.results["quality_score"] = self.calculate_quality_score()

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                    CRAFTNUDGE TEST REPORT                    ║
╚══════════════════════════════════════════════════════════════╝

📊 Test Summary:
   • Start Time: {self.results['start_time']}
   • End Time: {self.results['end_time']}
   • Tests Run: {self.results['tests_run']}
   • Tests Passed: {self.results['tests_passed']}
   • Tests Failed: {self.results['tests_failed']}
   • Coverage: {self.results['coverage_percentage']:.1f}%
   • Quality Score: {self.results['quality_score']:.1f}/100

"""

        if self.results["errors"]:
            report += "❌ Errors:\n"
            for error in self.results["errors"]:
                report += f"   • {error}\n"
            report += "\n"

        if self.results["warnings"]:
            report += "⚠️  Warnings:\n"
            for warning in self.results["warnings"]:
                report += f"   • {warning}\n"
            report += "\n"

        if self.results["quality_score"] >= 90:
            report += "🎉 EXCELLENT QUALITY - Ready for production!\n"
        elif self.results["quality_score"] >= 80:
            report += "✅ GOOD QUALITY - Minor issues to address\n"
        elif self.results["quality_score"] >= 70:
            report += "⚠️  ACCEPTABLE QUALITY - Issues need attention\n"
        else:
            report += "❌ POOR QUALITY - Significant issues must be fixed\n"

        return report

    def save_results(self):
        """Save test results to JSON file."""
        results_file = self.project_root / "test_results.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📄 Test results saved to: {results_file}")

    def run_all_tests(self) -> bool:
        """Run all tests and quality checks."""
        print("🚀 Starting comprehensive test suite...")
        print("=" * 60)

        # Check dependencies
        if not self.check_dependencies():
            return False

        # Run quality checks
        linting_passed = self.run_linting()
        type_checking_passed = self.run_type_checking()
        formatting_passed = self.run_code_formatting_check()

        # Run unit tests
        unit_tests_passed = self.run_unit_tests()

        # Run specific test modules
        module_results = self.run_specific_test_modules()

        # Calculate results
        self.results["tests_run"] = len(module_results)
        self.results["tests_passed"] = sum(1 for passed in module_results.values() if passed)
        self.results["tests_failed"] = self.results["tests_run"] - self.results["tests_passed"]

        # Generate and display report
        report = self.generate_report()
        print(report)

        # Save results
        self.save_results()

        # Determine overall success
        overall_success = (
            linting_passed
            and type_checking_passed
            and formatting_passed
            and unit_tests_passed
            and self.results["tests_failed"] == 0
            and self.results["coverage_percentage"] >= 100
            and self.results["quality_score"] >= 90
        )

        if overall_success:
            print("🎉 ALL TESTS PASSED WITH 100% COVERAGE AND 90+ QUALITY SCORE!")
            print("✅ Ready to proceed to the next implementation step!")
        else:
            print("❌ Some tests failed or quality requirements not met.")
            print("🔧 Please fix the issues before proceeding.")

        return overall_success


def main():
    """Main entry point."""
    runner = TestRunner()
    success = runner.run_all_tests()

    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
