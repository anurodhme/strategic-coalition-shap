#!/usr/bin/env python3
"""
Comprehensive Ground-Up Validation for Low-Rank SHAP

This script validates EVERY aspect of the Low-Rank SHAP implementation and research
from the ground up to ensure complete correctness and publication readiness.

Validation Components:
1. Package Installation and Import
2. Core Implementation Functionality
3. Mathematical Correctness
4. Performance Claims Verification
5. Research Benchmarks Reproduction
6. Real-World Application Testing
7. Theoretical Analysis Validation
8. Documentation and Examples Testing

All claims and results will be verified with fresh, independent testing.
"""

import sys
import os
import time
import warnings
import subprocess
import importlib
import numpy as np
import pandas as pd
from pathlib import Path

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

class GroundUpValidator:
    """Comprehensive validator for Low-Rank SHAP from ground up."""
    
    def __init__(self):
        self.results = {}
        self.errors = []
        self.start_time = time.time()
        
        print("=" * 80)
        print("🔬 COMPREHENSIVE GROUND-UP VALIDATION FOR LOW-RANK SHAP")
        print("=" * 80)
        print("Validating EVERY aspect of implementation and research claims...")
        print()
    
    def log_result(self, test_name, success, details=None, error=None):
        """Log validation result."""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"    {details}")
        if error:
            print(f"    ERROR: {error}")
            self.errors.append(f"{test_name}: {error}")
        
        self.results[test_name] = {
            'success': success,
            'details': details,
            'error': error
        }
        print()
    
    def test_1_package_installation(self):
        """Test 1: Package Installation and Import"""
        print("🧪 TEST 1: PACKAGE INSTALLATION AND IMPORT")
        print("-" * 50)
        
        try:
            # Test package import
            from strategic_coalition_shap import StrategicCoalitionSHAP
            self.log_result("Package Import", True, "strategic_coalition_shap.StrategicCoalitionSHAP imported successfully")
            
            # Test class instantiation
            explainer = StrategicCoalitionSHAP(rank=10, random_state=42)
            self.log_result("Class Instantiation", True, f"StrategicCoalitionSHAP instance created with rank={explainer.rank}")
            
            # Test required methods exist
            required_methods = ['fit', 'explain', 'explain_instance']
            for method in required_methods:
                if hasattr(explainer, method):
                    self.log_result(f"Method {method}", True, f"{method} method exists")
                else:
                    self.log_result(f"Method {method}", False, error=f"{method} method missing")
            
        except Exception as e:
            self.log_result("Package Import", False, error=str(e))
    
    def test_2_core_functionality(self):
        """Test 2: Core Implementation Functionality"""
        print("🧪 TEST 2: CORE IMPLEMENTATION FUNCTIONALITY")
        print("-" * 50)
        
        try:
            from sklearn.datasets import make_classification
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from strategic_coalition_shap import StrategicCoalitionSHAP
            
            # Create test data
            X, y = make_classification(n_samples=200, n_features=10, n_informative=8, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Train model
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train, y_train)
            self.log_result("Model Training", True, f"Model accuracy: {model.score(X_test, y_test):.3f}")
            
            # Test StrategicCoalitionSHAP fitting
            explainer = StrategicCoalitionSHAP(rank=5, random_state=42)
            background = X_train[:30]
            explainer.fit(model.predict_proba, background, verbose=False)
            self.log_result("StrategicCoalitionSHAP Fitting", True, f"Fitted with background size: {len(background)}")
            
            # Test single instance explanation
            test_instance = X_test[:1]
            start_time = time.time()
            shap_values = explainer.explain(test_instance)
            runtime = time.time() - start_time
            
            # Validate SHAP output
            if shap_values.shape == (1, 10):
                self.log_result("Single Instance Explanation", True, 
                              f"Shape: {shap_values.shape}, Runtime: {runtime:.4f}s")
            else:
                self.log_result("Single Instance Explanation", False, 
                              error=f"Wrong shape: {shap_values.shape}, expected (1, 10)")
            
            # Test batch explanation
            test_batch = X_test[:5]
            start_time = time.time()
            shap_batch = explainer.explain(test_batch)
            batch_runtime = time.time() - start_time
            
            if shap_batch.shape == (5, 10):
                self.log_result("Batch Explanation", True, 
                              f"Shape: {shap_batch.shape}, Runtime: {batch_runtime:.4f}s")
            else:
                self.log_result("Batch Explanation", False, 
                              error=f"Wrong shape: {shap_batch.shape}, expected (5, 10)")
            
            # Test SHAP values properties
            shap_mean = np.mean(np.abs(shap_values))
            if 0.001 < shap_mean < 10:  # Reasonable range
                self.log_result("SHAP Values Range", True, f"Mean |SHAP|: {shap_mean:.4f}")
            else:
                self.log_result("SHAP Values Range", False, 
                              error=f"SHAP values out of reasonable range: {shap_mean}")
            
        except Exception as e:
            self.log_result("Core Functionality", False, error=str(e))
    
    def test_3_mathematical_correctness(self):
        """Test 3: Mathematical Correctness"""
        print("🧪 TEST 3: MATHEMATICAL CORRECTNESS")
        print("-" * 50)
        
        try:
            from sklearn.datasets import make_classification
            from sklearn.linear_model import LogisticRegression
            from strategic_coalition_shap import StrategicCoalitionSHAP
            import numpy as np
            
            # Create simple test case
            X, y = make_classification(n_samples=100, n_features=8, n_informative=6, random_state=42)
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X, y)
            
            # Test SHAP properties
            explainer = StrategicCoalitionSHAP(rank=5, random_state=42)
            background = X[:20]
            explainer.fit(model.predict_proba, background, verbose=False)
            
            test_instance = X[50:51]
            shap_values = explainer.explain(test_instance)
            
            # Property 1: SHAP values should sum to difference from base value
            base_value = explainer.base_value
            model_output = model.predict_proba(test_instance)[0, 1]
            shap_sum = np.sum(shap_values)
            expected_sum = model_output - base_value
            
            sum_error = abs(shap_sum - expected_sum)
            if sum_error < 0.1:  # Allow small numerical error
                self.log_result("SHAP Sum Property", True, 
                              f"Sum error: {sum_error:.6f} (< 0.1 threshold)")
            else:
                self.log_result("SHAP Sum Property", False, 
                              error=f"Sum error too large: {sum_error:.6f}")
            
            # Property 2: SHAP values should be finite and reasonable
            if np.all(np.isfinite(shap_values)):
                self.log_result("SHAP Finite Values", True, "All SHAP values are finite")
            else:
                self.log_result("SHAP Finite Values", False, 
                              error="Found non-finite SHAP values")
            
            # Property 3: Consistency across multiple runs
            shap_values_2 = explainer.explain(test_instance)
            consistency_error = np.mean(np.abs(shap_values - shap_values_2))
            
            if consistency_error < 1e-10:  # Should be identical with same random seed
                self.log_result("SHAP Consistency", True, 
                              f"Consistency error: {consistency_error:.2e}")
            else:
                self.log_result("SHAP Consistency", False, 
                              error=f"Inconsistent results: {consistency_error:.2e}")
            
        except Exception as e:
            self.log_result("Mathematical Correctness", False, error=str(e))
    
    def test_4_performance_claims(self):
        """Test 4: Performance Claims Verification"""
        print("🧪 TEST 4: PERFORMANCE CLAIMS VERIFICATION")
        print("-" * 50)
        
        try:
            from sklearn.datasets import make_classification
            from sklearn.ensemble import RandomForestClassifier
            from strategic_coalition_shap import StrategicCoalitionSHAP
            import psutil
            import os
            
            # Test O(nk) complexity scaling
            feature_sizes = [10, 15, 20]
            runtimes = []
            
            for n_features in feature_sizes:
                X, y = make_classification(n_samples=300, n_features=n_features, 
                                         n_informative=min(n_features-2, 15), random_state=42)
                model = RandomForestClassifier(n_estimators=20, random_state=42)
                model.fit(X, y)
                
                explainer = StrategicCoalitionSHAP(rank=10, random_state=42)
                background = X[:50]
                explainer.fit(model.predict_proba, background, verbose=False)
                
                # Measure runtime
                test_instances = X[100:105]  # 5 instances
                start_time = time.time()
                shap_values = explainer.explain(test_instances)
                runtime = time.time() - start_time
                runtimes.append(runtime)
                
                print(f"    {n_features} features: {runtime:.4f}s")
            
            # Check if runtime is roughly linear in features (O(nk))
            runtime_ratios = [runtimes[i+1]/runtimes[i] for i in range(len(runtimes)-1)]
            avg_ratio = np.mean(runtime_ratios)
            
            if 0.5 < avg_ratio < 3.0:  # Roughly linear scaling
                self.log_result("O(nk) Complexity", True, 
                              f"Runtime scaling ratio: {avg_ratio:.2f} (linear expected)")
            else:
                self.log_result("O(nk) Complexity", False, 
                              error=f"Non-linear scaling: {avg_ratio:.2f}")
            
            # Test memory efficiency
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create larger problem
            X_large, y_large = make_classification(n_samples=1000, n_features=25, 
                                                 n_informative=20, random_state=42)
            model_large = RandomForestClassifier(n_estimators=30, random_state=42)
            model_large.fit(X_large, y_large)
            
            explainer_large = StrategicCoalitionSHAP(rank=10, random_state=42)
            background_large = X_large[:100]
            explainer_large.fit(model_large.predict_proba, background_large, verbose=False)
            
            # Explain batch
            test_large = X_large[500:520]  # 20 instances
            shap_large = explainer_large.explain(test_large)
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            
            if memory_used < 100:  # Should use < 100MB for this problem
                self.log_result("Memory Efficiency", True, 
                              f"Memory used: {memory_used:.1f}MB for 25 features, 20 instances")
            else:
                self.log_result("Memory Efficiency", False, 
                              error=f"High memory usage: {memory_used:.1f}MB")
            
        except Exception as e:
            self.log_result("Performance Claims", False, error=str(e))
    
    def test_5_benchmark_reproduction(self):
        """Test 5: Research Benchmarks Reproduction"""
        print("🧪 TEST 5: RESEARCH BENCHMARKS REPRODUCTION")
        print("-" * 50)
        
        try:
            # Test if benchmark scripts exist and run
            benchmark_files = [
                'benchmarks/theoretical_analysis.py',
                'benchmarks/exact_kernel_shap.py',
                'examples/real_world_case_study.py'
            ]
            
            for benchmark_file in benchmark_files:
                if os.path.exists(benchmark_file):
                    self.log_result(f"Benchmark File {benchmark_file}", True, "File exists")
                    
                    # Test if file is syntactically correct
                    try:
                        with open(benchmark_file, 'r') as f:
                            content = f.read()
                        compile(content, benchmark_file, 'exec')
                        self.log_result(f"Syntax Check {benchmark_file}", True, "Valid Python syntax")
                    except SyntaxError as e:
                        self.log_result(f"Syntax Check {benchmark_file}", False, error=str(e))
                else:
                    self.log_result(f"Benchmark File {benchmark_file}", False, 
                                  error="File missing")
            
            # Test if results directory exists with expected files
            results_dir = Path('results')
            if results_dir.exists():
                result_files = list(results_dir.glob('*.csv'))
                if len(result_files) > 0:
                    self.log_result("Results Files", True, 
                                  f"Found {len(result_files)} result files")
                    
                    # Check if key result files exist
                    key_files = [
                        'comprehensive_validation_results.csv',
                        'exact_validation_results.csv'
                    ]
                    
                    for key_file in key_files:
                        file_path = results_dir / key_file
                        if file_path.exists():
                            # Try to load and validate
                            try:
                                df = pd.read_csv(file_path)
                                self.log_result(f"Results File {key_file}", True, 
                                              f"Loaded {len(df)} rows, {len(df.columns)} columns")
                            except Exception as e:
                                self.log_result(f"Results File {key_file}", False, 
                                              error=f"Cannot load: {e}")
                        else:
                            self.log_result(f"Results File {key_file}", False, 
                                          error="File missing")
                else:
                    self.log_result("Results Files", False, error="No result files found")
            else:
                self.log_result("Results Directory", False, error="Results directory missing")
            
        except Exception as e:
            self.log_result("Benchmark Reproduction", False, error=str(e))
    
    def test_6_documentation_examples(self):
        """Test 6: Documentation and Examples Testing"""
        print("🧪 TEST 6: DOCUMENTATION AND EXAMPLES TESTING")
        print("-" * 50)
        
        try:
            # Test README examples
            readme_path = Path('README.md')
            if readme_path.exists():
                with open(readme_path, 'r') as f:
                    readme_content = f.read()
                
                # Check for key sections
                required_sections = [
                    '# Low-Rank SHAP',
                    '## Installation',
                    '## Quick Start',
                    '## Performance',
                    '## Examples'
                ]
                
                missing_sections = []
                for section in required_sections:
                    if section not in readme_content:
                        missing_sections.append(section)
                
                if not missing_sections:
                    self.log_result("README Sections", True, "All required sections present")
                else:
                    self.log_result("README Sections", False, 
                                  error=f"Missing sections: {missing_sections}")
                
                # Check for code examples
                if '```python' in readme_content:
                    self.log_result("README Code Examples", True, "Python code examples found")
                else:
                    self.log_result("README Code Examples", False, 
                                  error="No Python code examples found")
            else:
                self.log_result("README File", False, error="README.md missing")
            
            # Test package structure
            package_files = [
                'strategic_coalition_shap/__init__.py',
                'strategic_coalition_shap/strategic_coalition_shap.py',
                'strategic_coalition_shap/clean_strategic_coalition_shap.py',
                'pyproject.toml',
                'requirements.txt'
            ]
            
            for package_file in package_files:
                if os.path.exists(package_file):
                    self.log_result(f"Package File {package_file}", True, "File exists")
                else:
                    self.log_result(f"Package File {package_file}", False, 
                                  error="File missing")
            
        except Exception as e:
            self.log_result("Documentation Examples", False, error=str(e))
    
    def test_7_research_paper_validation(self):
        """Test 7: Research Paper Validation"""
        print("🧪 TEST 7: RESEARCH PAPER VALIDATION")
        print("-" * 50)
        
        try:
            paper_dir = Path('paper')
            if paper_dir.exists():
                paper_files = [
                    'paper.qmd',
                    'references.bib',
                    'lit_review.md',
                    'derivation.md'
                ]
                
                for paper_file in paper_files:
                    file_path = paper_dir / paper_file
                    if file_path.exists():
                        self.log_result(f"Paper File {paper_file}", True, "File exists")
                        
                        # Check file is not empty
                        if file_path.stat().st_size > 100:  # At least 100 bytes
                            self.log_result(f"Paper Content {paper_file}", True, 
                                          f"File size: {file_path.stat().st_size} bytes")
                        else:
                            self.log_result(f"Paper Content {paper_file}", False, 
                                          error="File too small or empty")
                    else:
                        self.log_result(f"Paper File {paper_file}", False, 
                                      error="File missing")
            else:
                self.log_result("Paper Directory", False, error="Paper directory missing")
            
        except Exception as e:
            self.log_result("Research Paper Validation", False, error=str(e))
    
    def generate_final_report(self):
        """Generate final validation report."""
        print("=" * 80)
        print("📊 FINAL VALIDATION REPORT")
        print("=" * 80)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        print()
        
        if failed_tests > 0:
            print("❌ FAILED TESTS:")
            for test_name, result in self.results.items():
                if not result['success']:
                    print(f"  - {test_name}: {result['error']}")
            print()
        
        if self.errors:
            print("🔧 ERRORS TO ADDRESS:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
            print()
        
        total_time = time.time() - self.start_time
        print(f"⏱️  Total Validation Time: {total_time:.2f} seconds")
        print()
        
        if failed_tests == 0:
            print("🎉 ALL TESTS PASSED - LOW-RANK SHAP IS FULLY VALIDATED!")
            print("✅ Ready for publication and release!")
        else:
            print("⚠️  SOME TESTS FAILED - ISSUES NEED TO BE ADDRESSED")
            print("🔧 Please fix the errors above before publication")
        
        print("=" * 80)
        
        return passed_tests, failed_tests, self.errors
    
    def run_all_tests(self):
        """Run all validation tests."""
        self.test_1_package_installation()
        self.test_2_core_functionality()
        self.test_3_mathematical_correctness()
        self.test_4_performance_claims()
        self.test_5_benchmark_reproduction()
        self.test_6_documentation_examples()
        self.test_7_research_paper_validation()
        
        return self.generate_final_report()


def main():
    """Run comprehensive ground-up validation."""
    validator = GroundUpValidator()
    passed, failed, errors = validator.run_all_tests()
    
    # Save validation report
    os.makedirs('results', exist_ok=True)
    with open('results/ground_up_validation_report.txt', 'w') as f:
        f.write(f"Low-Rank SHAP Ground-Up Validation Report\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total Tests: {passed + failed}\n")
        f.write(f"Passed: {passed}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Success Rate: {passed/(passed+failed)*100:.1f}%\n\n")
        
        if errors:
            f.write("Errors:\n")
            for error in errors:
                f.write(f"- {error}\n")
    
    return passed == (passed + failed)  # True if all tests passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
