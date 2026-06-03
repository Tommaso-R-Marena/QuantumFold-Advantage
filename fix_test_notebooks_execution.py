path = '.github/workflows/test-notebooks-execution.yml'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Add numpy to production and structure jobs
content = content.replace(
    'pip install pytest nbformat nbconvert torch',
    'pip install pytest nbformat nbconvert torch numpy'
).replace(
    'pip install pytest nbformat nbconvert torch pandas matplotlib seaborn psutil',
    'pip install pytest nbformat nbconvert torch pandas matplotlib seaborn psutil numpy'
)

# Fix 2: Correct class names in test-production-notebook
# Actual classes in tests/test_production_notebook.py:
# TestNotebookStructure, TestConfigurationAndSetup, TestDataTrainingEvaluationFlow,
# TestVisualizationAndExport, TestReliabilityAndReproducibility, TestProductionExecution, ...

content = content.replace(
    'pytest tests/test_production_notebook.py::TestConfigurationCells -v',
    'pytest tests/test_production_notebook.py::TestConfigurationAndSetup -v'
).replace(
    'pytest tests/test_production_notebook.py::TestDataProcessing -v',
    'pytest tests/test_production_notebook.py::TestDataTrainingEvaluationFlow -v'
).replace(
    'pytest tests/test_production_notebook.py::TestModelTraining -v',
    'pytest tests/test_production_notebook.py::TestProductionExecution -v'
).replace(
    'pytest tests/test_production_notebook.py::TestEvaluation -v',
    'pytest tests/test_production_notebook.py::TestDataTrainingEvaluationFlow -v' # Re-using for now as it covers both
).replace(
    'pytest tests/test_production_notebook.py::TestVisualization -v',
    'pytest tests/test_production_notebook.py::TestVisualizationAndExport -v'
).replace(
    'pytest tests/test_production_notebook.py::TestResultsExport -v',
    'pytest tests/test_production_notebook.py::TestVisualizationAndExport -v'
).replace(
    'pytest tests/test_production_notebook.py::TestMemoryManagement -v',
    'pytest tests/test_production_notebook.py::TestReliabilityAndReproducibility -v'
).replace(
    'pytest tests/test_production_notebook.py::TestErrorHandling -v',
    'pytest tests/test_production_notebook.py::TestReliabilityAndReproducibility -v'
).replace(
    'pytest tests/test_production_notebook.py::TestReproducibility -v',
    'pytest tests/test_production_notebook.py::TestReliabilityAndReproducibility -v'
).replace(
    'pytest tests/test_production_notebook.py::TestDocumentation -v',
    'pytest tests/test_production_notebook.py::TestNotebookStructure -v'
)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
