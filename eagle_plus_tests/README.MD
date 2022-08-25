# Execute additional tests and measure code coverage 
This directory contains the script to measure code coverage. It also contains the additional test files and test  cases that we added to test TorchRec further. 

## How to run tests and measure code coverage
1. (Optional) If coverage.py and pytest are not installed in your development environment, install them (both are open source software):
```
pip install coverage
pip install pytest
```

2. To run all test files and measure coverage using coverage.py, go to the project's root directory, and execute the test_coverage.sh file:

```
cd ..
bash test_coverage.sh
```

3. The coverage report should be in the folder `./htmlcov`. Open index.html, which shows the overall coverage and coverage breakdown by file. You can click each link to see which lines are covered by existing test cases. 


## How to change configurations

1. The default `test_coverage.sh` contains commands to execute all tests under `torchrec/` and `eagle_plus_tests/`. To run only tests under one of the two directories, delete the relevant `coverage run ...` command in file `test_coverage.sh`.

2. You can modify the two configuration files under the project's root directory for customized settings.

<ol>

- `.coveragerc` is the configuration file for the `coverage run` command. It specifies the source directory for coverage measurement and enables coverage measurement for multiprocessing. To turn off multiprocessing, remove `concurrency = multiprocessing` from `.coveragerc`. This file is required because the subprocess only sees options in the configuration file, and options set on the command line will not be used in the subprocesses. 

- `pytest.ini` is the configuration file for pytest. It specified the patterns for the test files.
