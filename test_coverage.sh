#! /bin/bash

# run tests under the torchrec dir
coverage run -m pytest torchrec -v -s -W ignore::pytest.PytestCollectionWarning --continue-on-collection-errors -k 'not test_sharding_gloo_cw'

# run tests under the eagle_plus_tests dir
coverage run -m pytest eagle_plus_tests -v -s -W ignore::pytest.PytestCollectionWarning --continue-on-collection-errors -k 'not test_sharding_gloo_cw'

coverage combine

# generaete a coverage report
coverage html -i --skip-empty --omit "*/tests/test_*.py","*/test_utils/test_*.py","*_tests.py" --include="torchrec/*"
