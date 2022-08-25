#/bin/bash!
if [ ! -d "./__logs" ]
then
	mkdir __logs
fi
python -m unittest eagle_plus_tests.plustest_sharding_gloo_cw >& __logs/log-$RANDOM  
#TODO: fix log file
