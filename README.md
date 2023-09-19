## “Old-school” Guidance to Learn A Better Traffic Signal Control Agent

This is official repo of paper "“Old-school” Guidance to Learn A Better Traffic Signal Control Agent".

### How to use
First, you need to [install sumo](https://sumo.dlr.de/docs/Downloads.php) or install it from requirement.txt, and then you need to set SUMO_HOME in the environment variable. For example, if sumo is installed from requirement.txt, the env should be setted like:
```shell
export SUMO_HOME=/your python env path/lib/python3.6/site-packages/sumo
```
Second, export PYTHONPATH to the root directory of this folder. That is
```shell
export PYTHONPATH=${PYTHONPATH}:/your own folder/root directory of this folder
```
Third, unzip scenario files:
```shell
cd sumo_files/scenarios
unzip sumo_fenglin_base_sub1.zip
cd -
```
Final:
- Model training
```shell
python -u tsc/main.py
```
- Eval
```shell
python -u tsc/eval_v2.py   # Evaluate all checkpoints and save results.
# or
python -u tsc/eval_model.py  # Evaluate the specified checkpoint, you can also adjust eval_config for visualization
```

Note that we currently do not provide scats in the guidance model here (the relevant code has been commented out). You can refer to official scats and modify get_sdk_label in PPO.py to complete.