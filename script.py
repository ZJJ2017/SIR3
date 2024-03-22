# -*- coding: utf-8 -*-

"""
nohup python script.py > /dev/null 2>&1 &
"""

import os
import time

# wait
# time.sleep((2)*60*60)

def runScript(DevIndex, paramsV):
    num = 5
    for i in range(1, 1+num):
        devV = " CUDA_VISIBLE_DEVICES="+str(DevIndex)
        beginV = " nohup python run.py "
        seed = " --seed " + str(i)
        endV = " > /dev/null 2>&1 &" 
        script = beginV + paramsV + seed + endV
        time.sleep(i)
        os.system(script)

runScript(0, "--env HalfCheetah-v2 --policy SIR3 --isSparse --reLabeling --reLabelingDone --trainBC --label noRL")
runScript(1, "--env HalfCheetah-v2 --policy SIR3 --isSparse --reLabeling --reLabelingDone --trainWithoutBC --label noIL")
