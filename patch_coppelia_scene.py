from __future__ import annotations

import os
import sys

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


SCENE = r"H:\attractor\MTB_scene.ttt"
OUTPUT = r"H:\attractor\MTB_scene_wheel_closed_loop.ttt"


def main() -> None:
    client = RemoteAPIClient(host="localhost", port=23000)
    sim = client.require("sim")
    script_handle = sim.getObject("/MTB/Script")
    text = sim.getScriptStringParam(script_handle, sim.scriptstringparam_text)

    if "setExternalJointTargets" not in text:
        text = text.replace(
            "    restarting=false\n    cmdMessage=''",
            "    restarting=false\n    cmdMessage=''\n"
            "    externalControlEnabled=false\n"
            "    externalJointTargets={0,0,0,0}\n"
            "    externalControlAlpha=0.18",
            1,
        )
        function_block = r'''

function setExternalJointTargets(joints)
    if joints==nil or #joints<2 then
        return false
    end
    for i=1,4,1 do
        if joints[i]~=nil then
            externalJointTargets[i]=joints[i]
        end
    end
    externalControlEnabled=true
    robotProgramExecutionState=2
    return true
end

function disableExternalJointTargets()
    externalControlEnabled=false
    robotProgramExecutionState=1
    return true
end
'''
        text = text.replace("\nfunction sysCall_cleanup()", function_block + "\nfunction sysCall_cleanup()", 1)

    start_marker = "    -- Following section is where the script is communicating with the extension module:"
    end_marker = "    -- Report the new joint positions to the MTB robot:"
    start = text.find(start_marker)
    end = text.find(end_marker, start)
    if start < 0 or end < 0:
        raise RuntimeError("MTB actuation block markers were not found")
    old_block = text[start:end]
    if "not externalControlEnabled and serverHandle>=0" not in old_block:
        old_block = old_block.replace(
            "    if serverHandle>=0 then",
            "    if (not externalControlEnabled) and serverHandle>=0 then",
            1,
        )
        old_block += r'''
    else
        for i=1,4,1 do
            jointPositions[i]=jointPositions[i]+externalControlAlpha*(externalJointTargets[i]-jointPositions[i])
        end
    end
'''
    text = text[:start] + old_block + text[end:]

    sim.setScriptStringParam(script_handle, sim.scriptstringparam_text, text)
    sim.saveScene(OUTPUT)
    print({"script_handle": script_handle, "source_length": len(text), "saved_scene": OUTPUT})


if __name__ == "__main__":
    main()
