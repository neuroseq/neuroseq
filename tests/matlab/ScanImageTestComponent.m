classdef ScanImageTestComponent < handle
    % Minimal mutable component double; never controls microscope hardware.
    properties
        logFilePath = 'original-path'
        logFileStem = 'original-stem'
        logFileCounter = 42
        framesPerSlice = 15
        loggingEnable = false
        logAverageFactor = 10
        numSlices = 1
        actualNumSlices = 1
        mroiEnable = false
        enable = false
        channelSave = [1 2]
        channelDisplay = [1 2]
        scanFrameRate = 30
        lastFrame = {reshape(int16(mod(0:512^2-1, 30000)),512,512), zeros(512,512,'single')}
        channelLUT = {[-88 436], [0 100]}
        displayRollingAverageFactor = 10
        userFunctionsCfg = struct('ExtraField','preserve me', 'Enable',true, ...
            'Arguments',{{}}, 'UserFcnName','existing_user_function', 'EventName','acqStart')
    end
end
