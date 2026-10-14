classdef ScanImageTestComponent < handle
    % Minimal mutable component double; never controls microscope hardware.
    properties
        logFilePath = 'original-path'
        logFileStem = 'original-stem'
        logFileCounter = 42
        framesPerSlice = 15
        loggingEnable = false
        logAverageFactor = 1
        numSlices = 1
        mroiEnable = false
        enable = false
        channelSave = [1 2]
        channelDisplay = [1 2]
        scanFrameRate = 30
        lastFrame = {reshape(single(1:512^2),512,512), zeros(512,512,'single')}
        userFunctionsCfg = struct('ExtraField','preserve me', 'Enable',true, ...
            'Arguments',{{}}, 'UserFcnName','existing_user_function', 'EventName','acqStart')
    end
end
