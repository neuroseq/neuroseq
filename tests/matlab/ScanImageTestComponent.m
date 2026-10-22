classdef ScanImageTestComponent < handle
    % Minimal mutable component double; never controls microscope hardware.
    properties
        logFilePath = 'original-path'
        logFileStem = 'original-stem'
        logFileCounter = 42
        logFramesPerFile = 1000
        framesPerSlice = 15
        loggingEnable = false
        logAverageFactor = 10
        numSlices = 1
        actualNumSlices = 1
        mroiEnable = false
        enable = false
        channelSave = [1 2]
        channelsAvailable = 4
        channelDisplay = [1 2]
        scanFrameRate = 30
        userFunctionsCfg = struct('ExtraField','preserve me', 'Enable',true, ...
            'Arguments',{{}}, 'UserFcnName','existing_user_function', 'EventName','acqStart')
    end
end
