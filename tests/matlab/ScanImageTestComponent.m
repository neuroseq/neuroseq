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
        channelsAvailable = 4
        channelDisplay = [1 2]
        scanFrameRate = 30
        channelLUT = {[-88 436], [0 100]}
        displayRollingAverageFactor = 10
        userFunctionsCfg = struct('ExtraField','preserve me', 'Enable',true, ...
            'Arguments',{{}}, 'UserFcnName','existing_user_function', 'EventName','acqStart')
        % Like ScanImage 2023.1.0, whose getters can raise while the
        % display holds no frames.
        failLastFrameChannels = false
        failLastFrame = false
    end
    properties (Dependent)
        lastFrame
        lastFrameChannels
    end
    properties (Access = private)
        storedLastFrame = {reshape(int16(mod(0:512^2-1, 30000)),512,512), zeros(512,512,'single')}
    end
    methods
        function value = get.lastFrame(obj)
            if obj.failLastFrame
                error('MATLAB:cellRefFromNonCell', 'Brace indexing is not supported for variables of this type.');
            end
            value = obj.storedLastFrame;
        end
        function set.lastFrame(obj, value)
            obj.storedLastFrame = value;
        end
        function value = get.lastFrameChannels(obj)
            if obj.failLastFrameChannels
                error('MATLAB:cellRefFromNonCell', 'Brace indexing is not supported for variables of this type.');
            end
            value = obj.channelDisplay;
        end
    end
end
