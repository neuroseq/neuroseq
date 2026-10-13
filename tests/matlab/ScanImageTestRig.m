classdef ScanImageTestRig < handle
    properties
        hDisplay
        hScan2D
        hStackManager
        hChannels
        hRoiManager
        hFastZ
        hUserFunctions
        acqState = 'idle'
        acqsPerLoop = 1
        failStart = false
    end
    methods
        function obj = ScanImageTestRig()
            names = {'hDisplay','hScan2D','hStackManager','hChannels','hRoiManager','hFastZ','hUserFunctions'};
            for k=1:numel(names), obj.(names{k}) = ScanImageTestComponent(); end
        end
        function startGrab(obj)
            if obj.failStart, error('Test:StartFailed','Simulated failure'); end
            obj.acqState = 'grab';
            neuroseq_scanimage('event', jsonencode(struct('name','acqStart')));
        end
        function startFocus(obj)
            obj.acqState = 'focus';
        end
        function abort(obj)
            neuroseq_scanimage('event', jsonencode(struct('name','acqAbort')));
            obj.acqState = 'idle';
        end
        function complete(obj, eventBeforeIdle)
            if ~eventBeforeIdle, obj.acqState = 'idle'; end
            neuroseq_scanimage('event', jsonencode(struct('name','acqModeDone')));
            obj.acqState = 'idle';
        end
    end
end
