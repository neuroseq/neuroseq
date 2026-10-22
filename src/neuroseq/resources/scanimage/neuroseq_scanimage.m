function reply = neuroseq_scanimage(operation, payload)
% NeuroSeq adapter for ScanImage Premium 2023.1.0, single-plane raster imaging.
% The installed API is checked before controls become available.
persistent run ownedFocus
started = tic;
if isempty(ownedFocus), ownedFocus = false; end
if isempty(run), run = idle_run(); end
args = jsondecode(payload);
hSI = evalin('base', 'hSI');
if strcmp(operation, 'event')
    if strcmp(args.name, 'focusDone'), ownedFocus = false; end
    if ~isempty(run.id) && any(strcmp(run.state, {'starting','waiting_for_first_frame','recording'}))
        switch args.name
            case 'acqStart', run.state = 'recording';
            case 'acqModeDone', run.state = 'completed';
            case 'acqAbort', run.state = 'aborted';
        end
    end
    run = restore_settings(hSI, run);
    reply = '{}'; return
end
run = restore_settings(hSI, run);
switch operation
    case 'probe'
        assert(isprop(hSI, 'extTrigEnable'), 'Missing ScanImage property: extTrigEnable');
        required = {'logFilePath','logFileStem','logFileCounter'};
        for k=1:numel(required)
            assert(isprop(hSI.hScan2D, required{k}), 'Missing ScanImage property: %s', required{k});
        end
        assert(isprop(hSI.hStackManager, 'framesPerSlice'), 'Missing framesPerSlice');
        assert(isprop(hSI.hChannels, 'loggingEnable'), 'Missing loggingEnable');
        cfg = hSI.hUserFunctions.userFunctionsCfg;
        originalCfg = cfg;
        names = {'acqStart','acqModeDone','acqAbort','focusDone'};
        for k=1:numel(names)
            existing = [];
            if ~isempty(cfg)
                existing = find(strcmp({cfg.UserFcnName}, 'neuroseq_scanimage_event') & ...
                    strcmp({cfg.EventName}, names{k}));
            end
            if ~isempty(existing)
                cfg(existing(1)).Enable = true;
                cfg(existing(1)).Arguments = {};
                cfg(existing(2:end)) = [];
                continue
            end
            if isempty(cfg) && (~isstruct(cfg) || isempty(fieldnames(cfg)))
                entry = struct('EventName','', 'UserFcnName','', 'Arguments',{{}}, 'Enable',true);
            else
                % Preserve the installation's field set and field order.
                entry = cell2struct(cell(size(fieldnames(cfg))), fieldnames(cfg), 1);
            end
            entry.EventName = names{k}; entry.UserFcnName = 'neuroseq_scanimage_event';
            entry.Arguments = {}; entry.Enable = true;
            if isempty(cfg), cfg = entry; else, cfg(end+1) = entry; end
        end
        if ~isequaln(cfg, originalCfg)
            assert(strcmp(hSI.acqState,'idle'), 'Stop acquisition before registering NeuroSeq callbacks.');
            hSI.hUserFunctions.userFunctionsCfg = cfg;
        end
    case 'preview'
        assert(strcmp(hSI.acqState, 'idle'), 'ScanImage is busy.');
        % Idle without a terminal event: put back settings before forgetting the run.
        run = restore_settings(hSI, run, true);
        run = idle_run();
        hSI.startFocus(); ownedFocus = true;
    case 'record'
        assert(strcmp(hSI.acqState, 'idle'), 'Stop the current acquisition before recording.');
        % Otherwise a run that never reported completion would leave its
        % NeuroSeq settings to be captured below as the user's originals.
        run = restore_settings(hSI, run, true);
        % numSlices keeps the configured stack size while stacks are disabled;
        % actualNumSlices is what Grab will acquire.
        if isprop(hSI.hStackManager, 'actualNumSlices')
            slices = hSI.hStackManager.actualNumSlices;
        else
            slices = hSI.hStackManager.numSlices;
        end
        assert(slices == 1, 'NeuroSeq:MultiplePlanes', 'Only single-plane acquisition is supported.');
        assert(~hSI.hRoiManager.mroiEnable, 'Multi-ROI acquisition is not supported.');
        assert(~hSI.hFastZ.enable, 'FastZ must be disabled.');
        channels = [];
        if isfield(args,'channels') && ~isempty(args.channels)
            % Several channels in one recording; ScanImage's own selection
            % is restored afterwards with the other settings.
            channels = unique(double(args.channels(:)'));
            available = hSI.hChannels.channelsAvailable;
            assert(all(channels >= 1 & channels <= available & channels == fix(channels)), ...
                'NeuroSeq:InvalidChannels', 'Saved channels must be between 1 and %d.', available);
        end
        assert(~isempty(channels) || ~isempty(hSI.hChannels.channelSave), ...
            'Select at least one saved channel.');
        assert(args.frames >= 1 && args.frames == fix(args.frames), 'Invalid frame count.');
        assert(args.counter >= 1 && args.counter == fix(args.counter), 'Invalid recording number.');
        % One TIFF per recording, named <stem>_<number>.tif, directly in the
        % recording folder (issues #628, #651).
        target = fullfile(args.directory, sprintf('%s_%05d.tif', args.stem, args.counter));
        assert(~exist(target, 'file'), 'NeuroSeq:FileExists', 'Recording %s already exists.', target);
        if ~exist(args.directory, 'dir')
            [ok, msg] = mkdir(args.directory); assert(ok, msg);
        end
        % Retain settings until completion/abort and restore only once idle.
        old = {hSI.hScan2D.logFilePath, hSI.hScan2D.logFileStem, hSI.hScan2D.logFileCounter, ...
            hSI.extTrigEnable, hSI.hStackManager.framesPerSlice, hSI.hChannels.loggingEnable, ...
            optional_prop(hSI.hScan2D, 'logAverageFactor'), optional_prop(hSI, 'acqsPerLoop'), ...
            hSI.hChannels.channelSave, optional_prop(hSI.hScan2D, 'logFramesPerFile')};
        run = struct('id',args.run_id, 'state','starting', 'directory',args.directory, ...
            'stem',args.stem, 'counter',args.counter, 'old',{old});
        try
            hSI.hScan2D.logFilePath = args.directory;
            hSI.hScan2D.logFileStem = args.stem;
            hSI.hScan2D.logFileCounter = args.counter;
            if ~isempty(old{10}), hSI.hScan2D.logFramesPerFile = Inf; end
            hSI.extTrigEnable = logical(args.external_trigger);
            hSI.hStackManager.framesPerSlice = args.frames;
            hSI.hChannels.loggingEnable = true;
            if ~isempty(channels), hSI.hChannels.channelSave = channels; end
            % Raw frames, one acquisition: averaging would drop frames and a
            % multi-acquisition loop could delay acqModeDone.
            if ~isempty(old{7}), hSI.hScan2D.logAverageFactor = 1; end
            if ~isempty(old{8}), hSI.acqsPerLoop = 1; end
            run.settings = recording_settings(hSI);
            hSI.startGrab();
            if strcmp(run.state,'starting')
                % Start returned, but this is NOT confirmation of hardware readiness.
                run.state = 'waiting_for_first_frame';
            end
        catch err
            run.state = 'failed';
            run = restore_settings(hSI, run);
            rethrow(err)
        end
    case 'abort'
        assert(ownedFocus || any(strcmp(run.state, {'starting','waiting_for_first_frame','recording'})), ...
            'NeuroSeq does not own the active acquisition.');
        hSI.abort(); ownedFocus = false;
    case 'disconnect'
        assert(strcmp(hSI.acqState,'idle'), 'Abort acquisition before disconnecting.');
        run = restore_settings(hSI, run, true);
        cfg = hSI.hUserFunctions.userFunctionsCfg;
        if isstruct(cfg) && ~isempty(cfg)
            cfg(strcmp({cfg.UserFcnName}, 'neuroseq_scanimage_event')) = [];
            hSI.hUserFunctions.userFunctionsCfg = cfg;
        end
        reply = '{}'; return
    case 'status'
        % Read-only. Never start an acquisition in response to polling.
    otherwise
        error('Unknown NeuroSeq operation');
end
run = restore_settings(hSI, run);
publicRun = run;
if isfield(publicRun,'old'), publicRun = rmfield(publicRun,'old'); end
out = struct('acq_state',hSI.acqState, 'run',publicRun, 'fps',hSI.hRoiManager.scanFrameRate, ...
    'matlab_version',version, 'channels',hSI.hChannels.channelDisplay, 'files',{{}});
% For the ScanImage tab's channel choice.
out.channels_available = hSI.hChannels.channelsAvailable;
out.saved_channels = hSI.hChannels.channelSave;
out.can_abort = ownedFocus || any(strcmp(run.state, {'starting','waiting_for_first_frame','recording'}));
out.settings = recording_settings(hSI);
if isfield(run,'settings'), out.settings = run.settings; end
if isprop(hSI,'VERSION_MAJOR')
    out.scanimage_version = char(string(hSI.VERSION_MAJOR));
    if isprop(hSI,'VERSION_MINOR')
        out.scanimage_version = [out.scanimage_version '.' char(string(hSI.VERSION_MINOR))];
    end
    if isprop(hSI,'VERSION_UPDATE')
        out.scanimage_version = [out.scanimage_version '.' char(string(hSI.VERSION_UPDATE))];
    end
end
if ~isempty(run.directory) && strcmp(hSI.acqState,'idle')
    files = dir(fullfile(run.directory, sprintf('%s_%05d*.tif', run.stem, run.counter)));
    out.files = arrayfun(@(f) fullfile(f.folder,f.name), files, 'UniformOutput',false);
end
out.elapsed_ms = 1000 * toc(started);
reply = jsonencode(out);
end

function value = optional_prop(obj, name)
% Empty when this ScanImage release lacks the property; restored only if set.
value = [];
if isprop(obj, name), value = obj.(name); end
end

function run = restore_settings(hSI, run, force)
if nargin < 3, force = false; end
% Some releases dispatch the terminal callback before acqState becomes idle.
% In that case the next status/command retries, retaining the saved settings.
if isfield(run,'old') && strcmp(hSI.acqState,'idle') && ...
        (force || any(strcmp(run.state, {'completed','aborted','failed'})))
    old = run.old;
    hSI.hScan2D.logFilePath = old{1}; hSI.hScan2D.logFileStem = old{2};
    hSI.hScan2D.logFileCounter = old{3}; hSI.extTrigEnable = old{4};
    hSI.hStackManager.framesPerSlice = old{5}; hSI.hChannels.loggingEnable = old{6};
    if ~isempty(old{7}), hSI.hScan2D.logAverageFactor = old{7}; end
    if ~isempty(old{8}), hSI.acqsPerLoop = old{8}; end
    if numel(old) >= 9, hSI.hChannels.channelSave = old{9}; end
    if numel(old) >= 10 && ~isempty(old{10}), hSI.hScan2D.logFramesPerFile = old{10}; end
    run = rmfield(run,'old');
end
end

function settings = recording_settings(hSI)
settings = struct('saved_channels',hSI.hChannels.channelSave, ...
    'frames_per_slice',hSI.hStackManager.framesPerSlice, ...
    'external_trigger',hSI.extTrigEnable, 'logging',hSI.hChannels.loggingEnable, ...
    'directory',hSI.hScan2D.logFilePath, 'stem',hSI.hScan2D.logFileStem);
end

function run = idle_run()
run = struct('id', '', 'state', 'idle', 'directory', '', 'stem', '', 'counter', 0);
end
