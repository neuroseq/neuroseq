function reply = neuroseq_scanimage(operation, payload)
% NeuroSeq adapter for ScanImage Premium 2023.1.0, single-plane raster imaging.
% The installed API is checked before controls become available.
persistent run ownedFocus
if isempty(ownedFocus), ownedFocus = false; end
if isempty(run), run = struct('id', '', 'state', 'idle', 'directory', ''); end
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
        assert(isprop(hSI, 'hDisplay') && isprop(hSI.hDisplay, 'lastFrame'), ...
            'This adapter requires the ScanImage 2023 hDisplay API.');
        required = {'logFilePath','logFileStem','logFileCounter','trigAcqEnable'};
        for k=1:numel(required)
            assert(isprop(hSI.hScan2D, required{k}), 'Missing ScanImage property: %s', required{k});
        end
        assert(isprop(hSI.hStackManager, 'framesPerSlice'), 'Missing framesPerSlice');
        assert(isprop(hSI.hChannels, 'loggingEnable'), 'Missing loggingEnable');
        validate_single_acquisition(hSI);
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
        run = struct('id', '', 'state', 'idle', 'directory', '');
        hSI.startFocus(); ownedFocus = true;
    case 'record'
        assert(strcmp(hSI.acqState, 'idle'), 'Stop the current acquisition before recording.');
        % Otherwise a run that never reported completion would leave its
        % NeuroSeq settings to be captured below as the user's originals.
        run = restore_settings(hSI, run, true);
        validate_single_acquisition(hSI);
        assert(hSI.hStackManager.numSlices == 1, 'Only single-plane acquisition is supported.');
        assert(~hSI.hRoiManager.mroiEnable, 'Multi-ROI acquisition is not supported.');
        assert(~hSI.hFastZ.enable, 'FastZ must be disabled.');
        assert(isprop(hSI.hScan2D, 'logAverageFactor') && hSI.hScan2D.logAverageFactor == 1, ...
            'Set ScanImage logging average factor to 1 to preserve raw frames.');
        assert(~isempty(hSI.hChannels.channelSave), 'Select at least one saved channel in ScanImage.');
        assert(args.frames >= 1 && args.frames == fix(args.frames), 'Invalid frame count.');
        assert(~exist(args.directory, 'dir'), 'Run directory already exists.');
        [ok, msg] = mkdir(args.directory); assert(ok, msg);
        % Retain settings until completion/abort and restore only once idle.
        old = {hSI.hScan2D.logFilePath, hSI.hScan2D.logFileStem, hSI.hScan2D.logFileCounter, ...
            hSI.hScan2D.trigAcqEnable, hSI.hStackManager.framesPerSlice, hSI.hChannels.loggingEnable};
        run = struct('id',args.run_id, 'state','starting', 'directory',args.directory, 'old',{old});
        try
            hSI.hScan2D.logFilePath = args.directory;
            hSI.hScan2D.logFileStem = 'acquisition';
            hSI.hScan2D.logFileCounter = 1;
            hSI.hScan2D.trigAcqEnable = logical(args.external_trigger);
            hSI.hStackManager.framesPerSlice = args.frames;
            hSI.hChannels.loggingEnable = true;
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
    files = dir(fullfile(run.directory, '*.tif'));
    out.files = arrayfun(@(f) fullfile(f.folder,f.name), files, 'UniformOutput',false);
end
if strcmp(operation,'status') && isfield(args,'channel') && ~strcmp(hSI.acqState,'idle')
    channels = hSI.hChannels.channelDisplay;
    if isprop(hSI.hDisplay,'lastFrameChannels'), channels = hSI.hDisplay.lastFrameChannels; end
    idx = find(channels == args.channel, 1);
    images = hSI.hDisplay.lastFrame;
    if ~isempty(idx) && numel(images) >= idx && ~isempty(images{idx}) && ismatrix(images{idx})
        source = images{idx};
        stride = max(1, ceil(max(size(source)) / 256));
        img = single(source(1:stride:end, 1:stride:end));
        out.frame = struct('shape',size(img), 'dtype','float32', 'order','F', ...
            'data',matlab.net.base64encode(typecast(img(:),'uint8')));
        if isprop(hSI.hDisplay,'lastFrameNumber'), out.frame.number = hSI.hDisplay.lastFrameNumber; end
        if isprop(hSI.hDisplay,'lastFrameTimestamp'), out.frame.timestamp = hSI.hDisplay.lastFrameTimestamp; end
    end
end
reply = jsonencode(out);
end

function validate_single_acquisition(hSI)
% startGrab is used, never startLoop; reject ambiguous multi-acquisition setups.
if isprop(hSI,'acqsPerLoop')
    assert(hSI.acqsPerLoop == 1, 'NeuroSeq:MultipleAcquisitions', 'Set ScanImage acquisitions per loop to 1.');
end
end

function run = restore_settings(hSI, run, force)
if nargin < 3, force = false; end
% Some releases dispatch the terminal callback before acqState becomes idle.
% In that case the next status/command retries, retaining the saved settings.
if isfield(run,'old') && strcmp(hSI.acqState,'idle') && ...
        (force || any(strcmp(run.state, {'completed','aborted','failed'})))
    old = run.old;
    hSI.hScan2D.logFilePath = old{1}; hSI.hScan2D.logFileStem = old{2};
    hSI.hScan2D.logFileCounter = old{3}; hSI.hScan2D.trigAcqEnable = old{4};
    hSI.hStackManager.framesPerSlice = old{5}; hSI.hChannels.loggingEnable = old{6};
    run = rmfield(run,'old');
end
end

function settings = recording_settings(hSI)
settings = struct('saved_channels',hSI.hChannels.channelSave, ...
    'frames_per_slice',hSI.hStackManager.framesPerSlice, ...
    'external_trigger',hSI.hScan2D.trigAcqEnable, 'logging',hSI.hChannels.loggingEnable, ...
    'directory',hSI.hScan2D.logFilePath, 'stem',hSI.hScan2D.logFileStem);
end
