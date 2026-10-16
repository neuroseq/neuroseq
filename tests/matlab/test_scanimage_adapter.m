function tests = test_scanimage_adapter
% Run with runtests('tests/matlab') in a separate MATLAB session without a rig.
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
assert(~evalin('base','exist(''hSI'',''var'')'), 'Run in a separate MATLAB session without hSI.');
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root,'src','neuroseq','resources','scanimage'));
addpath(fullfile(root,'tests','matlab'));
end

function setup(testCase)
clear neuroseq_scanimage
rig = ScanImageTestRig();
assignin('base','hSI',rig);
testCase.TestData.rig = rig;
testCase.TestData.directory = tempname;
neuroseq_scanimage('probe','{}');
end

function teardown(testCase)
evalin('base','clear hSI');
clear neuroseq_scanimage
if exist(testCase.TestData.directory,'dir'), rmdir(testCase.TestData.directory,'s'); end
end

function args = request(testCase)
args = struct('run_id','run', 'directory',testCase.TestData.directory, 'frames',100, 'external_trigger',true);
end

function old = settings(rig)
old = {rig.hScan2D.logFilePath,rig.hScan2D.logFileStem,rig.hScan2D.logFileCounter, ...
    rig.extTrigEnable,rig.hStackManager.framesPerSlice,rig.hChannels.loggingEnable, ...
    rig.hScan2D.logAverageFactor,rig.acqsPerLoop,rig.hChannels.channelSave};
end

function testRestoreOnCompletion(testCase)
rig = testCase.TestData.rig; old = settings(rig);
neuroseq_scanimage('record',jsonencode(request(testCase)));
verifyNotEqual(testCase,settings(rig),old);
rig.complete(false);
verifyEqual(testCase,settings(rig),old);
status = jsondecode(neuroseq_scanimage('status','{}'));
verifyEqual(testCase,status.run.state,'completed');
verifyEqual(testCase,status.settings.frames_per_slice,100);
verifyEqual(testCase,status.settings.directory,testCase.TestData.directory);
verifyFalse(testCase,isfield(status.run,'old'));
end

function testRestoreAfterEarlyTerminalEvent(testCase)
rig = testCase.TestData.rig; old = settings(rig);
neuroseq_scanimage('record',jsonencode(request(testCase)));
rig.complete(true);
neuroseq_scanimage('status','{}');
verifyEqual(testCase,settings(rig),old);
end

function testRestoreOnAbort(testCase)
rig = testCase.TestData.rig; old = settings(rig);
neuroseq_scanimage('record',jsonencode(request(testCase)));
status = jsondecode(neuroseq_scanimage('abort','{}'));
verifyEqual(testCase,settings(rig),old);
verifyEqual(testCase,status.run.state,'aborted');
end

function testRestoreOnDisconnect(testCase)
rig = testCase.TestData.rig; old = settings(rig);
neuroseq_scanimage('record',jsonencode(request(testCase)));
rig.complete(true);
neuroseq_scanimage('disconnect','{}');
verifyEqual(testCase,settings(rig),old);
verifyEqual(testCase,numel(rig.hUserFunctions.userFunctionsCfg),1);
end

function testRestoreOnStartFailure(testCase)
rig = testCase.TestData.rig; old = settings(rig); rig.failStart = true;
verifyError(testCase,@()neuroseq_scanimage('record',jsonencode(request(testCase))), 'Test:StartFailed');
verifyEqual(testCase,settings(rig),old);
end

function testCallbackRegistrationSurvivesClear(testCase)
rig = testCase.TestData.rig;
cfg = rig.hUserFunctions.userFunctionsCfg;
verifyEqual(testCase,numel(cfg),5);
verifyEqual(testCase,cfg(1).ExtraField,'preserve me');
clear neuroseq_scanimage
neuroseq_scanimage('probe','{}');
verifyEqual(testCase,rig.hUserFunctions.userFunctionsCfg,cfg);
% Duplicate stale adapter registrations are reconciled as well.
rig.hUserFunctions.userFunctionsCfg(end+1) = cfg(2);
neuroseq_scanimage('probe','{}');
verifyEqual(testCase,numel(rig.hUserFunctions.userFunctionsCfg),5);
end

function testEmptyCallbackTable(testCase)
rig = testCase.TestData.rig;
rig.hUserFunctions.userFunctionsCfg = struct([]);
neuroseq_scanimage('probe','{}');
verifyEqual(testCase,numel(rig.hUserFunctions.userFunctionsCfg),4);
end

function testRecordsRawSingleAcquisitionThenRestoresUserValues(testCase)
% As on the rig: logging averages 10 frames and a loop has 2 acquisitions.
rig = testCase.TestData.rig;
verifyEqual(testCase,[rig.hScan2D.logAverageFactor rig.acqsPerLoop],[10 2]);
neuroseq_scanimage('record',jsonencode(request(testCase)));
verifyEqual(testCase,[rig.hScan2D.logAverageFactor rig.acqsPerLoop],[1 1]);
rig.complete(false);
verifyEqual(testCase,[rig.hScan2D.logAverageFactor rig.acqsPerLoop],[10 2]);
end

function testRecordsChosenChannelsThenRestoresSelection(testCase)
rig = testCase.TestData.rig; rig.hChannels.channelSave = 1;
status = jsondecode(neuroseq_scanimage('status','{}'));
verifyEqual(testCase,status.channels_available,4);
verifyEqual(testCase,status.saved_channels,1);
req = request(testCase); req.channels = [3 1];
neuroseq_scanimage('record',jsonencode(req));
verifyEqual(testCase,rig.hChannels.channelSave,[1 3]);
status = jsondecode(neuroseq_scanimage('status','{}'));
verifyEqual(testCase,status.settings.saved_channels,[1;3]);
rig.complete(false);
verifyEqual(testCase,rig.hChannels.channelSave,1);
end

function testInvalidChannelsRejected(testCase)
rig = testCase.TestData.rig; old = settings(rig);
req = request(testCase); req.channels = [1 5];
verifyError(testCase,@()neuroseq_scanimage('record',jsonencode(req)), 'NeuroSeq:InvalidChannels');
verifyEqual(testCase,settings(rig),old);
end

function testDisabledStackKeepsSlicesButRecordsOnePlane(testCase)
% As on the rig: a 25-slice stack is configured but stacks are disabled.
rig = testCase.TestData.rig; rig.hStackManager.numSlices = 25;
neuroseq_scanimage('record',jsonencode(request(testCase)));
verifyEqual(testCase,rig.acqState,'grab');
end

function testActiveStackRejected(testCase)
rig = testCase.TestData.rig; old = settings(rig);
rig.hStackManager.numSlices = 25; rig.hStackManager.actualNumSlices = 25;
verifyError(testCase,@()neuroseq_scanimage('record',jsonencode(request(testCase))), 'NeuroSeq:MultiplePlanes');
verifyEqual(testCase,settings(rig),old);
end

function testPreviewOnlyOnRequestAtFullResolution(testCase)
neuroseq_scanimage('preview','{}');
status = jsondecode(neuroseq_scanimage('status','{}'));
verifyFalse(testCase,isfield(status,'frame'));
verifyGreaterThanOrEqual(testCase,status.elapsed_ms,0);
status = jsondecode(neuroseq_scanimage('status','{"channel":1}'));
verifyEqual(testCase,status.frame.shape,[512;512]);
verifyEqual(testCase,status.frame.dtype,'int16');
verifyEqual(testCase,status.frame.lut,[-88;436]);
verifyEqual(testCase,status.frame.display_average,10);
raw = matlab.net.base64decode(status.frame.data);
pixels = reshape(typecast(raw,'int16'),512,512);
rig = testCase.TestData.rig;
verifyEqual(testCase,pixels,rig.hDisplay.lastFrame{1});
end

function testPreviewSurvivesFailingChannelRead(testCase)
% ScanImage 2023.1.0 can raise from Display.get.lastFrameChannels.
rig = testCase.TestData.rig;
rig.hDisplay.failLastFrameChannels = true;
neuroseq_scanimage('preview','{}');
status = jsondecode(neuroseq_scanimage('status','{"channel":1}'));
verifyEqual(testCase,status.frame.shape,[512;512]);
verifyFalse(testCase,isfield(status,'preview_problem'));
end

function testPreviewReportsUnreadableFrame(testCase)
rig = testCase.TestData.rig;
rig.hDisplay.failLastFrameChannels = true;
rig.hDisplay.failLastFrame = true;
neuroseq_scanimage('preview','{}');
status = jsondecode(neuroseq_scanimage('status','{"channel":1}'));
verifyFalse(testCase,isfield(status,'frame'));
verifySubstring(testCase,status.preview_problem,'Brace indexing');
verifyEqual(testCase,status.acq_state,'focus');
end

function testPreviewOfUndisplayedChannel(testCase)
neuroseq_scanimage('preview','{}');
status = jsondecode(neuroseq_scanimage('status','{"channel":3}'));
verifyFalse(testCase,isfield(status,'frame'));
verifySubstring(testCase,status.preview_problem,'Channel 3 is not displayed');
end

function testLargePreviewIsBlockAveraged(testCase)
rig = testCase.TestData.rig;
rig.hDisplay.lastFrame{1} = reshape(single(1:1024^2),1024,1024);
neuroseq_scanimage('preview','{}');
status = jsondecode(neuroseq_scanimage('status','{"channel":1}'));
verifyEqual(testCase,status.frame.shape,[512;512]);
raw = matlab.net.base64decode(status.frame.data);
pixels = reshape(typecast(raw,'single'),512,512);
source = rig.hDisplay.lastFrame{1};
verifyEqual(testCase,pixels(1,1),mean(source(1:2,1:2),'all'),'AbsTol',1e-3);
verifyEqual(testCase,pixels(3,5),mean(source(5:6,9:10),'all'),'AbsTol',1e-3);
end

function testReconnectWhileRecordingDoesNotChangeCallbacks(testCase)
rig = testCase.TestData.rig;
neuroseq_scanimage('record',jsonencode(request(testCase)));
cfg = rig.hUserFunctions.userFunctionsCfg;
status = jsondecode(neuroseq_scanimage('probe','{}'));
verifyEqual(testCase,status.run.state,'recording');
verifyEqual(testCase,rig.hUserFunctions.userFunctionsCfg,cfg);
end

function testDisconnectRestoresAfterMissingEvent(testCase)
rig = testCase.TestData.rig; old = settings(rig);
neuroseq_scanimage('record',jsonencode(request(testCase)));
rig.acqState = 'idle';
neuroseq_scanimage('disconnect','{}');
verifyEqual(testCase,settings(rig),old);
end

function testEmptyNumericCallbackTable(testCase)
rig = testCase.TestData.rig;
rig.hUserFunctions.userFunctionsCfg = [];
neuroseq_scanimage('probe','{}');
verifyEqual(testCase,numel(rig.hUserFunctions.userFunctionsCfg),4);
end

function testNextRecordAfterMissingEventKeepsOriginalSettings(testCase)
rig = testCase.TestData.rig; old = settings(rig);
neuroseq_scanimage('record',jsonencode(request(testCase)));
rig.acqState = 'idle';
second = request(testCase); second.run_id = 'second'; second.directory = [testCase.TestData.directory '_2'];
testCase.addTeardown(@() rmdir(second.directory,'s'));
neuroseq_scanimage('record',jsonencode(second));
rig.complete(false);
neuroseq_scanimage('status','{}');
verifyEqual(testCase,settings(rig),old);
end

function testPreviewAfterMissingEventRestoresSettings(testCase)
rig = testCase.TestData.rig; old = settings(rig);
neuroseq_scanimage('record',jsonencode(request(testCase)));
rig.acqState = 'idle';
neuroseq_scanimage('preview','{}');
verifyEqual(testCase,settings(rig),old);
end
