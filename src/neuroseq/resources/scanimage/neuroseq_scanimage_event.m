function neuroseq_scanimage_event(~, evt, varargin)
% Forward lifecycle events; no frame processing or IO on the acquisition path.
neuroseq_scanimage('event', jsonencode(struct('name', evt.EventName)));
end
