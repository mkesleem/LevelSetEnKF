% Copyright (c) 2026 Michael Sleeman
% SPDX-License-Identifier: MIT


function ens_dirs = find_ensemble_dirs(base_dir, case_name)
% Find ensemble-member directories for a case.
%
% Input:
% - base_dir: parent directory of the ensemble cases
% - case_name: ensemble directory prefix
%
% Output:
% - ens_dirs: directory-structure array for folders matching case_name_*
%

ens_dirs = dir(fullfile(base_dir, case_name + "_*"));
ens_dirs = ens_dirs([ens_dirs.isdir]);
if isempty(ens_dirs)
    error("No ensemble folders found in %s", base_dir);
end


end

