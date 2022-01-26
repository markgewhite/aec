% ************************************************************************
% Function: getJumpGRFData
% Purpose:  Load and extract ground reaction force data from jumps
%
% Parameters:
%
% Output:
%       curveSet: extracted VGRF data structure
%       IDSet: associated identifiers
%       typeSet: jump type
%
% ************************************************************************


function [ X, A, S ] =  getJumpGRFData

% load data from file
load( 'data/jumpGRFData.mat', ...
      'grf', 'bwall', 'sDataID', 'sJumpID', ...
      'jumpOrder', 'nJumpsPerSubject' );

% exclude jumps from subjects in the second data collection
subjectExclusions = find( ismember( sDataID, ...
            [ 14, 39, 68, 86, 87, 11, 22, 28, 40, 43, 82, 88, 95, 97, ...
              100, 121, 156, 163, 196 ] ) );

% specific jumps that should be excluded
jumpExclusions = [3703 3113 2107 2116 0503 0507 6010 1109];

nSubjects = length( sDataID );

nTotal = sum( nJumpsPerSubject );

X = cell( nTotal, 1 );
A = zeros( nTotal, 1 );
S = zeros( nTotal, 1 );
k = 0;

for i = 1:nSubjects
    for j = 1:nJumpsPerSubject(i)
        
        jump = jumpOrder( sJumpID==sDataID(i), j );
        jumpID = sDataID(i)*100+j;
        
        if jump{1}(1) == 'V' ...
                && ~ismember( i, subjectExclusions ) ...
                && ~ismember( jumpID, jumpExclusions )
            
            k = k+1;
            X{ k } = grf.raw{i,j}( 1:grf.takeoff(i,j) ) ...
                               / bwall(i,j);

            S( k ) = i;
            A( k ) = (length(jump{1}) == 2);
        end
        
    end
end

X = X(1:k);
S = S(1:k);
A = A(1:k);

end