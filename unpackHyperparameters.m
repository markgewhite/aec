% ************************************************************************
% Function: unpackHyperparameters
% Purpose:  Update the setup configuration structure using the
%           hyperparameters vector
%
%
% Parameters:
%           setup: current configuration structure
%           hp: hyperparameters
%
% Output:
%           setup: updated configuration structure
%
% ************************************************************************


function setup = unpackHyperparameters( setup, hp )

if isempty( hp ) 
    return
end

for i = size( hp, 2 )

    hpName = hp.Properties.VariableNames{i};

    if contains( hpName, '__' )
        optGroup = extractBefore( hpName, '__' );
        optVar = extractAfter( hpName, '__' );
        setup.(optGroup).(optVar) = hp.(hpName);
    else
        setup.(optVar) = hp.(hpName);
    end

end
    
end