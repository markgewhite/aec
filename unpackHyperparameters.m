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

for i = 1:size( hp, 2 )

    hpName = hp.Properties.VariableNames{i};

    if iscategorical( hp.(hpName) )
        switch hp.(hpName)
            case 'false'
                value = false;
            case 'true'
                value = true;
            otherwise
                value = char(hp.(hpName));
        end
    else
        value = hp.(hpName);
    end

    if contains( hpName, '__' )
        optGroup = extractBefore( hpName, '__' );
        optVar = extractAfter( hpName, '__' );
        setup.(optGroup).(optVar) = value;
    else
        setup.(hpName) = value;
    end

end
    
end