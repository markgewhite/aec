classdef UCRDataset < ModelDataset
    % Subclass for retrieving a specified UCR dataset
    % NB. The time domain (sampling interval) is not specified.

    properties
        UCRDatasetID        % id number
    end

    methods

        function self = UCRDataset( set, args, superArgs )
            % Load the countermovement jump GRF dataset
            arguments
                set                 char ...
                    {mustBeMember( set, ...
                           {'Training', 'Testing', 'Combined'} )}
                args.SetID          double {mustBeInteger, ...
                     mustBeInRange( args.SetID, 1, 128 )} = 33
                args.PaddingLength  double = 0
                args.Lambda         double = []
                superArgs.?ModelDataset
            end

            [ XRaw, Y, name ] = loadData( args.SetID, set );

            if args.PaddingLength==0
                pad.Length = max(cellfun( @length, XRaw ));
            else
                pad.Length = args.PaddingLength;
            end
            tSpan = 1:pad.Length;

            pad.Longest = false;
            pad.Location = 'Left';
            pad.Value = 1;
            pad.Same = true;
            pad.Anchoring = 'None';
         
            % process the data and complete the initialization
            superArgsCell = namedargs2cell( superArgs );
            nClasses = length(unique(Y));
            labels = strings( nClasses, 1 );
            for i = 1:nClasses
                labels(i) = strcat( "Class ", char(64+i) );
            end

            self = self@ModelDataset( XRaw, Y, tSpan, ...
                            superArgsCell{:}, ...
                            padding = pad, ...
                            lambda = args.Lambda, ...
                            datasetName = name, ...
                            channelLabels = "X (no units)", ...
                            timeLabel = "Time Domain", ...
                            classLabels = labels, ...
                            channelLimits = [] );

            self.UCRDatasetID = args.SetID;


        end

    end

end


function [ X, Y, name ] = loadData( id, set )
    % Load the specified dataset 
    arguments
        id              double {mustBeInteger, mustBePositive}
        set             char {mustBeMember( set, ...
                               {'Training', 'Testing', 'Combined'} )}
    end

    path = fileparts( which('UCRDataset.m') );
    path = [path '/../../data/ucr'];

    % read the master reference table
    refTable = readtable( fullfile(path, 'DataSummaryExpanded_v03.xlsx'), ...
                          NumHeaderLines = 0, ...
                          ReadVariableNames = true, ...
                          VariableNamingRule = 'Preserve' );

    % find the identified dataset's reference
    row = find( refTable.ID==id, 1 );

    if refTable.Include{row}=='N'
        warning('Reference spreadsheet indicates dataset should not be included.');
    end

    name = refTable.Name{row};
    path = [path '/' name];

    % load the data - TRAIN or TEST or both
    switch set
        case 'Training'
            [X, Y] = readFile( path, [name '_TRAIN.txt'] );
        case 'Testing'
            [X, Y] = readFile( path, [name '_TEST.txt'] );
        case 'Combined'
            [X1, Y1] = readFile( path, [name '_TRAIN.txt'] );
            [X2, Y2] = readFile( path, [name '_TEST.txt'] );
            X = [X1; X2];
            Y = [Y1; Y2];
    end
    
    % convert to cell array
    X = num2cell( X', 1 )';

    % convert labels to a sequence 1, 2, 3 ...
    Y = double(categorical(Y));

    % check the length of each observation, trimming if nans
    for i = 1:length(X)
        lmax = size(X{i},1);
        for d = 1:size(X{i},2)
            l = find(isnan(X{i}(:,d)),1)-1;
            if ~isempty(l)
                lmax = min( l, lmax );
            end
        end
        X{i} = X{i}(1:lmax,:);
    end
    
end


function [X, Y] = readFile( path, filename )

    raw = readtable( fullfile(path, filename) );
    X = table2array( raw(:,2:end) );
    Y = table2array( raw(:,1) );

end




