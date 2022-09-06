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
                                   {'Training', 'Testing'} )}
                args.SetID          double {mustBeInteger, ...
                     mustBeInRange( args.SetID, 1, 128 )} = 33
                args.PaddingLength  double = 0
                args.Lambda         double = 1E-9
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
       
            % setup fda
            paramsFd.BasisOrder = 4;
            paramsFd.PenaltyOrder = 2;
            paramsFd.Lambda = args.Lambda;
         
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
                            fda = paramsFd, ...
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
                                   {'Training', 'Testing'} )}
    end

    path = fileparts( which('UCRDataset.m') );
    path = [path '/../../data/ucr'];

    % read the master reference table
    refTable = readtable( fullfile(path, 'DataSummaryExpanded_v03.xlsx'), ...
                           VariableNamingRule = 'Preserve' );

    % find the identified dataset's reference
    row = find( refTable.ID==id, 1 );
    name = refTable.Name{row};
    path = [path '/' name];

    % load the data - either TRAIN or TEST
    switch set
        case 'Training'
            suffix = '_TRAIN.txt';
        case 'Testing'
            suffix = '_TEST.txt';
    end
    filename = [name suffix];
    raw = readtable( fullfile(path, filename) );
    X = table2array( raw(:,2:end) );
    Y = table2array( raw(:,1) );
    
    % convert to cell array
    X = num2cell( X', 1 )';

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




