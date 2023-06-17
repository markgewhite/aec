classdef JumpGRFDataset < ModelDataset
    % Subclass for loading the countermovement jump VGRF dataset

    properties
        SubjectID       % identifying participants
        OutcomeVar      % outcome variable
    end

    methods

        function self = JumpGRFDataset( set, superArgs, args )
            % Load the countermovement jump GRF dataset
            arguments
                 set                char ...
                    {mustBeMember( set, ...
                            {'Training', 'Testing', 'Combined'} )}
                superArgs.?ModelDataset
                args.OutcomeVar     char ...
                    {mustBeMember( args.OutcomeVar, ...
                           {'JumpType', 'JumpHeightWD', ...
                           'JumpHeightTOV', 'PeakPower'} )} = 'JumpType'
                args.PaddingLength  double = 0
                args.Lambda         double = []
            end

            [ XRaw, Y, S, labels ] = JumpGRFDataset.load( set, args.OutcomeVar );

            % setup padding
            if args.PaddingLength==0
                pad.Length = 1501;
            else
                pad.Length = args.PaddingLength;
            end
            pad.Longest = false;
            pad.Location = 'Left';
            pad.Value = 1;
            pad.Same = false;
            pad.Anchoring = 'Right';

            tSpan= -pad.Length+1:0;
         
            % process the data and complete the initialization
            superArgsCell = namedargs2cell( superArgs );

            self = self@ModelDataset( XRaw, Y, tSpan, ...
                            superArgsCell{:}, ...
                            padding = pad, ...
                            lambda = args.Lambda, ...
                            datasetName = "Jumps VGRF Data", ...
                            channelLabels = "VGRF (BW)", ...
                            timeLabel = "Time (ms)", ...
                            classLabels = labels, ...
                            channelLimits = [0 3] );

            self.SubjectID = S;
            self.OutcomeVar = args.OutcomeVar;

        end


        function unit = getPartitioningUnit( self )
            % Provide the SubjectID for partitioning (overriding parent)
            arguments
                self    JumpGRFDataset
            end

            unit = self.SubjectID;

        end

    end

    methods (Static)

        function [ X, Y, S, L ] = load( set, outcomeVar )

            path = fileparts( which('JumpGRFDataset.m') );
            path = [path '/../../data/jumps'];
            
            % load the data - training or testing or both
            switch set

                case 'Training'
                    load( fullfile( path, 'jumpGRFData-Training2.mat' ), ...
                              'X', 'Y', 'C', 'S' );

                case 'Testing'
                    load( fullfile( path, 'jumpGRFData-Testing2.mat' ), ...
                              'X', 'Y', 'C', 'S' );

                case 'Combined'
                    load( fullfile( path, 'jumpGRFData-Testing2.mat' ), ...
                              'X', 'Y', 'C', 'S' );
                    X1 = X;
                    Y1 = Y;
                    C1 = C;
                    S1 = S;

                    load( fullfile( path, 'jumpGRFData-Training2.mat' ), ...
                              'X', 'Y', 'C', 'S' );

                    for i = 1:2
                        X{i} = [X{i}; X1{i}];
                        Y.JHwd{i} = [Y.JHwd{i} Y1.JHwd{i}];
                        Y.JHtov{i} = [Y.JHtov{i} Y1.JHtov{i}];
                        Y.PP{i} = [Y.PP{i} Y1.PP{i}];
                        S{i} = [S{i}; S1{i}];
                    end
                    C = [C; C1];

           end
                       
            switch outcomeVar
                case 'JumpType'
                    Y = C;
                    X = X{2};
                    S = S{2};
                    L = ["Without Arms", "With Arms"];
                case 'JumpHeightWD'
                    Y = Y.JHwd{1}';
                    X = X{1};
                    S = S{1};
                    L = "Jump Height (WD)";
                case 'JumpHeightTOV'
                    Y = Y.JHtov{1}';
                    X = X{1};
                    S = S{1};
                    L = "Jump Height (TOV)";
                case 'PeakPower'
                    Y = Y.PP{1}';
                    X = X{1};
                    S = S{1};
                    L = "Peak Power";
            end
            
       end

   end


end


