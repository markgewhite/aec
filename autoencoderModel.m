% ************************************************************************
% Class: autoencoderModel
%
% Subclass defining the framework for an autoencoder model
%
% ************************************************************************

classdef autoencoderModel < representationModel

    properties
        XOutDim        % output dimension (may differ from XDim)
        nets           % networks defined in this model (structure)
        netNames       % names of the networks (for convenience)
        nNets          % number of networks
        isVAE          % flag indicating if variational autoencoder
        lossFcns       % array of loss functions
        lossFcnNames   % names of the loss functions
        lossFcnWeights % weights to be applied to the loss function
        lossFcnTbl     % convenient table summarising loss function details
        nLoss          % number of computed losses
        isInitialized  % flag indicating if fully initialized
        hasSeqInput    % supports variable-length input

        trainer        % trainer object holding training parameters
        optimizer      % optimizer object


    end

    methods

        function self = autoencoderModel( lossFcns, ...
                                          superArgs, ...
                                          args )
            % Initialize the model
            arguments (Repeating)
                lossFcns      lossFunction
            end
            arguments
                superArgs.?representationModel
                args.XOutDim        double = 0
                args.hasSeqInput    logical = false
                args.isVAE          logical = false
                args.weights        double ...
                                    {mustBeNumeric,mustBeVector} = 1
            end

            % set the superclass's properties
            superArgsCell = namedargs2cell( superArgs );
            self = self@representationModel( superArgsCell{:} );

            % placeholders for subclasses to define
            self.nets.encoder = [];
            self.nets.decoder = [];
            self.netNames = {'encoder', 'decoder'};
            self.nNets = 2;
            self.isVAE = args.isVAE;
            self.hasSeqInput = args.hasSeqInput;

            if args.XOutDim == 0
                self.XOutDim = self.XDim;
            end

            % copy over the loss functions associated
            % and any networks with them for later training 
            self = addLossFcns( self, lossFcns{:}, weights = args.weights );

            % check a reconstruction loss is present
            if ~any( self.lossFcnTbl.types=='Reconstruction' )
                eid = 'aeModel:NoReconstructionLoss';
                msg = 'No reconstruction loss object has been specified.';
                throwAsCaller( MException(eid,msg) );
            end 

            self.nLoss = sum( self.lossFcnTbl.nLosses );

            % indicate that further initialization is required
            self.isInitialized = false;

        end


        function self = addLossFcns( self, newFcns, args )
            % Add one or more loss function objects to the model
            arguments
                self
            end
            arguments (Repeating)
                newFcns   lossFunction
            end
            arguments
                args.weights double {mustBeNumeric,mustBeVector} = 1
            end
       
            nFcns = length( newFcns );

            % check the weights
            if args.weights==1
                % default is to assign a weight of 1 to all functions
                w = ones( nFcns, 1 );
            elseif length( args.weights ) ~= nFcns
                % weights don't correspond to the functions
                eid = 'aeModel:WeightsMismatch';
                msg = 'Number of assigned weights does not match number of functions';
                throwAsCaller( MException(eid,msg) );
            else
                w = args.weights;
            end
            self.lossFcnWeights = [ self.lossFcnWeights w ];

            % update the names list
            self.lossFcnNames = [ self.lossFcnNames getFcnNames(newFcns) ];
            % add to the loss functions
            for i = 1:length( newFcns )
                self.lossFcns.(newFcns{i}.name) = newFcns{i};
            end

            % add networks, if required
            self = addLossFcnNetworks( self, newFcns );

            % store the loss functions' details 
            % and relevant details for easier access when training
            self = self.setLossInfoTbl;
            self.lossFcnTbl.types = categorical( self.lossFcnTbl.types );
            self.lossFcnTbl.inputs = categorical( self.lossFcnTbl.inputs );

        end


        function self = addLossFcnNetworks( self, newFcns )
            % Add one or more networks to the model
            arguments
                self
                newFcns
            end

            nFcns = length( newFcns );
            k = length( self.nets );
            for i = 1:nFcns
                thisLossFcn = newFcns{i};
                if thisLossFcn.hasNetwork
                    k = k+1;
                    % add the network object
                    self.nets.(thisLossFcn.name) = thisLossFcn.initNetwork;
                    % record its name
                    self.netNames = [ string(self.netNames) thisLossFcn.name ];
                    % increment the counter
                    self.nNets = self.nNets + 1;
                end
            end


        end


        function self = initTrainer( self, args )
            arguments
                self
                args.?modelTrainer
            end

            % set the trainer's properties
            argsCell = namedargs2cell( args );
            self.trainer = modelTrainer( self.lossFcnTbl, ...
                                         self.XChannels, ...
                                         self.ZDim, ...
                                         argsCell{:} );

            % confirm if initialization is complete
            self.isInitialized = ~isempty(self.optimizer);

        end
        

        function self = initOptimizer( self, args )
            arguments
                self
                args.?modelOptimizer
            end

            % set the trainer's properties
            argsCell = namedargs2cell( args );
            self.optimizer = modelOptimizer( self.netNames, argsCell{:} );

            % confirm if initialization is complete
            self.isInitialized = ~isempty(self.trainer);

        end


        function [ self, lossTrn, lossVal ] = train( self, thisDataset )
            % Train the autoencoder model
            arguments
                self          autoencoderModel
                thisDataset   modelDataset
            end

            if ~self.isInitialized
                eid = 'Autoencoder:NotInitialized';
                msg = 'The trainer, optimizer or dataset parameters have not been set.';
                throwAsCaller( MException(eid,msg) );
            end

            % check dataset is suitable
            if thisDataset.isFixedLength == self.hasSeqInput
                eid = 'aeModel:DatasetNotSuitable';
                if thisDataset.isFixedLength
                    msg = 'The dataset should have variable length for the model.';
                else
                    msg = 'The dataset should have fixed length for the model.';
                end
                throwAsCaller( MException(eid,msg) );
            end

            % re-partition the data to create training and validation sets
            cvPart = cvpartition( thisDataset.nObs, 'Holdout', 0.25 );
            
            thisTrnSet = thisDataset.partition( training(cvPart) );
            thisValSet = thisDataset.partition( test(cvPart) );

            % run the training loop
            [ self, self.optimizer ] = ...
                            self.trainer.runTraining( self, ...
                                                      thisTrnSet, ...
                                                      thisValSet );


        end

    end


    methods (Access = protected)

        function self = setLossInfoTbl( self )
            % Update the info table
            
            nFcns = length( self.lossFcnNames );
            names = strings( nFcns, 1 );
            types = strings( nFcns, 1 );
            inputs = strings( nFcns, 1 );
            weights = zeros( nFcns, 1 );
            nLosses = zeros( nFcns, 1 );
            lossNets = strings( nFcns, 1 );
            hasNetwork = false( nFcns, 1 );
            doCalcLoss = false( nFcns, 1 );
            useLoss = false( nFcns, 1 );

            for i = 1:nFcns
                
                thisLossFcn = self.lossFcns.(self.lossFcnNames(i));
                names(i) = thisLossFcn.name;
                types(i) = thisLossFcn.type;
                inputs(i) = thisLossFcn.input;
                weights(i) = self.lossFcnWeights(i);
                nLosses(i) = thisLossFcn.nLoss;
                hasNetwork(i) = thisLossFcn.hasNetwork;
                doCalcLoss(i) = thisLossFcn.doCalcLoss;
                useLoss(i) = thisLossFcn.useLoss;

                nNets = length(thisLossFcn.lossNets);
                for j = 1:nNets
                    if length(string( thisLossFcn.lossNets{j} ))==1
                        assignments = thisLossFcn.lossNets{j};
                    else
                        assignments = strjoin( thisLossFcn.lossNets{j,:}, '+' );
                    end
                    lossNets(i) = strcat( lossNets(i), assignments ) ;
                    if j<nNets
                        lossNets(i) = strcat( lossNets(i), "; " );
                    end
                end

            end

            self.lossFcnTbl = table( names, types, inputs, weights, ...
                    nLosses, lossNets, hasNetwork, doCalcLoss, useLoss );

        end

    end


    methods (Static)

        function Z = encode( self, dlX )
            % Encode features Z from X using the model

            Z = predict( self.nets.encoder, dlX );

        end

        function XHat = reconstruct( self, Z )
            % Reconstruct X from Z using the model

            XHat = predict( self.nets.decoder, Z );


        end

        function net = getNetwork( self, name )
            arguments
                self
                name         string {mustBeNetName( self, name )}
            end
            
            net = self.nets.(name);
            if isa( net, 'lossFunction' )
                net = self.lossFcns.(name).net;
            end

        end

        
        function names = getNetworkNames( self )
            arguments
                self
            end
            
            names = fieldnames( self.nets );

        end


        function loss = getReconLoss( self, dlX, dlXHat )
            % Calculate the reconstruction loss
            arguments
                self
                dlX     dlarray
                dlXHat  dlarray
            end
            
            name = self.lossFcnTbl.names( self.lossFcnTbl.types=='Reconstruction' );
            loss = self.lossFcns.(name).calcLoss( dlX, dlXHat );

        end


        function isValid = mustBeNetName( self, name )
            arguments
                self
                name
            end

            isValid = ismember( name, self.names );

        end


        function dlXC = latentComponents( decoder, dlZ, args )
            % Calculate the funtional components from the latent codes
            % using the decoder network. For each component, the relevant 
            % code is varied randomly about the mean. This is more 
            % efficient than calculating two components at strict
            % 2SD separation from the mean.
            arguments
                decoder         dlnetwork
                dlZ             dlarray
                args.sampling   char ...
                    {mustBeMember(args.sampling, ...
                        {'Random', 'Fixed'} )} = 'Random'
                args.centre     logical = true
                args.nSample    double {mustBeInteger,mustBePositive} = 10
                args.range      double {mustBePositive} = 2.0
                args.dlZMean    dlarray = []
                args.dlZLogVar  dlarray = []
            end

            nSample = args.nSample;
            ZDim = size( dlZ, 1 );
            if isempty( args.dlZMean ) || isempty( args.dlZLogVar )
                % compute the mean and SD across the batch
                dlZMean = mean( dlZ, 2 );
                dlZStd = std( dlZ, [], 2 );
            else
                % use the assigned mean and SD for the batch
                dlZMean = args.dlZMean;
                dlZStd = sqrt(exp( args.dlZLogVar ));
            end
            
            % initialise the components' Z codes at the mean
            % include an extra one that will be preserved
            dlZC = repmat( dlZMean, 1, ZDim*nSample+1 );

            % convert to double to greatly speed up processing
            % (tracing is not an issue)
            ZC = double(extractdata( dlZC ));
            ZMean = double(extractdata( dlZMean ));
            ZStd = double(extractdata( dlZStd ));
            
            if strcmp( args.sampling, 'Fixed' )
                % define the offset spacing
                offsets = linspace( -args.range, args.range, nSample );
            end

            for j = 1:nSample
                
                switch args.sampling
                    case 'Random'
                        offset = args.range*rand;
                    case 'Fixed'
                        offset = offsets(j);
                end

                for i =1:ZDim
                    % vary ith component randomly about its mean
                    ZC(i,(i-1)*nSample+j) = ZMean(i) + offset*ZStd(i);
                end

            end

            % convert back
            dlZC = dlarray( ZC, 'CB' );
           
            % generate all the component curves using the decoder
            dlXC = forward( decoder, dlZC );
            XDim = size( dlXC );

            if args.centre
                % centre about the mean curve (last curve)
                if length( XDim )==2
                    dlXC = dlXC( :, 1:end-1 ) - dlXC( :, end );
                else
                    dlXC = dlXC( :, :, 1:end-1 ) - dlXC( :, :, end );
                end
            end

        end
            

        function plotLatentComp( axes, dlXC, fda, nSample, args )
            % Plot characteristic curves of the latent codings which are similar
            % in conception to the functional principal components
            arguments
                axes          
                dlXC                dlarray
                fda                 struct
                nSample             double
                args.type           char ...
                    {mustBeMember(args.type, ...
                        {'Smoothed', 'Predicted', 'Both'} )} = 'Smoothed'
                args.shading        logical = false
                args.legend         logical = true
                args.plotTitle      string = "<Dataset>"
                args.xAxisLabel     string = "Time"
                args.yAxisLabel     string = "<Channel>"
                args.yAxisLimits    double

            end
            
            XC = double( extractdata( dlXC ) );
            % re-order the dimensions for FDA
            %XC = permute( XC, [1 3 2] );

            % smooth and re-evaluate all curves
            XCFd = smooth_basis( fda.tSpan, XC, fda.fdPar );
            XCsmth = eval_fd( fda.tSpan, XCFd );

            % set the colours from blue and red
            compColours = [ 0.0000 0.4470 0.7410; ...
                            0.6350 0.0780 0.1840 ];
            gray = [ 0.5, 0.5, 0.5 ];
            black = [ 0, 0 , 0 ];
            % set positive/negative plot characteristics
            names = [ "+ve", "-ve" ];

            nPlots = length(axes);
            j = 0;
            for c = 1:nPlots

                cla( axes(c) );
                hold( axes(c), 'on' );
                k = 1;
    
                pltObj = gobjects( 3, 1 );
                l = 0;

                % plot the gradations
                for i = 1:nSample
                    
                    % next sample
                    j = j+1;

                    isOuterCurve = (i==1 || i==nSample);
                    if isOuterCurve
                        % fill-up shading area
                        if args.shading
                            plotShadedArea( axes(c), ...
                                            fda.tSpan, ...
                                            XCsmth( :, j ), ...
                                            XCsmth( :, end ), ...
                                            compColours(k,:), ...
                                            name = names(k) );
                        end
                        width = 2.0;
                    
                    else
                        width = 1.0;
                    end

                    % plot gradation curve

                    if any(strcmp( args.type, {'Predicted','Both'} ))
                        % plot predicted values
                        plot( axes(c), ...
                              fda.tSpan, XC( :, j ), ...
                              Color = gray, ...
                              LineWidth = 0.5 );
                    end
                    if any(strcmp( args.type, {'Smoothed','Both'} ))
                        % plot smoothed curves
                        if isOuterCurve
                            % include in legend
                            l = l+1;
                            pltObj(l) = plot( axes(c), ...
                                              fda.tSpan, XCsmth( :, j ), ...
                                              Color = compColours(k,:), ...
                                              LineWidth = width, ...
                                              DisplayName = names(k));
                        else
                            % don't record it for the legend
                            plot( axes(c), ...
                                              fda.tSpan, XCsmth( :, j ), ...
                                              Color = compColours(k,:), ...
                                              LineWidth = width );
                        end

                    end

                    if mod( i, nSample/2 )==0
                        % next colour
                        k = k+1;
                    end

                end

                % plot the mean curve
                l = l+1;
                pltObj(l) = plot( axes(c), fda.tSpan, XCsmth( :, end ), ...
                               Color = black, ...
                               LineWidth = 2, ...
                               DisplayName = 'Mean' );

                hold( axes(c), 'off' );

                % finalise the plot with formatting, etc
                if args.legend && c==1
                    legend( axes(c), pltObj, Location = 'best' );
                end

                title( axes(c), ['Component ' num2str(c)] );
                xlabel( axes(c), args.xAxisLabel );
                if c==1
                    ylabel( axes(c), args.yAxisLabel );
                end
                xlim( axes(c), [fda.tSpan(1) fda.tSpan(end)] );

                if ~isempty( args.yAxisLimits )
                    ylim( axes(c), args.yAxisLimits );
                end               

                axes(c).YAxis.TickLabelFormat = '%.1f';

                finalisePlot( axes(c), square = true );


            end
        
        
        end
    

        
        function plotZDist( axis, dlZ, args )
            % Update the Z distributions plot
            arguments
                axis
                dlZ                 dlarray
                args.name           string = 'Latent Distribution'
                args.standardize    logical = false
                args.pdfLimit       double = 0.05
            end

            Z = double( extractdata( dlZ ) );

            if args.standardize
                Z = (Z-mean(Z,2))./std(Z,[],2);
                xAxisLbl = 'Std Z';
            else
                xAxisLbl = 'Z';
            end
        
            nPts = 101;
            nCodes = size( Z, 1 );
               
            hold( axis, 'off');
            for i = 1:nCodes

                % fit a distribution
                pdZ = fitdist( Z(i,:)', 'Kernel', 'Kernel', 'epanechnikov' );

                % get extremes
                Z01 = prctile( Z(i,:), 1 );
                Z50 = prctile( Z(i,:), 50 );
                Z99 = prctile( Z(i,:), 99 );

                % go well beyond range
                ZMin = Z50 - 2*(Z50-Z01);
                ZMax = Z50 + 2*(Z99-Z50);

                % evaluate the probability density function
                ZPts = linspace( ZMin, ZMax, nPts );
                Y = pdf( pdZ, ZPts );
                Y = Y/sum(Y);

                plot( axis, ZPts, Y, 'LineWidth', 1 );
                hold( axis, 'on' );

            end
            
            hold( axis, 'off');
            
            ylim( axis, [0 args.pdfLimit] );
            
            title( axis, args.name );
            xlabel( axis, xAxisLbl );
            ylabel( axis, 'Q(Z)' );   
            axis.YAxis.TickLabelFormat = '%.2f';
            
            finalisePlot( axis, square = true );

            drawnow;
            
            end
    
    
    end

end


function names = getFcnNames( lossFcns )

    nFcns = length( lossFcns );
    names = strings( nFcns, 1 );
    for i = 1:nFcns
        names(i) = lossFcns{i}.name;
    end

end


function obj = plotShadedArea( ax, t, y1, y2, colour, args )
    % Plot a shaded area
    arguments
        ax          
        t               double
        y1              double
        y2              double
        colour          double  
        args.alpha      double = 0.25
        args.name       char
    end

    % set the boundary
    tRev = [ t, fliplr(t) ];
    yRev = [ y1; flipud(y2) ];
        
    % draw shaded region
    obj = fill( ax, tRev, yRev, colour, ...
                    'FaceAlpha', args.alpha, ...
                    'EdgeColor', 'none', ...
                    'DisplayName', args.name );

end


function finalisePlot( ax, args )
    arguments
        ax
        args.square     logical = false
    end

    ax.Box = false;
    ax.TickDir = 'out';
    ax.XAxis.LineWidth = 1;
    ax.YAxis.LineWidth = 1;
    ax.FontName = 'Arial';
    if args.square
        ax.PlotBoxAspectRatio = [1 1 1];
    end

end


