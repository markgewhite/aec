% ************************************************************************
% Class: componentLoss
%
% Subclass for the loss functions computed from AE components
%
% ************************************************************************

classdef componentLoss < lossFunction

    properties
        Criterion     % criterion function for component loss
        NumSamples    % number of samples to draw to generate components
        Scale         % scaling factor between channels
    end

    methods

        function self = componentLoss( name, args, superArgs )
            % Initialize the loss function
            arguments
                name                 char {mustBeText}
                args.criterion       char ...
                    {mustBeMember( args.criterion, ...
                        {'Orthogonality', 'Varimax', 'ExplainedVariance'} )} ...
                                        = 'Orthogonality'
                args.nSample         double ...
                    {mustBeInteger, mustBePositive} = 10
                superArgs.?lossFunction
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@lossFunction( name, superArgsCell{:}, ...
                                 type = 'Component', ...
                                 input = 'XC', ...
                                 lossNets = {'encoder', 'decoder'} );

            self.Criterion = args.criterion;
            self.NumSamples = args.nSample;
            self.Scale = 1;

        end


        function self = setScale( self, data )
            % Set the scaling factor when calculating reconstruction loss
            arguments
                self        componentLoss
                data        double
            end

            scaling = squeeze(mean(var( data )))';
            self.Scale = scaling;

        end


        function loss = calcLoss( self, dlXC )
            % Calculate the component loss
            arguments
                self
                dlXC  dlarray  % generated AE components
            end

            % for speed convert to ordinary numeric array
            % tracing is maintained through the mandatory reconstruction loss
            XC = double(extractdata( dlXC ));
        
            if size( XC, 3 ) == 1
                XC = permute( XC, [1 3 2] );
            end
            nComp = size( XC, 3 )/self.NumSamples;
            nChannels = size( XC, 2 );
        
            XC = reshape( XC, size(XC,1), nChannels, nComp, self.NumSamples );
        
            % re-centre by component
            XC = XC - mean( XC, 4 );

            switch self.Criterion
                case 'Orthogonality'
                    % compute the inner product as a test
                    % of component orthogonality
                    loss = innerProduct( XC, nChannels, nComp, ...
                                         self.NumSamples, self.Scale );

                case 'Varimax'
                    % compute the component variance across 
                    % its length, penalising low variance
                    loss = varimax( XC );

                case 'ExplainedVariance'
                    % compute the component variance across 
                    % its length, penalising high variance
                    loss = explainedVariance( XC, nChannels, nComp, ...
                                         self.NumSamples, self.Scale );

            end

        end

    end

end


function loss = innerProduct( XC, nChannels, nComps, nSamples, scale )
    % Calculate the inner product

    orth = zeros( 1, nChannels );

    for c = 1:nChannels
        for i = 1:nComps
            for j = i+1:nComps
                for k = 1:nSamples
                    orth(c) = orth(c) + mean(XC(:,c,i,k).*XC(:,c,j,k))^2;
                end
            end
        end
    end

    orth = orth./scale;

    loss = 1E3*mean(orth)/(nChannels*nComps*nSamples*(nSamples-1));

end


function loss = varimax( XC )
    % Calculate the varimax loss which is the
    % negative mean of component variances

    nObs = size( XC, 2 );
    nChannels = size( XC, 3 );
    var = zeros( nChannels, 1 );
    for i = 1:nObs
        var = var + std( XC(:,i), [], 1 ).^2;
    end

    var = var./scale;

    loss = -0.01*mean(var)/nObs;   

end


function loss = explainedVariance( XC, nChannels, nComps, nSamples, scale )
    % Calculate the explained variance loss 

    var = zeros( 1, nChannels );

    for c = 1:nChannels
        for i = 1:nComps
            for k = 1:nSamples
                var(c) = var(c) + mean( XC(:,c,i,k).^2 );
            end
        end
    end

    var = var./scale;

    loss = 1 - mean(var)/(nChannels*nComps*nSamples);


end

