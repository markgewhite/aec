classdef ComponentLoss < LossFunction
    % Subclass for the loss functions computed from AE components

    properties
        Criterion     % criterion function for component loss
        NumSamples    % number of samples to draw to generate components
        Scale         % scaling factor between channels
    end

    methods

        function self = ComponentLoss( name, args, superArgs )
            % Initialize the loss function
            arguments
                name                 char {mustBeText}
                args.criterion       char ...
                    {mustBeMember( args.criterion, ...
                        {'Orthogonality', 'Varimax', 'ExplainedVariance'} )} ...
                                        = 'Orthogonality'
                args.nSample         double ...
                    {mustBeInteger, mustBePositive} = 10
                superArgs.?LossFunction
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@LossFunction( name, superArgsCell{:}, ...
                                 type = 'Component', ...
                                 input = 'XC', ...
                                 lossNets = {'Encoder', 'Decoder'} );

            self.Criterion = args.criterion;
            self.NumSamples = args.nSample;
            self.Scale = 1;

        end


        function self = setScale( self, data )
            % Set the scaling factor when calculating reconstruction loss
            arguments
                self        ComponentLoss
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
        
            if size( dlXC, 3 ) == 1
                dlXC = permute( dlXC, [1 3 2] );
            end
            [nPts, nChannels, nComp] = size( dlXC );
            nComp = nComp/self.NumSamples;
        
            dlXC = reshape( dlXC, nPts, nChannels, self.NumSamples, nComp );

            switch self.Criterion
                case 'Orthogonality'
                    % compute the inner product as a test
                    % of component orthogonality
                    loss = innerProduct( dlXC, nChannels, nComp, ...
                                         self.NumSamples, self.Scale );

                case 'Varimax'
                    % compute the component variance across 
                    % its length, penalising low variance
                    loss = varimax( dlXC );

                case 'ExplainedVariance'
                    % compute the component variance across 
                    % its length, penalising high variance
                    loss = explainedVariance( dlXC, nChannels, nComp, ...
                                         self.NumSamples, self.Scale );

            end

        end

    end

end


function loss = innerProduct( XC, nChannels, nComps, nSamples, scale )
    % Calculate the inner product

    orth = dlarray( zeros(1, nChannels), 'CB' );

    for c = 1:nChannels
        for k = 1:nSamples
            for i = 1:nComps
                for j = i+1:nComps
                    orth(c) = orth(c) + mean(XC(:,c,k,i).*XC(:,c,k,j))^2;
                end
            end
        end
    end

    orth = orth./scale;

    loss = 1E3*mean(orth)/(nSamples*nComps*(nComps-1));

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
        for k = 1:nSamples
            for i = 1:nComps
                var(c) = var(c) + mean( XC(:,c,k,i).^2 );
            end
        end
    end

    var = var./scale;

    loss = 1 - mean(var)/(nChannels*nComps*nSamples);


end

