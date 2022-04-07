% ************************************************************************
% Class: componentLoss
%
% Subclass for the loss functions computed from AE components
%
% ************************************************************************

classdef componentLoss < lossFunction

    properties
        criterion   % criterion function for component loss
        nSample     % number of samples to draw to generate components
    end

    methods

        function self = componentLoss( name, args, superArgs )
            % Initialize the loss function
            arguments
                name                 char {mustBeText}
                args.criterion       char ...
                    {mustBeMember( args.criterion, ...
                        {'Orthogonality', 'Varimax'} )} = 'Orthogonality'
                args.nSample         double ...
                    {mustBeInteger, mustBePositive} = 10
                superArgs.?lossFunction
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@lossFunction( name, superArgsCell{:}, ...
                                 type = 'Component' );
            self.criterion = args.criterion;
            self.nSample = args.nSample;

        end

    end

    methods (Static)

        function loss = calcLoss( self, model, dlZ )
            % Calculate the component loss
            if self.doCalcLoss

                dlXC = latentComponents( model, dlZ, self.nSample );
                
                switch self.criterion
                    case 'Orthogonality'
                        % compute the inner product as a test
                        % of component orthogonality
                        loss = innerProduct( dlXC, self.nSample );

                    case 'Varimax'
                        % compute the component variance across 
                        % its length, penalising low variance
                        loss = varimax( dlXC );

                end
    
            else
                loss = 0;
            end

        end

        function loss = innerProduct( dlXC, nSample )
            % Calculate the inner product
            nDim = size( dlXC, 1 );
            orth = 0;
            for i = 1:nDim
                for j = i+1:nDim
                    iIdxRng = (i-1)*nSample+1:i*nSample;
                    jIdxRng = (j-1)*nSample+1:j*nSample;
                    orth = orth + ...
                        mean( dlXC(i,iIdxRng,:).*dlXC(j,jIdxRng,:), 'all' );
                end
            end
            loss = orth/(nDim*(nDim-1));

        end


        function loss = varimax( dlXC )
            % Calculate the varimax loss which is the
            % negative mean of component variances
            nObs = size( dlXC, 2 );
            nChannels = size( dlXC, 3 );
            var = zeros( nChannels, 1 );
            for i = 1:nObs
                var = var + std( dlXC(:,i), [], 2 )^2;
            end
            loss = -mean(var)/nObs;
        end

    end


end