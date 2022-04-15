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
                                 type = 'Component', ...
                                 input = 'XC', ...
                                 lossNets = {'encoder', 'decoder'} );

            self.criterion = args.criterion;
            self.nSample = args.nSample;

        end


        function loss = calcLoss( self, dlXC )
            % Calculate the component loss
            arguments
                self
                dlXC  dlarray  % generated AE components
            end

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

        end

    end

end


function loss = innerProduct( dlXC, nSample )
    % Calculate the inner product

    % for speed convert to ordinary numeric array
    % tracing is maintained through the mandatory reconstruction loss
    XC = double(extractdata( dlXC ));

    nComp = size( XC, 2 )/nSample;
    orth = 0;
    for i = 1:nComp
        for j = i+1:nComp
            for k = 1:nSample
                for l = k+1:nSample
                    iSample = (i-1)*nSample+k;
                    jSample = (j-1)*nSample+l;
                    orth = orth + abs( mean(XC(i,iSample,:).*XC(j,jSample,:)) );
                end
            end
        end
    end
    loss = 4*orth/(nComp*(nComp-1)*nSample*(nSample-1));

end


function loss = varimax( dlXC )
    % Calculate the varimax loss which is the
    % negative mean of component variances

    % convert to ordinary numeric array for speed
    % tracing is maintained through the mandatory reconstruction loss
    XC = double(extractdata( dlXC ));

    nObs = size( XC, 2 );
    nChannels = size( XC, 3 );
    var = zeros( nChannels, 1 );
    for i = 1:nObs
        var = var + std( XC(:,i), [], 1 ).^2;
    end

    loss = -0.01*mean(var)/nObs;   

end

