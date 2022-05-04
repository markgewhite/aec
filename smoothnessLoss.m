% ************************************************************************
% Class: smoothnessLoss
%
% Subclass for the loss functions computed from AE components
%
% ************************************************************************

classdef smoothnessLoss < lossFunction

    properties
        NumBasisFcns        % number of basis functions
        BasisOrder          % smoothing basis order
        PenaltyOrder        % roughness penalty order
        Lambda              % roughness penalty
        BasisFd             % basis functional data object
        FdParams            % functional data parameters
    end

    methods

        function self = smoothnessLoss( name, args, superArgs )
            % Initialize the loss function
            arguments
                name                 char {mustBeText}
                args.NumBasisFcns    double ...
                    {mustBeInteger,mustBePositive} = 10
                args.BasisOrder      double ...
                    {mustBeInteger,mustBePositive} = 4
                args.PenaltyOrder    double ...
                    {mustBeInteger,mustBePositive} = 1
                args.Lambda          double ...
                    {mustBePositive} = 1E-1
                superArgs.?lossFunction
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@lossFunction( name, superArgsCell{:}, ...
                                 type = 'Output', ...
                                 input = 'XHat', ...
                                 lossNets = {'encoder', 'decoder'} );

            self.NumBasisFcns = args.NumBasisFcns;
            self.BasisOrder = args.BasisOrder;
            self.PenaltyOrder = args.PenaltyOrder;
            self.Lambda = args.Lambda;

            self.BasisFd = create_bspline_basis( ...
                              [ 0, 1 ], ...
                              self.NumBasisFcns+self.PenaltyOrder, ...
                              self.BasisOrder );

            self.FdParams = fdPar( self.BasisFd, ...
                                   self.PenaltyOrder, ...
                                   self.Lambda );

        end


        function loss = calcLoss( self, dlXHat )
            % Calculate the component loss
            arguments
                self        smoothnessLoss
                dlXHat      dlarray  % output
            end

            XHat = double(extractdata( dlXHat));
            nPts = size( XHat, 1 );

            % calculate the smoothed curve
            tSpan = linspace( 0, 1, nPts );
            XHatFd = smooth_basis( tSpan, XHat, self.FdParams );
            XHatSmth = eval_fd( tSpan, XHatFd );

            loss = mean( (XHat - XHatSmth).^2, 'all' );

        end

    end

end


