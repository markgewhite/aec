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
        Scale               % scaling factor for channels
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
                    {mustBePositive} = 1E-2
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

            self.Scale = 1;

        end



        function self = setScale( self, data )
            % Set the scaling factor when calculating the loss
            arguments
                self        smoothnessLoss
                data        double
            end

            scaling = squeeze(mean(var( data )))';
            self.Scale = scaling;

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

            if isa( dlXHat, 'dlarray' )
                % dimensions are Time x Channel x Batch
                if size( XHat, 3 ) == 1
                    loss = mean( (XHat-XHatSmth).^2, [1 2] )/self.Scale;
                else
                    loss = mean(mean( (XHat-XHatSmth).^2, [1 3] )./self.Scale);
                end
            else
                % dimensions are Time x Batch x Channel
                if size( XHat, 3 ) == 1
                    loss = mean( (XHat-XHatSmth).^2, [1 2] )/self.Scale;
                else
                    loss = mean(mean( (XHat-XHatSmth).^2, [1 2] )./ ...
                                        permute(self.Scale, [1 3 2]));
                end
            end

            loss = 10*loss;

        end

    end

end


