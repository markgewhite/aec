function Hline = subplotFd( ax, fdobj, Lfdobj, matplt, href, nx)
%  PLOT   Plot a functional data object.
%  *** MODIFICATION ***
%  Arguments:
%  AX      ... Axis object for the plot (new)
%  FDOBJ   ... A functional data object to be plotted.
%  LFDOBJ  ... A linear differential operator object or a positive
%              integer specifying a derivative.  This operator is
%              applied to the functions before plotting.
%  MATPLOT ... If MATPLT is nonzero, all curves are plotted in a 
%              single plot.
%              Otherwise, each curve is plotted separately, and the
%              next curve is plotted when the mouse is clicked.
%  HREF    ... If HREF is nonzero, a horizontal dotted line through 0 
%              is plotted.
%  NX      ... The number of plotting points to be used.

%  Last modified 20 October 2015

%  set default arguments

if nargin < 5 || isempty(href),   href = 1;             end
if nargin < 4 || isempty(matplt), matplt = 1;           end
if nargin < 3 || isempty(Lfdobj), Lfdobj = int2Lfd(0);  end

%  check arguments

if ~isa_fd(fdobj)
    error ('Argument fdobj not a functional data object.');
end

Lfdobj = int2Lfd(Lfdobj);
if ~isa_Lfd(Lfdobj)
    error ('Argument Lfdobj not a linear differential operator object.');
end

%  extract basis information

basisobj = getbasis(fdobj);
rangex   = getbasisrange(basisobj);
nbasis   = getnbasis(basisobj);
type     = getbasistype(basisobj);

%  special case of an FEM basis

if strcmp(type, 'FEM')
    plot_FEM(fdobj);
    return;
end

%  special case of an fdVar basis

if strcmp(type, 'fdVar')
    cvec     = getcoef(fdobj);
    basisobj = getbasis(fdobj);
    T        = max(getbasisrange(basisobj));
    tfine    = linspace(0,T,51)';
    RstCell  = eval_basis(tfine, basisobj);
    Sigma    = fdVar_Sigma(cvec, RstCell);      
    surf(tfine, tfine, Sigma);
    return;
end


% set up dimensions of problem

coef   = getcoef(fdobj);
coefd  = size(coef);
ndim   = length(coefd);
ncurve = coefd(2);

if ndim > 2
    nvar = coefd(3);
else
    nvar = 1;
end

%  set up fine mesh of evaluation points and evaluate curves

if nargin < 5
    nx = min([201, 10*nbasis+1]);
end

x     = linspace(rangex(1),rangex(2),nx)';
fdmat = eval_fd(x, fdobj, Lfdobj);

%  calculate range of values to plot

switch ndim
    case 2
        frng(1) = min(min(fdmat));
        frng(2) = max(max(fdmat));
    case 3
        frng(1) = min(min(min(fdmat)));
        frng(2) = max(max(max(fdmat)));
    otherwise
        frng = [1 1];
end

%  fix range if limits are equal

if frng(1) == frng(2)
    if abs(frng(1)) < 1e-1
        frng(1) = frng(1) - 0.05;
        frng(2) = frng(2) + 0.05;
    else
        frng(1) = frng(1) - 0.05*abs(frng(1));
        frng(2) = frng(2) + 0.05*abs(frng(1));
    end
end

%  extract argument, case and variable names

fdnames  = getnames(fdobj);

%  --------------------  Plot for a single variable  ----------------------

colororder( ax, 'default' );

if ndim == 2
    if matplt
        %  plot all curves
        if href && (frng(1) <= 0 && frng(2) >= 0)
            if nargout > 0
                Hline = plot( ax, x, fdmat, '-', ...
                                  x, zeros(nx), ':', ...
                                  'Linewidth', 1 );
            else
                plot( ax, x, fdmat, '-', ...
                          x, zeros(nx), ':', ...
                          'Linewidth', 1 );
            end
        else
            if nargout > 0
                Hline = plot( ax, x, fdmat, '-');
            else
                plot( ax, x, fdmat, '-')
            end
        end
        xlabel(['\fontsize{12} ',fdnames{1}]);
        if iscell(fdnames{3})
            ylabel( ax, ['\fontsize{12} ',fdnames{3}{1}] )
        else
            ylabel( ax, ['\fontsize{12} ',fdnames{3}] )
        end
        if frng(2) > frng(1)
            axis( ax, [x(1), x(nx), frng(1), frng(2)] );
        end
    else
        %  page through curves one by one
        for icurve = 1:ncurve
            if href && (frng(1) <= 0 && frng(2) >= 0)
                if nargout > 0
                    Hline = plot( ax, x, fdmat(:,icurve), 'b-', ...
                                 [min(x),max(x)], [0,0], 'r:');
                else
                    plot( ax, x, fdmat(:,icurve), 'b-', ...
                         [min(x),max(x)], [0,0], 'r:')
                end
            else
                if nargout > 0
                    Hline = plot( ax, x, fdmat(:,icurve), 'b-');
                else
                    plot( ax, x, fdmat(:,icurve), 'b-')
                end
            end
            xlabel( ax, ['\fontsize{12} ',fdnames{1}] )
            if iscell(fdnames{3})
                ylabel( ax, ['\fontsize{12} ',fdnames{3}{1}])
            else
                ylabel( ax, ['\fontsize{12} ',fdnames{3}])
            end
            if iscell(fdnames{2})
                title( ax, ['\fontsize{12}', fdnames{2}{2}(icurve,:)]);
            else
                title( ax, ['\fontsize{12} Curve ', num2str(icurve)]);
            end
            pause;
        end
    end
end

%  --------------------  Plot for multiple variables  ---------------------

if ndim == 3
    error('Multiple variables not supported for this function.')
end

end


