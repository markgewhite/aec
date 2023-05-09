function [coeff1, coeff2, coeffR] = coeffRatio( report )


b = length( report.IsComplete );

m = length( report.GridSearch{1} );
n = length( report.GridSearch{2} );

coeff1 = zeros( m, n );
coeff2 = zeros( m, n );
coeffR = zeros( m, n );

for i = 1:m
    for j = 1:n

        c1 = zeros( m, n );
        c2 = zeros( m, n );
        cR = zeros( m, n );

        for k = 1:b

            results = report.TrainingResults.Models{b};

            sc = sort( abs(results.AuxModelCoeff{i,j}), 'descend' );
            c1(i,j) = c1(i,j) + sc(1);
            c2(i,j) = c2(i,j) + sc(2);
            cR(i,j) = cR(i,j) + sc(1)/sc(2);

        end

        coeff1(i,j) = c1(i,j)/b;
        coeff2(i,j) = c2(i,j)/b;
        coeffR(i,j) = cR(i,j)/b;

    end
end

end