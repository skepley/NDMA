% how do the saddle nodes change depending on the Hill coefficient -
% minimal code

% we will actually FIX delta2, so the full vector of unknwons is "only"
% (delta1, x1, x2, v1, v2, n)


% these lines can be changed to consider different types fo non-linearity
H = @(delta, x, n) (1+ delta) /(x^n + 1);
D_xH = @(delta, x, n) (1+ delta)*n*x^(n-1)/(x^n+1)^2;
D_deltaH = @(delta, x, n) 1/(x^n + 1);
D_xdeltaH = @(delta, x, n) n*x^(n-1)/(x^n+1)^2;
second_der_ugly = @(x,n) n* x^(n-2)*(n*x^n+x^n-n+1)/(x^n+1)^3;

F = @(delta1, delta2, x1, x2,n) [ - x1/delta2 + H(delta1, x2,n);
    - x2/delta1 + H(delta2, x1,n)];

DF = @(delta1, delta2, x1, x2, n)...
    [-1/delta2,                 D_xH(delta1, x2, n)
    D_xH(delta2, x1, n),      -1/delta1];

DFv = @(delta1, delta2, x1, x2, v1, v2, n)...
      DF(delta1, delta2, x1, x2, n)*[v1;v2];

d_delta1_F = @(delta1, delta2, x1, x2,n) [ D_deltaH(delta1, x2, n)
    x2/(delta1^2)];

d_x_delta1_F = @(delta1, delta2, x1, x2,n)[0, D_xdeltaH(delta1, x2, n)
    0, 1/(delta1^2)];

d_x_delta1_Fv = @(delta1, delta2, x1, x2, v1, v2, n) d_x_delta1_F(delta1, delta2, x1, x2,n)*[v1;v2];

DDFx = @(delta1, delta2, x1, x2, v1, v2, n)...
    [0  (1+delta1)*v2*second_der_ugly(x2,n)
(1+delta2)*v1*second_der_ugly(x1,n) 0];

phase_cond = @(v1, v2) v1 * v1 + v2*v2 - 1;

d_v_phase_cond = @(v1,v2) [2*v1, 2*v2];

saddle_problem = @(delta1, delta2, x1, x2, v1, v2, n)...
    [F(delta1, delta2, x1, x2, n)
    DFv(delta1, delta2, x1, x2, v1, v2, n)
    phase_cond(v1, v2)];

Dsaddle = @(delta1, delta2, x1, x2, v1, v2, n) ...
    [d_delta1_F(delta1, delta2, x1, x2,n)  DF(delta1, delta2, x1, x2, n) zeros(2,2)
    d_x_delta1_Fv(delta1, delta2, x1, x2, v1, v2, n) DDFx(delta1, delta2, x1, x2, v1, v2, n) DF(delta1, delta2, x1, x2, n)
     0 0 0 d_v_phase_cond(v1,v2)];

a = Dsaddle(1,2,3,4,5,6,7);

n_fix = 8;
delta2_fix = 1;


% steps: 
% find approx (x1, x2) for delta1 = 1 or 2 - DONE
%     problem: needs to be the one undergoing a saddle node!
% compute eigenvector associated to smallest eigenvalue 
% compute solution of saddle_problem with Newton starting at approximate
% solution
% wrap all in for loop for quite some values of delta2 and n
% plot nicely
% ----> conquer the world

% STEP 1: approximate (x1, x2) given delta1 = 2, all rest fixed

for i = -0.5:0.1:0.5

delta1_fix = 1 + i;

F_vector = @(X) F(delta1_fix, delta2_fix, X(1), X(2),n_fix); 
DF_vector = @(X) DF(delta1_fix, delta2_fix, X(1), X(2),n_fix); 

X_sol = find_eqs(F_vector, DF_vector);

figure(1)
plot(delta1_fix, X_sol(1,:), 'o')
hold on
figure(2)
plot(delta1_fix,X_sol(2,:), 'o')
hold on

if size(X_sol, 2)>2
    disp(delta1_fix)
end
end


F_vector_longer = @(X) F(X(1), delta2_fix, X(2), X(3),n_fix); 
DF_vector_longer = @(X) DF(X(1), delta2_fix, X(2), X(3),n_fix);
X_approx_longer = [delta1_fix; X_approx; v_approx];

saddle_problem_vector = @(X) saddle_problem(X(1), delta2_fix, X(2), X(3), X(4), X(5), n_fix);
Dsaddle_vector = @(X) Dsaddle(X(1), delta2_fix, X(2), X(3), X(4), X(5), n_fix);


X_longer = Newton(saddle_problem_vector, Dsaddle_vector, X_approx_longer);



function x = Newton(f, df, x_hat)
x = x_hat;
max_iter = 200;
iter_info = zeros(max_iter,2);
for i = 1:max_iter
    x_new = x - df(x)\f(x);
    if norm(f(x))<10^-6
        break
    end
    x = x_new;
    iter_info(i,1) = norm(f(x));
    iter_info(i,2) = det(df(x));
end

x = x_new;

if i == max_iter
    warning('Newton did not converge')
    %plot(iter_info(:,2))
    %figure
    %plot(iter_info(:,1))
end

end



function X_sol = find_eqs(F, DF)
x1_start = meshgrid([0.1:0.2:3],[0.1:0.2:3]);
x2_start = meshgrid([0.1:0.2:3],[0.1:0.2:3]).';
X_start = [x1_start(:),x2_start(:)]';

warning OFF

X_sol =  [];

for i = 1:size(X_start,2)
    
    X_approx_new = X_start(:,i);
    
    X_new = Newton(F, DF, X_approx_new);
    
    if norm(F(X_new))<10^-6
        if ~is_a_column(X_new, X_sol)
            X_sol = [X_sol , X_new];
        end
    end
end
warning ON

[~,neg_index_column] = find(X_sol<0);
X_sol(:,neg_index_column) = [];

end

function bool_ = is_a_column(X_new, X_sol)

bool_ = 0;
for i = 1:size(X_sol, 2)
    bool_ = (norm(X_new - X_sol(:,i))<10^-5);
    if bool_
        return
    end
end

end
