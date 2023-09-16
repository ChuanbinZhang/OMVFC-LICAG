function [F, cost] = OMVFC_LICAG_core(X, H, n_clusters, alpha, beta, max_iter, error, verbose)
rng(123,'twister')
n_views = length(X);
n_points = size(X{1}, 2);

G = zeros(n_clusters, n_points, n_views);
D = zeros(n_clusters, n_points, n_views + 1);
w = zeros(max_iter + 1, 1, n_views);
term = zeros(max_iter + 1, 3, n_views);
term3 = zeros(max_iter + 1, 1);
cost = zeros(max_iter + 1, 1);

%% Initialization
[~, F, ~, Dh] = myfcm(H', n_clusters, 1.1, 1);
D(:,:,n_views + 1) = Dh .^ 2;
F_old = F;
term3(1) = beta * trace(F * D(:,:,n_views + 1)');
for k=1:n_views
    Xk = X{k};
    n_features = size(Xk, 1);
    [~, Uk, ~, Dk] = myfcm(Xk', n_clusters, 1.1, 1);
    D(:,:,k) = (Dk.^2) ./ n_features;

    Qk = Uk * F';
    [S,~,V] = svd(Qk, 'econ');
    Pk = V * S';

    G(:,:,k) = Pk * Uk;
    term(1, 1, k) = alpha * trace(Uk * D(:,:,k)');
    term(1, 2, k) = norm(F - G(:,:,k), "fro")^2;
    w(1,1,k) = 0.5 / sqrt(term(1, 1, k) + term(1, 2, k));
    term(1, 3, k) = w(1,1,k) * sum(term(1, 1:2,k));
end
cost(1) = sum(term(1, 3, :), 3) + term3(1);

%% Optimization
for iter=1:max_iter
    % Update F
    G_sum = (sum(w(iter,1,:).*G, 3) - 0.5*beta*D(:,:,n_views + 1)) ./ sum(w(iter,1,:), "all");
    for j=1:n_points
        f_j = EProjSimplex(G_sum(:,j),1) + eps;
        F(:,j) = f_j ./ sum(f_j, "all");
    end

    % Update O
    O = ((F * H')./sum(F, 2))';
    D(:,:,n_views + 1) = L2_distance(O, H);

    term3(iter + 1) = beta * trace(F * D(:,:,n_views + 1)');

    for k=1:n_views
        Xk = X{k};
        n_features = size(X{k}, 1);
        Uk = zeros(n_clusters, n_points);

        % Update U
        Bk = 0.5 * (2*Pk*F - alpha * D(:,:,k));
        for j=1:n_points
             u_jk= EProjSimplex(Bk(:,j),1) + eps;
             Uk(:,j) = u_jk ./ sum(u_jk, "all");
        end

        % Update P
        Qk = Uk * F';
        [S,~,V] = svd(Qk, 'econ');
        Pk = V * S';

        % Update C
        Ck = ((Uk * Xk')./sum(Uk, 2))';
        D(:,:,k) = L2_distance(Ck, Xk) ./ n_features;

        % Update w
        G(:,:,k) = Pk * Uk;
        term(iter+1, 1, k) = alpha * trace(Uk * D(:,:,k)');
        term(iter+1, 2, k) = norm(F - G(:,:,k), "fro")^2;
        w(iter+1,1,k) = 0.5 / sqrt(term(iter+1, 1, k) + term(iter+1, 2, k));
        term(iter+1,3,k) = w(iter+1,1,k)*sum(term(iter+1, 1:2,k));
    end

    % Compute objective function value
    cost(iter + 1) = sum(term(iter + 1, 3, :), 3) + term3(iter + 1);

    % Convergence criterion
    if (norm(F - F_old, 'inf') < error || abs(cost(iter + 1) - cost(iter)) / cost(iter) < error)
        cost = cost(1:iter + 1);
        term = term(1:iter + 1, :,:);
        term3 = term3(1:iter + 1, :,:);
        w = w(1:iter + 1, :);
        break;
    end

    F_old = F;
end

if verbose == 1
    for k=1:n_views
        figure('Name',sprintf("%s:view%d", dataset, k));
        plot(term(:,1:3,k));
        legend('term1','term2', 'OBJ')
    end
    figure('Name',sprintf("%s:Laplacian Eigenmaps", dataset));
    plot(term3);
    legend('term3');

    figure('Name',sprintf("%s:w", dataset));
    plot(reshape(w, iter+1, n_views));
end
end