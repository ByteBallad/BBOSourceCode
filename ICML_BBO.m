function [RunResult, RunValue, RunTime, RunFES, RunOptimization, RunParameter] = ICML_BBO(problem, N, runmax)
'ICML_BBO'
D = Dim(problem); % 3-6 rows, please refer to CEP
lu = Boundary(problem, D);
TEV = Error();
FESMAX = 10000 * D;
RunOptimization = zeros(runmax, D);

for run = 1 : runmax
    TimeFlag = 0;
    TempFES = FESMAX;
    t1 = clock;
    
    x = Initpop(N, D, lu); % population initialization
    
    fitness = benchmark_func(x, problem); % calculate individual fitness
    FES = N; % current function evaluation times
    [fitness_sorted, order] = sort(fitness); % the population was ordered by fitness
    sort_population = x(order, :);
    
    % Parameterized initialization
    Iteration = 1; % Current iteration
    OPTIONS.popsize = N; % total population size
    OPTIONS.Maxgen = FESMAX / N; % generation count limit
    OPTIONS.numVar = D; % number of genes in each population member
    OPTIONS.pmodify = 1; % habitat modification probability
    OPTIONS.pmutate = 0.005; % initial mutation probability
    Keep = 2; % elitism parameter: how many of the best habitats to keep from one generation to the next
    lambdaLower = 0.0; % lower bound for migration probability per gene
    lambdaUpper = 1; % upper bound for migration probability per gene
    dt = 1; % step size used for numerical integration of probabilities
    I = 1; % max immigration rate for each island
    E = 1; % max emigration rate for each island
    P = OPTIONS.popsize; % max species count for each island
    for popindex = 1 : OPTIONS.popsize
        Population(popindex).chrom = sort_population(popindex, :);
        Population(popindex).fitness = fitness_sorted(popindex);
    end
    % Initialize the species count probability of each habitat
    % Later we might want to initialize probabilities based on cost
    for j = 1 : length(x)
        Prob(j) = 1 / length(x);
    end
    h = 1; % sampling points during each run
    
    % ICML
    Pe = 0.5; % devising this parameter to control the ratio of CMM to the original migration
    %Pr = 0.0; % the probability of using GCML in DCML
    %Pc = 0.2; % the probability of using nearest_learning in LCML
    Pc = 0.5;
    T1 = ceil(P/5); % the size of nearest subpopulation
    T2 = ceil(P);% the size of furthest subpopulation
    % The relavent parameters of learning period
    learngen = 50;  % learning period
    % Recording the number of successfully migrated habitats under
    % different coordinate systems for the current generation
    ns_pcount = [];
    % Recording the number of unsuccessfully migrated habitats under
    % different coordinate systems for the current generation
    nf_pcount = [];
    % Recording the number of successfully migrated habitats based on
    % different subpopulations under the Eigen coordinate system for the current generation
    nx_pcount = [];
    % Recording the number of unsuccessfully migrated habitats based on
    % different subpopulations under the Eigen coordinate system for the current generation
    nu_pcount = [];
    % Recording the number of successfully migrated habitats under
    % different coordinate systems for the previous learngen generations
    ns = [];
    % Recording the number of unsuccessfully migrated habitats under
    % differernt coordinate systems for the previous learngen generations
    nf = [];
    % Recording the number of successfully migrated habitats based on
    % different subpopulations under the Eigen coordinate system for the previous learngen generations
    nx = [];
    % Recording the number of unsuccessfully migrated habitats based on
    % different subpopulations under the Eigen coordinate system for the previous learngen generations
    nu = [];
    % the success rate
    SPe = [];
    SPc = [];
    % ICML
     % Begin the optimization loop 
    while FES <= FESMAX
        % ICML
        ns_pcount = zeros(1, 2);
        nf_pcount = zeros(1, 2);
        nx_pcount = zeros(1, 2);
        nu_pcount = zeros(1, 2);
        %ICML
        % Save the best habitats in a temporary array
        for j = 1 : Keep
            chromKeep(j, :) = Population(j).chrom;
            fitnessKeep(j) = Population(j).fitness;
        end
        % Map fitness values to species counts
        [Population] = GetSpeciesCounts(Population, P);
        % Calculate immigration rate and emigration rate for each species count
        % lambda(i) is the immigration rate for individual i
        % mu(i) is the emigration rate for individual i
        [lambda, mu] = GetLambdaMu(Population, I, E, P);
        % ProbFlag = true or false, whether or not to use probabilities to
        % update migration rates and to mutate
        ProbFlag = true;
        if ProbFlag
            % Compute the time derivative of Prob(i) for each habitat i.
            for j = 1 : length(Population)
                % Compute lambda for one less than the species count of habitat i.
                lambdaMinus = I * (1 - (Population(j).SpeciesCount - 1) / P);
                % Compute mu for one more than the species count of habitat i.
                muPlus = E * (Population(j).SpeciesCount + 1) / P;
                % Compute Prob for one less than and one more than the species count of habitat i.
                % Note that species counts are arranged in an order opposite to that presented in
                % MacArthur and Wilson's book - that is, the most fit
                % habitat has index 1, which has the highest species count.
                if j < length(Population)
                    ProbMinus = Prob(j+1);
                else
                    ProbMinus = 0;
                end
                if j > 1
                    ProbPlus = Prob(j-1);
                else
                    ProbPlus = 0;
                end
                ProbDot(j) = -(lambda(j) + mu(j)) * Prob(j) + lambdaMinus * ProbMinus + muPlus * ProbPlus;
            end
            % Compute the new probabilities for each species count.
            Prob = Prob + ProbDot * dt;
            Prob = max(Prob, 0);
            Prob = Prob / sum(Prob);
        end
        % Now use lambda and mu to decide how much information to share between habitats
        lambdaMin = min(lambda);
        lambdaMax = max(lambda);
%         [Q1, ~] = eig(cov(sort_population)); % calculate the eigenvectors of the covariance matrix of the whole population
%         X_eig1 = sort_population * Q1; % transform the original coordinate system into eigen coordinate system
        % ICML
        Pmax = max(Prob);
        MutationRate = OPTIONS.pmutate * (1 - Prob / Pmax);
        % ICML
        for k = 1 : length(Population)
            if rand > OPTIONS.pmodify
                continue;
            end
            % Normalize the immigration rate
            lambdaScale = lambdaLower + (lambdaUpper - lambdaLower) * (lambda(k) - lambdaMin) / (lambdaMax - lambdaMin);
            % Probabilistically input new information into habitat i
            % Select migration based covariance matrix or original
            % migration according to parameter Pe
            % Dynamic Covariance Matrix Learning
%             if rand < Pr
%                 X_eig = X_eig1;
%                 flag = 1;
%             else
%                 dist = pdist2(sort_population(k,:),sort_population);
%                 [~,index] = sort(dist);
%                 SubPop_nearest = sort_population(index(1:T1),:);
%                 SubPop_furthest = sort_population(index(P-T2+2:P),:);
%                 SubPop_furthest(T2,:) = sort_population(k,:);
%                 %SubPop_furthest = sort_population(index(1:P),:);
%                 if rand < Pc
%                     [Q2,~] = eig(cov(SubPop_nearest));
%                     X_eig = sort_population * Q2;
%                     flag = 2;
%                 else
%                     [Q3,~] = eig(cov(SubPop_furthest));
%                     X_eig = sort_population * Q3;
%                     flag = 3;
%                 end
%             end
            % Pe = 1;
            if rand < Pe
                dist = pdist2(sort_population(k,:),sort_population);
                [~,index] = sort(dist);
                SubPop_nearest = sort_population(index(1:T1),:);
                SubPop_furthest = sort_population(index(P-T2+2:P),:);
                SubPop_furthest(T2,:) = sort_population(k,:);
                %SubPop_furthest = sort_population(index(1:P),:);
                % Pc = 1;
                if rand < Pc
                    [Q2,~] = eig(cov(SubPop_nearest));
                    X_eig = sort_population * Q2;
                    flag = 2;
                else
                    [Q3,~] = eig(cov(SubPop_furthest));
                    X_eig = sort_population * Q3;
                    flag = 3;
                end
                for j = 1 : OPTIONS.numVar 
                    if rand < lambdaScale
                        % Pick a habitat from which to obtain a feature
                        RandomNum = rand * sum(mu);
                        Select = mu(1);
                        SelectIndex = 1;
                        while (RandomNum > Select) && (SelectIndex < OPTIONS.popsize)
                            SelectIndex = SelectIndex + 1;
                            Select = Select + mu(SelectIndex);
                        end
                        Island(k,j) = X_eig(SelectIndex, j);
                    else
                        Island(k,j) = X_eig(k, j);
                    end
                end
                if flag == 1
                    Island(k, :) =Island(k, :) * Q1';
                elseif flag == 2
                    Island(k, :) =Island(k, :) * Q2';
                    % ICML
                    for parnum = 1 : OPTIONS.numVar
                        if MutationRate(k) > rand
                            Island(k,parnum) = floor(lu(1) + (lu(2) - lu(1)) * rand);
                            % Make sure each individual is legal
                            Island(k, parnum) = max(Island(k, parnum), lu(1));
                            Island(k, parnum) = min(Island(k, parnum), lu(2));
                        end
                    end
                    tmp_fitness = benchmark_func(Island(k,:), problem);% calculate the fitness of the kth habitat
                    if tmp_fitness < fitness(k)
                        ns_pcount(2) = ns_pcount(2) + 1;
                        nx_pcount(2) = nx_pcount(2) + 1;
                    else
                        nf_pcount(2) = nf_pcount(2) + 1;
                        nu_pcount(2) = nu_pcount(2) + 1;
                    end
                    fitness(k) = tmp_fitness;
                    % ICML
                else
                    Island(k, :) =Island(k, :) * Q3'; 
                    % ICML
                    for parnum = 1 : OPTIONS.numVar
                        if MutationRate(k) > rand
                            Island(k,parnum) = floor(lu(1) + (lu(2) - lu(1)) * rand);
                            % Make sure each individual is legal
                            Island(k, parnum) = max(Island(k, parnum), lu(1));
                            Island(k, parnum) = min(Island(k, parnum), lu(2));
                        end
                    end
                    tmp_fitness = benchmark_func(Island(k,:), problem);% calculate the fitness of the kth habitat
                    if tmp_fitness < fitness(k)
                        ns_pcount(2) = ns_pcount(2) + 1;
                        nx_pcount(1) = nx_pcount(1) + 1;
                    else
                        nf_pcount(2) = nf_pcount(2) + 1;
                        nu_pcount(1) = nu_pcount(1) + 1;
                    end
                    fitness(k) = tmp_fitness;
                    % ICML
                end
            else
                for j = 1 : OPTIONS.numVar
                    if rand < lambdaScale
                        % Pick a habitat from which to obtain a feature
                        RandomNum = rand * sum(mu);
                        Select = mu(1);
                        SelectIndex = 1;
                        while (RandomNum > Select) && (SelectIndex < OPTIONS.popsize)
                            SelectIndex = SelectIndex + 1;
                            Select = Select + mu(SelectIndex);
                        end
                        Island(k,j) = Population(SelectIndex).chrom(j);
                    else
                        Island(k,j) = Population(k).chrom(j);
                    end
                end
                % ICML
                for parnum = 1 : OPTIONS.numVar
                    if MutationRate(k) > rand
                        Island(k,parnum) = floor(lu(1) + (lu(2) - lu(1)) * rand);
                        % Make sure each individual is legal
                        Island(k, parnum) = max(Island(k, parnum), lu(1));
                        Island(k, parnum) = min(Island(k, parnum), lu(2));
                    end
                end
                tmp_fitness = benchmark_func(Island(k,:), problem);% calculate the fitness of the kth habitat
                if tmp_fitness < fitness(k)
                    ns_pcount(1) = ns_pcount(1) + 1;
                else
                    nf_pcount(1) = nf_pcount(1) + 1;
                end
                fitness(k) = tmp_fitness;
                % ICML
            end
        end
        %{
        if ProbFlag
            % Mutation
            Pmax = max(Prob);
            MutationRate = OPTIONS.pmutate * (1 - Prob / Pmax);
            % Mutate the all of the solutions
            for k = 1 : length(Population)
            % Mutate only the worst half of the solutions          
            % for k = round(length(Population)/2) : length(Population)
                for parnum = 1 : OPTIONS.numVar
                    if MutationRate(k) > rand
                        Island(k,parnum) = floor(lu(1) + (lu(2) - lu(1)) * rand);
                        % Make sure each individual is legal
                        Island(k, parnum) = max(Island(k, parnum), lu(1));
                        Island(k, parnum) = min(Island(k, parnum), lu(2));
                    end
                end
            end
        end
        %}
        % ICML
        %fitness = benchmark_func(Island, problem); % calculate fitness
        ns = [ns; ns_pcount];
        nf = [nf; nf_pcount];
        nx = [nx; nx_pcount];
        nu = [nu; nu_pcount];
        if Iteration >= learngen
            SPe = zeros(1, 2);
            SPc = zeros(1, 2);
            SPe(1) = sum(ns(:,1)) / (sum(ns(:,1))+sum(nf(:,1))+0.01);
            SPe(2) = sum(ns(:,2)) / (sum(ns(:,2))+sum(nf(:,2))+0.01);
            SPc(1) = sum(nx(:,1)) / (sum(nx(:,1))+sum(nu(:,1))+0.01);
            SPc(2) = sum(nx(:,2)) / (sum(nx(:,2))+sum(nu(:,2))+0.01);
            if (SPe(1)+SPe(2) > 0)
                Pe = SPe(2) / (SPe(1)+SPe(2));
            else
                Pe = 1;
            end
            if (SPc(1)+SPc(2) > 0)
                Pc = SPc(2) / (SPc(1)+SPc(2));
            else
                Pc = 1;
            end
            %{
            if (sum(ns(:,1))+sum(ns(:,2))) > 0
                %Pe = sum(ns(:, 2)) / (sum(ns(:,1))+sum(ns(:,2)))+0.01;
                Pe = sum(ns(:,2)) / (sum(ns(:,1))+sum(ns(:,2))+0.01);
            else
                Pe = 0.5;
            end
            if (sum(nx(:,1))+sum(nx(:,2))) > 0
                %Pc = sum(nx(:,1)) / (sum(nx(:,1))+sum(nx(:,2)))+0.01;
                Pc = sum(nx(:,1)) / (sum(nx(:,1))+sum(nx(:,2))+0.01);
            else
                Pc = 0.5;
            end
            %}
            if ~isempty(ns)
                ns(1,:) = [];
                nf(1,:) = [];
            end
            if ~isempty(nx)
                nx(1,:) = [];
                nu(1,:) = [];
            end
        end
        % ICML
        [fitness_sorted, order] = sort(fitness); % the population was ordered by fitness
        sort_population = Island(order, :);
        % Replace the worst with the previous generation's elites
        n = length(Population);
        for k = 1 : Keep
            sort_population(n - k + 1, :) = chromKeep(k, :);
            fitness_sorted(n - k + 1) =  fitnessKeep(k);
        end
        [fitness_sorted, order] = sort(fitness_sorted); % the population was ordered again
        sort_population = sort_population(order, :);
        for i = 1 : OPTIONS.popsize
            Population(i).chrom = sort_population(i, :);
            Population(i).fitness = fitness_sorted(i);
        end
        % Make sure the population does not have duplicates
        Population = ClearDups(Population, lu(2), lu(1));
        for i = 1 : OPTIONS.popsize
            sort_population(i, :) = Population(i).chrom;
        end
        for i = 1 : OPTIONS.popsize
            if FES == 10000 * 0.1 || mod(FES, 10000) == 0
                [kk, ll] = min(fitness_sorted);
                RunValue(run, h) = kk;
                Para(h, :) = sort_population(ll, :);
                h = h + 1;
                fprintf('Algorithm:%s problemIndex:%d Run:%d FES:%d Best:%g\n','ICML_BBO',problem,run,FES,kk);
            end
            FES = FES + 1;
            if TimeFlag == 0
                if min(fitness_sorted) <= TEV
                    TempFES = FES;
                    TimeFlag = 1;
                end
            end
        end
        Iteration = Iteration + 1;
    end
    [kk, ll] = min(fitness_sorted);
    gbest = sort_population(ll, :);
    t2 = clock;
    RunTime(run) = etime(t2, t1);
    RunResult(run) = kk;
    RunFES(run) = TempFES;
    RunOptimization(run, 1 : OPTIONS.numVar) = gbest;
    RunParameter{run} = Para;
end
return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Population] = GetSpeciesCounts(Population, P)
% Map fitness values to species counts
% This loop assumes the population is already sorted from most fit to least
% fit
for i = 1 : length(Population)
    if Population(i).fitness < inf
        Population(i).SpeciesCount = P - i;
    else
        Population(i).SpeciesCount = 0;
    end
end
return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [lambda, mu] = GetLambdaMu(Population, I, E, P)
% Calculate immigration rate and emigration rate for each species count
% lambda(i) is the immigration rate for individual i
% mu(i) is the emigration rate for individual i
for i = 1 : length(Population)
    lambda(i) = I * (1 - Population(i).SpeciesCount / P);
    mu(i) = E * Population(i).SpeciesCount / P;
end
return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Population] = ClearDups(Population, MaxParValue, MinParValue)
% Make sure there are no duplicate individuals in the population.
% This logic does not make 100% sure that no duplicates exist, but any duplicates that are found are
% randomly mutated, so there should be a good chance that there are no duplicates after this procedure.
for i = 1 : length(Population)
    Chrom1 = sort(Population(i).chrom);
    for j = i+1 : length(Population)
        Chrom2 = sort(Population(j).chrom);
        if isequal(Chrom1, Chrom2)
            parnum = ceil(length(Population(j).chrom) * rand);
            Population(j).chrom(parnum) = floor(MinParValue + (MaxParValue - MinParValue + 1) * rand);
        end
    end
end
return;