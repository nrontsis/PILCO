%% settings_pendulum.m
% *Summary:* Script set up the pendulum scenario
%
% Copyright (C) 2008-2013 by 
% Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
%
% Last modified: 2013-05-24
%
%% High-Level Steps
% # Define state and important indices
% # Set up scenario
% # Set up the plant structure
% # Set up the policy structure
% # Set up the cost structure
% # Set up the GP dynamics model structure
% # Parameters for policy optimization
% # Plotting verbosity
% # Some array initializations

%% Code

warning('off','all'); format short; format compact; 

% include some paths
try
  rd = '../pilcoV0.9/';
  addpath([rd 'base'],[rd 'util'],[rd 'gp'],[rd 'control'],[rd 'loss']);
catch
end

rand('seed',5); randn('seed',13); 



% 1. Define state and important indices

% 1a. Full state representation (including all augmentations)
%  1  dtheta        angular velocity of inner pendulum
%  2  theta2        angle inner pendulum
%  3  sin(theta)    complex representation ...
%  4  cos(theta)    ... of angle of pendulum
%  5  u             torque applied to the pendulum

% 1b. Important indices
% odei  indicies for the ode solver
% augi  indicies for variables augmented to the ode variables
% dyno  indicies for the output from the dynamics model and indicies to loss
% angi  indicies for variables treated as angles (using sin/cos representation)
% dyni  indicies for inputs to the dynamics model
% poli  indicies for variables that serve as inputs to the policy
% difi  indicies for training targets that are differences (rather than values)

odei = [1 2];               % varibles for the ODE solver
augi = [];                  % variables to be augmented
dyno = [1 2];               % variables to be predicted (and known to loss)
angi = [2];                 % angle variables
dyni = [1 3 4];             % variables that serve as inputs to the dynamics GP
poli = [1 3 4];             % variables that serve as inputs to the policy
difi = [1 2];               % variables that are learned via differences

% 2. Set up the scenario
dt = 0.1;                      % [s] sampling time
T = 4;                         % [s] prediction time
H = ceil(T/dt);                % prediction steps (optimization horizon)
mu0 = [0 0]';                  % initial state mean
S0 = 0.01*eye(2);              % initial state variance
N = 10;                        % number of policy optimizations
J = 1;                         % no. of inital training rollouts (of length H)
K = 1;                         % number of initial states for which we optimize
nc = 20;                       % size of controller training set

% 3. Set up the plant structure
plant.dynamics = @dynamics_pendulum;    % dynamics ODE function
plant.noise = diag([0.1^2 0.01^2]);     % measurement noise
plant.dt = dt;
plant.ctrl = @zoh;                  % controler is zero-order-hold
plant.odei = odei;                  % indices of the varibles for the ODE solver
plant.augi = augi;                  % indices of augmented variables
plant.angi = angi;
plant.poli = poli;
plant.dyno = dyno;
plant.dyni = dyni;
plant.difi = difi;
plant.prop = @propagated;        

% 4. Set up the policy structure
global policy
policy.fcn = @(policy,m,s)conCat(@congp,@gSat,policy,m,s);% controller 
                                                          % representation
policy.maxU = 2.5;                                        % max. amplitude of 
                                                          % torque
[mm ss cc] = gTrig(mu0, S0, plant.angi);                  % represent angles 
mm = [mu0; mm]; cc = S0*cc; ss = [S0 cc; cc' ss];         % in complex plane      
policy.p.inputs = gaussian(mm(poli), ss(poli,poli), nc)'; % init. location of 
                                                          % basis functions
policy.p.targets = 0.1*randn(nc, length(policy.maxU));    % init. policy targets 
                                                          % (close to zero)
policy.p.hyp = log([1 0.7 0.7 1 0.01]');                  % initialize 
                                                          % hyper-parameters



% 5. Set up the cost structure
cost.fcn = @loss_pendulum;                           % cost function
cost.gamma = 1;                                      % discount factor
cost.p = 1.0;                                        % length of pendulum
cost.width = 0.5;                                    % cost function width
cost.expl = 0;                                       % exploration parameter
cost.angle = plant.angi;                             % angle variables in cost
cost.target = [0 pi]';                               % target state


% 6. Set up the GP dynamics model structure
dynmodel.fcn = @gp0d;                % function for GP predictions
dynmodel.train = @train;             % function to train dynamics model
dynmodel.induce = zeros(300,0,1);    % shared inducing inputs (sparse GP)
trainOpt = [300 500];                % defines the max. number of line searches
                                     % when training the GP dynamics models
                                     % trainOpt(1): full GP,
                                     % trainOpt(2): sparse GP (FITC)
                                     
% 7. Parameters for policy optimization
opt.length = 75;                         % max. number of line searches
opt.MFEPLS = 30;                         % max. number of function evaluations
                                         % per line search
opt.verbosity = 1;                       % verbosity: specifies how much 
                                         % information is displayed during
                                         % policy learning. Options: 0-3
opt.method = 'BFGS';                     % optimization algorithm. Options:
                                         % 'BFGS' (default), 'LBFGS', 'CG'


% 8. Plotting verbosity
plotting.verbosity = 1;            % 0: no plots
                                   % 1: some plots
                                   % 2: all plots

% 9. Some initializations
x = []; y = [];
fantasy.mean = cell(1,N); fantasy.std = cell(1,N);
realCost = cell(1,N); M = cell(N,1); Sigma = cell(N,1);

% Things copied from pendulum_learn
basename = 'pendulum_';       % filename used for saving data
mu0Sim(odei,:) = mu0; S0Sim(odei,odei) = S0;
mu0Sim = mu0Sim(dyno); S0Sim = S0Sim(dyno,dyno);