%% dynamics_pendulum.m
% *Summary:* Implements ths ODE for simulating the pendulum dynamics, where 
% an input torque f can be applied  
%
%    function dz = dynamics_pendulum(t,z,u)
%
%
% *Input arguments:*
%
%		t     current time step (called from ODE solver)
%   z     state                                                    [2 x 1]
%   u     (optional): torque f(t) applied to pendulum
%
% *Output arguments:*
%   
%   dz    if 3 input arguments:      state derivative wrt time
%
%   Note: It is assumed that the state variables are of the following order:
%         dtheta:  [rad/s] angular velocity of pendulum
%         theta:   [rad]   angle of pendulum
%
% A detailed derivation of the dynamics can be found in:
%
% M.P. Deisenroth: 
% Efficient Reinforcement Learning Using Gaussian Processes, Appendix C, 
% KIT Scientific Publishing, 2010.
%
%
% Copyright (C) 2008-2013 by
% Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
%
% Last modified: 2013-03-18

function dz = dynamics_pendulum(t,z,u)
%% Code

l = 1;    % [m]        length of pendulum
m = 1;    % [kg]       mass of pendulum
g = 9.82; % [m/s^2]    acceleration of gravity
b = 0.01; % [s*Nm/rad] friction coefficient

dz = zeros(2,1);
dz(1) = ( u(t) - b*z(1) - m*g*l*sin(z(2))/2 ) / (m*l^2/3);
dz(2) = z(1);